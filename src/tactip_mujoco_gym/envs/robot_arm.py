from .base import TactileGymEnv, TaxelMixin
import numpy as np
import mujoco                                   # official package only (no dm_control "mujoco" shadowing)
from mujoco import mj_step, mj_resetData
from dm_control.utils import inverse_kinematics


# =============================================================================
# Base arm environment
# =============================================================================
class RobotArmEnv(TaxelMixin):
    # ---- control ----
    FRAME_SKIP   = 10        # physics steps per env step (recorder reads this too)
    ACTION_SCALE = 0.005     # metres per env step at |action| = 1
    IK_STEPS     = 20        # IK iterations per env step (warm-started, so small is fine)
    ROT_WEIGHT   = 0.1       # soft orientation constraint -> fewer unreachable targets

    # reachable box for commanded tip positions; override per task
    WS_LO = np.full(3, -np.inf)
    WS_HI = np.full(3,  np.inf)

    # ---- shared push-reward weights ----
    W_APPROACH = 10.0        # must dominate the -0.01 time penalty
    W_PUSH     = 20.0
    R_CONTACT  = 2.0         # one-off, first touch of the episode
    R_KEEP     = 0.02        # per step while touching
    W_ALIGN    = 0.05        # pushing direction vs goal direction

    def __init__(self, xml_subpath=["assets", "tactip_arm.xml"], obs_dim=8, action_dim=3):
        self._ik_fail = False
        super().__init__(xml_subpath=xml_subpath, obs_dim=obs_dim, action_dim=action_dim)
        ee = self.model.site("ee_site")
        self._arm_root = int(self.model.body_rootid[self.model.site_bodyid[ee.id]])
        self.current_position = self.data.site_xpos[ee.id].copy()
        self.set_arm(iter=10, ik_steps=500)          # first move can be large

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.step_count = 0
        self.set_arm(iter=250, ik_steps=500)
        return self._get_obs(), {}

    def step(self, action):
        mujoco.mj_forward(self.model, self.data)
        action = np.clip(np.asarray(action, dtype=float).ravel()[:3], -1.0, 1.0)

        # always step the physics (so objects keep moving even for a zero action)
        self.current_position = np.clip(
            self.current_position + action * self.ACTION_SCALE, self.WS_LO, self.WS_HI)
        self.set_arm(self.current_position, iter=self.FRAME_SKIP, ik_steps=self.IK_STEPS)

        self.step_count += 1
        obs        = self._get_obs()
        reward     = self._reward()
        terminated = self._done()
        truncated  = self.step_count >= self.max_steps

        # resync commanded position with where the tip really is
        self.current_position = self.data.site_xpos[self.model.site("ee_site").id].copy()
        info = {"ik_fail": self._ik_fail, "is_success": self._is_success()}
        return obs, reward, terminated, truncated, info

    # ---------- arm control ----------
    def set_arm(self, default=[0.3, 0.3, 0.15], iter=10, ik_steps=150):
        """Solve IK for `ik_steps` iterations, then run `iter` physics steps toward the result."""
        self.move_gripper_to(default, iter=ik_steps)
        for _ in range(iter):
            self.data.ctrl[:6] = self.targets[:6]
            mj_step(self.model, self.data)

    def move_gripper_to(self, fingertip_coords, iter=150):
        self.physics.data.qpos[:] = self.data.qpos[:]
        self.physics.data.qvel[:] = self.data.qvel[:]
        self.physics.forward()
        result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name="ee_site",
            joint_names=["joint" + str(i) for i in range(1, 7)],
            target_pos=fingertip_coords,
            target_quat=[1.0, 0.0, 0.0, 0.0],
            rot_weight=self.ROT_WEIGHT,
            tol=1e-3,
            max_steps=iter,
        )
        self._ik_fail = not result.success
        self.targets = result.qpos[:7]

    # ---------- hooks ----------
    def _is_success(self):
        return False

    def _reward(self):
        return 0.0

    # ---------- shared reward helpers ----------
    def _touching(self, obj_body):
        """True if any geom of the arm's kinematic tree touches the object's body."""
        m, d = self.model, self.data
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == obj_body:
                other = b2
            elif b2 == obj_body:
                other = b1
            else:
                continue
            if m.body_rootid[other] == self._arm_root:
                return True
        return False

    def _push_reward(self, tip, obj, goal_xy, push_pt, obj_body):
        """Stage 1: reach the push point. Stage 2 (touching): push the object to goal_xy."""
        touching = self._touching(obj_body)
        goal_d = float(np.linalg.norm(obj[:2] - goal_xy))
        tip_d  = float(np.linalg.norm(tip - push_pt))

        r = -0.01
        if not touching:
            r += self.W_APPROACH * (self._prev_tip - tip_d)
        else:
            if not self._made_contact:
                r += self.R_CONTACT
                self._made_contact = True
            r += self.R_KEEP
            to_goal = goal_xy - obj[:2]
            to_obj  = obj[:2] - tip[:2]
            to_goal = to_goal / (np.linalg.norm(to_goal) + 1e-8)
            to_obj  = to_obj / (np.linalg.norm(to_obj) + 1e-8)
            r += self.W_ALIGN * float(to_goal @ to_obj)      # +1 = pushing the right way
        # progress counts whenever the object moves (it can only move if pushed)
        r += self.W_PUSH * (self._prev_goal - goal_d)

        self._prev_tip, self._prev_goal = tip_d, goal_d      # always update: no stale deltas
        return r


# =============================================================================
# Peg: push a cylinder into a hole
# =============================================================================
class Peg(RobotArmEnv):
    HOLE_XY      = np.array([0.45, 0.0])
    HOLE_HALF    = 0.03
    PLATFORM_TOP = 0.05
    CYL_R        = 0.02
    CYL_HALF_H   = 0.03

    TILT_LIMIT = 0.87        # cos(~30 deg); below this the cylinder counts as fallen
    R_SUCCESS  = 100.0
    R_FALL     = -50.0

    WS_LO = np.array([0.20, -0.20, 0.06])
    WS_HI = np.array([0.55,  0.20, 0.25])

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "peg_worldbody_tx.xml"],
            obs_dim=3 + 5,
            action_dim=3,
        )
        self._cyl_body = self.model.body("peg_cyl").id
        self._cyl_qadr = self.model.joint("peg_free").qposadr[0]
        self._cyl_dadr = self.model.joint("peg_free").dofadr[0]
        self._ee_site  = self.model.site("ee_site").id
        self._prev_goal = 0.0
        self._prev_tip = 0.0
        self._made_contact = False

    # ---------- helpers ----------
    def _tip_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _cyl_pos(self):
        return self.data.xpos[self._cyl_body].copy()

    def _upright(self):
        """World-z component of the cylinder's z axis (1 = upright, 0 = on its side)."""
        return self.data.xmat[self._cyl_body][8]

    def _push_point(self, cyl, tip_radius=0.01, margin=0.01):
        """Point on the far side of the cylinder from the hole, at cylinder height."""
        d = cyl[:2] - self.HOLE_XY
        d = d / (np.linalg.norm(d) + 1e-8)
        xy = cyl[:2] + d * (self.CYL_R + tip_radius + margin)
        return np.array([xy[0], xy[1], cyl[2]])

    def _in_hole(self, cyl):
        return bool(np.all(np.abs(cyl[:2] - self.HOLE_XY) < self.HOLE_HALF)
                    and cyl[2] < self.PLATFORM_TOP + 0.5 * self.CYL_HALF_H)

    def _fallen(self, cyl):
        tipped = self._upright() < self.TILT_LIMIT
        off_platform = (cyl[2] < self.PLATFORM_TOP) and not self._in_hole(cyl)
        return bool(tipped or off_platform)

    def _is_success(self):
        return self._in_hole(self._cyl_pos())

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                      # resets data, moves arm to default

        rng = getattr(self, "np_random", None) or np.random.default_rng(seed)
        x = rng.uniform(0.33, 0.37)
        y = rng.uniform(-0.02, 0.02)
        z = self.PLATFORM_TOP + self.CYL_HALF_H + 1e-3

        a = self._cyl_qadr
        self.data.qpos[a:a + 3] = [x, y, z]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]    # upright
        self.data.qvel[self._cyl_dadr:self._cyl_dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        target = self._cyl_pos()
        target[2] += (z / 3) * 2                      # hover above the cylinder
        self.set_arm(target, iter=600, ik_steps=500)
        self.current_position = self._tip_pos()

        cyl, tip = self._cyl_pos(), self._tip_pos()
        self._prev_goal = float(np.linalg.norm(cyl[:2] - self.HOLE_XY))
        self._prev_tip  = float(np.linalg.norm(tip - self._push_point(cyl)))
        self._made_contact = False
        return self._get_obs(), {}

    def _get_obs(self):
        tip = self._tip_pos()
        cyl = self._cyl_pos()
        # 3 tip + 3 cylinder + 2 (hole - cylinder, xy) = 8
        return np.concatenate([tip, cyl, self.HOLE_XY - cyl[:2]]).astype(np.float32)

    def _reward(self):
        tip, cyl = self._tip_pos(), self._cyl_pos()
        if self._in_hole(cyl):                        # success first: tilting while dropping in is OK
            return self.R_SUCCESS
        if self._fallen(cyl):
            return self.R_FALL
        return self._push_reward(tip, cyl, self.HOLE_XY,
                                 self._push_point(cyl), self._cyl_body)

    def _done(self):
        cyl = self._cyl_pos()
        return bool(self._in_hole(cyl) or self._fallen(cyl))


# =============================================================================
# Edge: follow the edge of a randomly rotated block (reward unchanged)
# =============================================================================
class Edge(RobotArmEnv):
    def __init__(self):
        self.previous_progress = 0
        super().__init__(
            xml_subpath=["assets", "edge_worldbody_tx.xml"],
            obs_dim=213,
            action_dim=3,
        )

    def random_edge_point(self):
        geom_id = self.model.geom("block_geom").id
        geom_pos = self.data.geom_xpos[geom_id]
        geom_rot = self.data.geom_xmat[geom_id].reshape(3, 3)
        hx, hy, hz = self.model.geom_size[geom_id]

        edge = np.random.randint(4)
        if edge == 0:       # +x edge
            x, y = hx, np.random.uniform(-hy, hy)
        elif edge == 1:     # -x edge
            x, y = -hx, np.random.uniform(-hy, hy)
        elif edge == 2:     # +y edge
            x, y = np.random.uniform(-hx, hx), hy
        else:               # -y edge
            x, y = np.random.uniform(-hx, hx), -hy

        local_point = np.array([x, y, hz])            # top surface
        return geom_pos + geom_rot @ local_point

    def randomize_block_rotation(self):
        angle = np.random.uniform(-np.pi, np.pi)
        self.quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        body_id = self.model.body("block").id
        self.model.body_quat[body_id] = self.quat
        mujoco.mj_forward(self.model, self.data)
        self.positions = self.get_top_edge("block_geom", step=0.01)
        self.get_first_point()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_progress = 0
        self.randomize_block_rotation()
        target = self.random_edge_point()
        target[2] += 0.005
        self.set_arm(target, iter=100, ik_steps=500)
        self.current_position = self.data.site_xpos[self.model.site("ee_site").id].copy()
        return self._get_obs(), {}

    def closest_edge_point(self, point):
        block_pos = self.data.geom_xpos[self.model.geom("block_geom").id]
        half_size = self.model.geom_size[self.model.geom("block_geom").id]
        return np.clip(point, block_pos - half_size, block_pos + half_size)

    def get_top_edge(self, geom_name, step):
        geom = self.model.geom(geom_name)
        width, length, height = 2 * geom.size
        position = self.data.geom_xpos[geom.id]
        rotation = self.data.geom_xmat[geom.id].reshape(3, 3)

        corners = np.array([
            [-width / 2, -length / 2, height / 2],
            [ width / 2, -length / 2, height / 2],
            [ width / 2,  length / 2, height / 2],
            [-width / 2,  length / 2, height / 2],
        ])
        world_corners = corners @ rotation.T + position

        points = []
        for i in range(4):
            start = world_corners[i]
            end = world_corners[(i + 1) % 4]
            edge = end - start
            edge_length = np.linalg.norm(edge)
            distances = np.arange(0, edge_length, step)
            points.append(start + (distances[:, None] / edge_length) * edge)

        points = np.vstack(points)
        return np.vstack([points, world_corners[0]])  # close the loop

    def get_first_point(self):
        tip = self.data.site_xpos[self.model.site("ee_site").id]
        dist = np.array([np.linalg.norm(tip[:2] - p[:2]) for p in self.positions])
        self.current_target = int(np.argmax(dist))

    def edge_progress(self, tip):
        edge_error = np.linalg.norm(tip[:2] - self.positions[self.current_target][:2])
        if edge_error < 0.01:
            self.current_target += 1
        if self.current_target >= len(self.positions):
            self.current_target = 0
        return 1 / (edge_error + 1e-6)

    def _reward(self):
        tip = self.data.site_xpos[self.model.site("ee_site").id]
        edge_point = self.closest_edge_point(tip)
        edge_error = np.linalg.norm(tip[:2] - edge_point[:2])
        progress = self.edge_progress(tip)

        delta_progress = progress - self.previous_progress
        self.previous_progress = progress

        force = 0.2                                   # TODO: self.get_contact_force()
        contact_penalty = 5 if force < 0.1 else 0

        reward = -2.0 * edge_error + 5.0 * delta_progress - contact_penalty
        r_low = 0
        if tip[2] < edge_point[2]:
            r_low = (tip[2] - edge_point[2]) * 5
        return reward + r_low

    def _get_obs(self):
        obs = self.get_nodes().flatten()
        if self.events:
            obs = np.concatenate([obs, self._event_obs()])
        return obs.astype(np.float32)


# =============================================================================
# Shape insertion: push the right shape into its matching hole
# =============================================================================
PLATE_TOP    = 0.05
PLATE_HALF_T = 0.01            # plate spans z = 0.03 .. 0.05
OBJ_HALF_H   = 0.03
CLEARANCE    = 0.004           # hole is this much bigger than the object, per side
PLATE_X      = (0.28, 0.62)
HOLE_X       = 0.50
HOLE_Y       = [-0.15, -0.05, 0.05, 0.15]   # one hole per shape, in SHAPES order
CELL_HALF    = 0.05            # half-size of the square patch around each hole
STRIP        = 0.0025          # resolution used to cut non-box holes

# sym = rotational symmetry order (0 = circle, any yaw fits)
SHAPES = {
    "circle":    dict(kind="cylinder", r=0.02,           sym=0, bound_r=0.020, rgba="0.9 0.2 0.2 1"),
    "square":    dict(kind="box", hx=0.02, hy=0.02,      sym=4, bound_r=0.029, rgba="0.2 0.7 0.3 1"),
    "rectangle": dict(kind="box", hx=0.03, hy=0.015,     sym=2, bound_r=0.034, rgba="0.2 0.4 0.9 1"),
    "triangle":  dict(kind="prism", R=0.025,             sym=3, bound_r=0.025, rgba="0.9 0.7 0.1 1"),
}
PLATE_ATTR = 'rgba="0.6 0.6 0.6 1" friction="0.8 0.005 0.0001"'


class ShapeInsertion(RobotArmEnv):
    SHAPE_NAMES = list(SHAPES.keys())
    N_SHAPES    = len(SHAPE_NAMES)

    START_TIP  = [0.25, 0.0, 0.085]   # tip behind the spawn zone, at object mid-height
    SPAWN_X    = (0.34, 0.38)
    SPAWN_Y    = (-0.04, 0.04)

    TILT_LIMIT = 0.87                 # cos(~30 deg)
    HOLE_TOL   = 0.03                 # xy distance for "went through the correct hole"
    R_SUCCESS  = 100.0
    R_FALL     = -50.0

    YAW_FADE = 0.10                   # start caring about yaw within 10 cm of the hole
    W_YAW    = 3.0

    WS_LO = np.array([0.15, -0.25, 0.06])
    WS_HI = np.array([0.60,  0.25, 0.25])

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactip_arm_shapes.xml"],
            obs_dim=12 + 3,           # tip3 + obj3 + sin/cos yaw 2 + to-hole 2 + yaw err 1 + onehot 4
            action_dim=3,
        )
        m = self.model
        self._rng = np.random.default_rng()
        self.spawn_yaw = np.pi        # curriculum: set to 0.0 to start with no rotation needed
        self._ee_site = m.site("ee_site").id

        self._body, self._geom, self._qadr, self._dadr = {}, {}, {}, {}
        self._rgba0, self._contype0, self._conaff0 = {}, {}, {}
        for n in self.SHAPE_NAMES:
            self._body[n] = m.body(f"shape_{n}").id
            g = m.geom(f"shape_{n}_geom").id
            self._geom[n] = g
            jid = m.joint(f"shape_{n}_free").id
            self._qadr[n] = m.jnt_qposadr[jid]
            self._dadr[n] = m.jnt_dofadr[jid]
            self._rgba0[n]    = m.geom_rgba[g].copy()
            self._contype0[n] = int(m.geom_contype[g])
            self._conaff0[n]  = int(m.geom_conaffinity[g])

        self.shape = self.SHAPE_NAMES[0]
        self._target_xy = np.array([HOLE_X, HOLE_Y[0]])
        self._prev_goal = self._prev_yaw = self._prev_tip = 0.0
        self._made_contact = False

    # ---------- helpers ----------
    def _tip_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _obj_pos(self):
        return self.data.xpos[self._body[self.shape]].copy()

    def _upright(self):
        return self.data.xmat[self._body[self.shape]][8]

    def _yaw(self):
        R = self.data.xmat[self._body[self.shape]]
        return np.arctan2(R[3], R[0])

    def _yaw_error(self):
        """Signed yaw error to the hole (hole yaw = 0), wrapped by the shape's symmetry."""
        k = SHAPES[self.shape]["sym"]
        if k == 0:
            return 0.0
        period = 2 * np.pi / k
        return (self._yaw() + period / 2) % period - period / 2

    def _push_point(self, obj, tip_radius=0.01, margin=0.01):
        d = obj[:2] - self._target_xy
        d = d / (np.linalg.norm(d) + 1e-8)
        xy = obj[:2] + d * (SHAPES[self.shape]["bound_r"] + tip_radius + margin)
        return np.array([xy[0], xy[1], obj[2]])

    def _show_only(self, name):
        """Hide every shape except `name`: transparent, collision off, gravity compensated."""
        m = self.model
        for n in self.SHAPE_NAMES:
            g, b = self._geom[n], self._body[n]
            on = (n == name)
            m.geom_rgba[g] = self._rgba0[n] if on else np.append(self._rgba0[n][:3], 0.0)
            m.geom_contype[g]     = self._contype0[n] if on else 0
            m.geom_conaffinity[g] = self._conaff0[n] if on else 0
            m.body_gravcomp[b]    = 0.0 if on else 1.0

    def _status(self):
        obj = self._obj_pos()
        dist = np.linalg.norm(obj[:2] - self._target_xy)
        if obj[2] < PLATE_TOP - 0.005:                # dropped below the plate surface
            return "success" if dist < self.HOLE_TOL else "fail"
        # tipping over the target hole's mouth is allowed, anywhere else it's game over
        if self._upright() < self.TILT_LIMIT and dist > 0.05:
            return "fail"
        return None

    def _is_success(self):
        return self._status() == "success"

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                      # resets data, step_count, arm to default
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        name = options.get("shape") or str(self._rng.choice(self.SHAPE_NAMES))
        self.shape = name
        idx = self.SHAPE_NAMES.index(name)
        self._target_xy = np.array([HOLE_X, HOLE_Y[idx]])

        self.set_arm(self.START_TIP, iter=250, ik_steps=500)   # arm first, so it can't sweep the object
        self._show_only(name)

        # park the hidden shapes far away, at rest
        for i, n in enumerate(self.SHAPE_NAMES):
            if n != name:
                a, d = self._qadr[n], self._dadr[n]
                self.data.qpos[a:a + 7] = [1.5, -0.6 + 0.4 * i, 0.5, 1, 0, 0, 0]
                self.data.qvel[d:d + 6] = 0

        # place the chosen shape with a random yaw
        x = self._rng.uniform(*self.SPAWN_X)
        y = self._rng.uniform(*self.SPAWN_Y)
        yaw = self._rng.uniform(-self.spawn_yaw, self.spawn_yaw)
        a, d = self._qadr[name], self._dadr[name]
        self.data.qpos[a:a + 7] = [x, y, PLATE_TOP + OBJ_HALF_H + 1e-3,
                                   np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        self.data.qvel[d:d + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # hover above the object, then hand over to the policy
        target = self._obj_pos()
        target[2] += 0.1
        self.set_arm(target, iter=600, ik_steps=500)
        self.current_position = self._tip_pos()

        obj, tip = self._obj_pos(), self._tip_pos()
        self._prev_goal = float(np.linalg.norm(obj[:2] - self._target_xy))
        self._prev_yaw  = abs(self._yaw_error())
        self._prev_tip  = float(np.linalg.norm(tip - self._push_point(obj)))
        self._made_contact = False
        return self._get_obs(), {"shape": name}

    def _get_obs(self):
        tip = self._tip_pos()
        obj = self._obj_pos()
        yaw = self._yaw()
        onehot = np.zeros(self.N_SHAPES)
        onehot[self.SHAPE_NAMES.index(self.shape)] = 1.0
        obs = np.concatenate([
            tip, obj,
            [np.sin(yaw), np.cos(yaw)],
            self._target_xy - obj[:2],
            [self._yaw_error()],
            onehot,
        ]).astype(np.float32)

        if self.events:
            obs = np.concatenate([obs, self._event_obs()])
        return obs.astype(np.float32)

    def _reward(self):
        status = self._status()
        if status == "success":
            return self.R_SUCCESS
        if status == "fail":
            return self.R_FALL

        tip, obj = self._tip_pos(), self._obj_pos()

        # Stage 3: yaw only matters near the hole
        goal_d  = np.linalg.norm(obj[:2] - self._target_xy)
        yaw_abs = abs(self._yaw_error())
        near    = np.clip(1.0 - goal_d / self.YAW_FADE, 0.0, 1.0)
        r_yaw   = self.W_YAW * near * (self._prev_yaw - yaw_abs)
        self._prev_yaw = yaw_abs

        # Stages 1+2: reach, then push to the correct hole
        r = self._push_reward(tip, obj, self._target_xy,
                              self._push_point(obj), self._body[self.shape])
        return r + r_yaw

    def _done(self):
        return self._status() is not None


# =============================================================================
# Incline: stop a ball rolling down a ramp, then push it up
# =============================================================================
class Incline(RobotArmEnv):
    # ---- geometry ----
    BALL_R = 0.02
    TIP_R  = 0.01            # approx radius of the contact tip
    TIP_H  = 0.02            # tip height above the surface along the normal (= ball centre height)

    # ---- friction randomisation: one level f in [0,1] maps to both coefficients ----
    F_SLIDE = (0.05, 1.0)    # sliding friction, log-interpolated
    F_ROLL  = (0.0, 0.008)   # rolling friction, linear (needs condim=6)
    OBSERVE_FRICTION = False # True = give the policy f (useful for debugging)

    # ---- control ----
    S_MIN, Y_LIM = -0.06, 0.06

    # ---- reward ----
    SUCCESS_MARGIN = 0.04    # success when the ball is this close to the top
    PROGRESS_W = 30.0        # reward per metre of progress up the slope
    R_SUCCESS  = 100.0
    R_FAIL     = -50.0
    HOLD_STEPS = 10          # consecutive steps of stable hold before pushing starts
    HOLD_V     = 0.02        # m/s downhill speed counted as "held"

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "hill_worldbody_tx.xml"],
            obs_dim=3 + 10 if self.OBSERVE_FRICTION else 3 + 9,
            action_dim=2,
        )
        m, d = self.model, self.data
        self._rng = np.random.default_rng()
        self._ee_site   = m.site("ee_site").id
        self._ball_body = m.body("ball").id
        jid = m.joint("ball_free").id
        self._ball_qadr = m.jnt_qposadr[jid]
        self._ball_dadr = m.jnt_dofadr[jid]
        self._ramp_geom = m.geom("ramp_surface").id

        # ramp frame from the model (static body, so compute once)
        mujoco.mj_forward(m, d)
        ramp = m.body("ramp").id
        R = d.xmat[ramp].reshape(3, 3)
        self._S = d.xpos[ramp].copy()      # bottom edge of the surface
        self._D = R[:, 0].copy()           # up-slope direction
        self._N = R[:, 2].copy()           # surface normal
        self._L = 2.0 * m.geom_size[self._ramp_geom][0]   # surface length

        self.friction_level = 0.5
        self.S_MAX = self._L - 0.03
        self._cmd_s = self._cmd_y = 0.0
        self._prev_s = self._prev_tip = 0.0
        self._stage = 0
        self._hold = 0
        self._made_contact = False

    # ---------- helpers ----------
    def _tip_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _ball_pos(self):
        return self.data.xpos[self._ball_body].copy()

    def _slope_coords(self, p):
        """(s, h, lat): distance up the slope, height above surface, sideways offset."""
        rel = p - self._S
        return float(rel @ self._D), float(rel @ self._N), float(rel[1])

    def _tip_target(self, s, y):
        t = self._S + s * self._D + self.TIP_H * self._N
        t[1] += y
        return t

    def _set_friction(self, f):
        self.friction_level = float(f)
        slide = self.F_SLIDE[0] * (self.F_SLIDE[1] / self.F_SLIDE[0]) ** f
        roll  = self.F_ROLL[0] + f * (self.F_ROLL[1] - self.F_ROLL[0])
        self.model.geom_friction[self._ramp_geom] = [slide, 0.005, roll]
        g = 0.8 - 0.5 * f                                    # darker ramp = grippier (visual only)
        self.model.geom_rgba[self._ramp_geom] = [g, g, g, 1.0]

    def _status(self):
        s, h, lat = self._slope_coords(self._ball_pos())
        if s >= self._L - self.SUCCESS_MARGIN:
            return "success"
        if s < -0.04 or abs(lat) > 0.10 or h > 0.08 or h < -0.02:
            return "fail"                                    # off the bottom / side / popped off
        return None

    def _is_success(self):
        return self._status() == "success"

    def _push_dist(self, tip, ball):
        """Distance (in the slope plane) from the tip to the point just below the ball."""
        s_b, _, lat_b = self._slope_coords(ball)
        s_t, _, lat_t = self._slope_coords(tip)
        s_goal = s_b - (self.BALL_R + self.TIP_R + 0.01)
        return float(np.hypot(s_t - s_goal, lat_t - lat_b))

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        f = options["friction"] if "friction" in options else self._rng.uniform(0.0, 1.0)
        self._set_friction(f)

        a, d = self._ball_qadr, self._ball_dadr

        # 1. park the ball away from the arm's workspace
        self.data.qpos[a:a + 7] = [1.0, 1.0, 0.5, 1, 0, 0, 0]
        self.data.qvel[d:d + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # 2. put the arm downhill of where the ball will start
        s_tip = self._rng.uniform(0.0, 0.5) * self._L
        y0    = self._rng.uniform(-0.02, 0.02)
        gap   = 0.003
        self.set_arm(self._tip_target(s_tip - (self.BALL_R + self.TIP_R + gap), y0),
                     iter=600, ik_steps=500)

        # 3. place the ball uphill of the tip so it rolls toward it
        s_ball = self._rng.uniform(0.5, 0.85) * self._L
        y_ball = self._rng.uniform(-0.02, 0.02)
        centre = self._S + s_ball * self._D + (self.BALL_R + 1e-3) * self._N
        centre[1] += y_ball
        self.data.qpos[a:a + 7] = [*centre, 1, 0, 0, 0]
        self.data.qvel[d:d + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # 4. sync commands to where the tip really is, reset episode state
        s_t, _, lat_t = self._slope_coords(self._tip_pos())
        self._cmd_s, self._cmd_y = s_t, lat_t
        self.current_position = self._tip_pos()

        tip, ball = self._tip_pos(), self._ball_pos()
        self._prev_s   = self._slope_coords(ball)[0]
        self._prev_tip = self._push_dist(tip, ball)
        self._stage, self._hold, self._made_contact = 0, 0, False
        return self._get_obs(), {"friction": self.friction_level}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel()[:2], -1.0, 1.0)
        self._cmd_s = float(np.clip(self._cmd_s + a[0] * self.ACTION_SCALE, self.S_MIN, self.S_MAX))
        self._cmd_y = float(np.clip(self._cmd_y + a[1] * self.ACTION_SCALE, -self.Y_LIM, self.Y_LIM))

        self.move_gripper_to(self._tip_target(self._cmd_s, self._cmd_y), iter=self.IK_STEPS)
        for _ in range(self.FRAME_SKIP):
            self.data.ctrl[:6] = self.targets[:6]
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs        = self._get_obs()
        reward     = self._reward()
        terminated = self._done()
        truncated  = self.step_count >= self.max_steps
        self.current_position = self._tip_pos()
        info = {"friction": self.friction_level,
                "ball_s": self._slope_coords(self._ball_pos())[0],
                "stage": self._stage,
                "ik_fail": self._ik_fail,
                "is_success": self._status() == "success"}
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        tip  = self._tip_pos()
        ball = self._ball_pos()
        vel  = self.data.qvel[self._ball_dadr:self._ball_dadr + 3]   # world-frame linear velocity
        s_b, _, lat_b = self._slope_coords(ball)
        s_t, _, lat_t = self._slope_coords(tip)
        obs = [tip, ball, vel, [s_b / self._L], [s_t - s_b, lat_t - lat_b]]
        if self.OBSERVE_FRICTION:
            obs.append([self.friction_level])
        return np.concatenate(obs).astype(np.float32)

    def _reward(self):
        status = self._status()
        if status == "success":
            return self.R_SUCCESS
        if status == "fail":
            return self.R_FAIL

        tip, ball = self._tip_pos(), self._ball_pos()
        s, _, lat_b = self._slope_coords(ball)
        _, _, lat_t = self._slope_coords(tip)
        tip_d  = self._push_dist(tip, ball)
        v_up   = float(self.data.qvel[self._ball_dadr:self._ball_dadr + 3] @ self._D)
        v_down = max(0.0, -v_up)
        ds     = s - self._prev_s
        touching = self._touching(self._ball_body)

        r = -0.01
        if not touching:
            # STAGE 0: reach the ball
            self._stage, self._hold = 0, 0
            r += self.W_APPROACH * (self._prev_tip - tip_d)
            r += self.PROGRESS_W * min(0.0, ds)            # rolling downhill is bad
        else:
            if not self._made_contact:
                r += self.R_CONTACT
                self._made_contact = True
            if self._stage == 0:
                self._stage = 1

            if self._stage == 1:
                # STAGE 1: stop it falling; no reward for pushing up yet
                r += 0.05 - 2.0 * v_down
                r += self.PROGRESS_W * min(0.0, ds)
                self._hold = self._hold + 1 if v_down < self.HOLD_V else 0
                if self._hold >= self.HOLD_STEPS:
                    self._stage = 2
                    r += 5.0                                # bonus for a stable hold
            else:
                # STAGE 2: push up the slope
                r += self.PROGRESS_W * ds
                r += self.R_KEEP - 0.5 * abs(lat_t - lat_b) # stay centred behind the ball
                if v_down > 0.05:                           # ball slipped past the tip
                    self._stage, self._hold = 1, 0

        self._prev_s, self._prev_tip = s, tip_d
        return r

    def _done(self):
        return self._status() is not None