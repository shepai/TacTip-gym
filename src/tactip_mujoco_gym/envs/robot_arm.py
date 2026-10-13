from .base import TactileGymEnv, TaxelMixin
import numpy as np
from mujoco import mj_step, mj_resetData
import mujoco 
from dm_control import mujoco
from dm_control.utils import inverse_kinematics

class RobotArmEnv(TaxelMixin):
    def __init__(self, xml_subpath=["assets", "tactip_arm.xml"],
                obs_dim=8,
                action_dim=3):
        super().__init__(
            xml_subpath=xml_subpath,
            obs_dim=obs_dim,
            action_dim=action_dim
        )
        tip = self.data.site_xpos[
            self.model.site("ee_site").id
        ]
        self.current_position=tip.copy()
        self.set_arm()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.step_count = 0
        self.set_arm()
        return self._get_obs(), {}
    def set_arm(self,default=[0.3,0.3,0.15]):
        print("MOVING ARM TO:", default)
        self.move_gripper_to(default,iter=750)
        for i in range(250):       
            self.data.ctrl[:7] = self.targets[:-1]
            mj_step(self.model, self.data)
            """mujoco.mj_forward(
                self.model,
                self.data
            )"""
    def _reward(self):
        return np.random.randint(0,1)
    def step(self,action):
        mujoco.mj_forward(self.model, self.data)
        if not(action[0]==0 and action[1]==0 and action[2]==0):
            self.current_position+=action*0.5 #treat as vector instead of position
            action[2]=0 # make z axis not move to reduce complexity 
            self.move_gripper_to(self.current_position)
            self.data.ctrl[:6] = self.targets[:-1]
            mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        tip = self.data.site_xpos[
            self.model.site("ee_site").id
        ]
        self.current_position=tip.copy()
        return obs, reward, terminated, truncated, {}
    def move_gripper_to(self, fingertip_coords,iter=150):
        fixed_orientation = [1.0, 0.0, 0.0, 0.0] 
        self.physics.data.qpos[:] = self.data.qpos[:]
        self.physics.data.qvel[:] = self.data.qvel[:]
        self.physics.forward()
        result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name="ee_site",
            joint_names=["joint"+str(i) for i in range(1,7)],
            target_pos=fingertip_coords,
            target_quat=fixed_orientation,
            max_steps=iter
        )

        # ONLY update sim state once
        self.targets = result.qpos[:7]
        #mj.mj_forward(self.model, self.data)

class Peg(RobotArmEnv):
    # ---- task constants (match the XML) ----
    HOLE_XY      = np.array([0.45, 0.0])
    HOLE_HALF    = 0.03
    PLATFORM_TOP = 0.05
    CYL_R        = 0.02
    CYL_HALF_H   = 0.03

    TILT_LIMIT = 0.87     # cos(~30 deg); below this the cylinder counts as fallen
    R_SUCCESS  = 100.0
    R_FALL     = -50.0

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "peg_worldbody_tx.xml"],
            obs_dim=213+5,
            action_dim=3
        )
        self._cyl_body = self.model.body("peg_cyl").id
        self._cyl_qadr = self.model.joint("peg_free").qposadr[0]
        self._cyl_dadr = self.model.joint("peg_free").dofadr[0]
        self._ee_site  = self.model.site("ee_site").id
        self._prev_goal_dist = 0.0
        self._prev_tip_dist = 0.0

    # ---------- helpers ----------
    def _tip_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _cyl_pos(self):
        return self.data.xpos[self._cyl_body].copy()

    def _upright(self):
        """World-z component of the cylinder's own z axis (1 = upright, 0 = on its side)."""
        return self.data.xmat[self._cyl_body][8]

    def _push_point(self, cyl, tip_radius=0.01, margin=0.01):
        """Point on the far side of the cylinder from the hole, at cylinder height."""
        d = cyl[:2] - self.HOLE_XY
        d = d / (np.linalg.norm(d) + 1e-8)
        xy = cyl[:2] + d * (self.CYL_R + tip_radius + margin)
        return np.array([xy[0], xy[1], cyl[2]])

    def _in_hole(self, cyl):
        return (np.all(np.abs(cyl[:2] - self.HOLE_XY) < self.HOLE_HALF)
                and cyl[2] < self.PLATFORM_TOP + 0.5 * self.CYL_HALF_H)

    def _fallen(self, cyl):
        tipped = self._upright() < self.TILT_LIMIT
        off_platform = (cyl[2] < self.PLATFORM_TOP) and not self._in_hole(cyl)
        return tipped or off_platform

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)   # mj_resetData, step_count=0, set_arm()

        # random cylinder start on the platform, on the near side of the hole
        rng = getattr(self, "np_random", None) or np.random.default_rng(seed)
        x = rng.uniform(0.33, 0.37)
        y = rng.uniform(-0.02, 0.02)
        z = self.PLATFORM_TOP + self.CYL_HALF_H + 1e-3

        a = self._cyl_qadr
        self.data.qpos[a:a + 3] = [x, y, z]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]          # upright
        self.data.qvel[self._cyl_dadr:self._cyl_dadr + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        # let the cylinder settle while the arm holds its pose
        for _ in range(50):
            self.data.ctrl[:6] = self.targets[:-1]
            mujoco.mj_step(self.model, self.data)

        # your base step() integrates from current_position, so resync it
        self.current_position = self._tip_pos()

        cyl, tip = self._cyl_pos(), self._tip_pos()
        self._prev_goal_dist = np.linalg.norm(cyl[:2] - self.HOLE_XY)
        self._prev_tip_dist  = np.linalg.norm(tip - self._push_point(cyl))
        return self._get_obs(), {}

    def _get_obs(self):
        tip = self.get_nodes().flatten()
        cyl = self._cyl_pos()
        # 3 tip + 3 cylinder + 2 (hole - cylinder, xy) = 8
        return np.concatenate([tip, cyl, self.HOLE_XY - cyl[:2]]).astype(np.float32)

    # called by base step() as self._reward()
    def _reward(self):
        tip, cyl = self._tip_pos(), self._cyl_pos()

        # success first: the cylinder tilts as it drops in, which must not count as a fall
        if self._in_hole(cyl):
            return self.R_SUCCESS
        if self._fallen(cyl):
            return self.R_FALL

        goal_dist = np.linalg.norm(cyl[:2] - self.HOLE_XY)
        tip_dist  = np.linalg.norm(tip - self._push_point(cyl))

        r_push     = 10.0 * (self._prev_goal_dist - goal_dist)   # main signal
        r_approach = 1.0  * (self._prev_tip_dist - tip_dist)     # weak, gets tip to cylinder
        r_step     = -0.01

        self._prev_goal_dist, self._prev_tip_dist = goal_dist, tip_dist
        return r_push + r_approach + r_step

    # called by base step() as self._done()
    def _done(self):
        cyl = self._cyl_pos()
        return bool(self._in_hole(cyl) or self._fallen(cyl))
class Edge(RobotArmEnv):
    def __init__(self):
        self.previous_progress=0
        super().__init__(
            xml_subpath=["assets", "edge_worldbody_tx.xml"],
            obs_dim=213,
            action_dim=3
        )
        #self.set_arm()
    def random_edge_point(self):
        geom_id = self.model.geom("block_geom").id

        # World pose of the geom
        geom_pos = self.data.geom_xpos[geom_id]
        geom_rot = self.data.geom_xmat[geom_id].reshape(3, 3)

        # Half extents in the geom's local frame
        hx, hy, hz = self.model.geom_size[geom_id]

        # Choose one of the four top-face edges (local coordinates)
        edge = np.random.randint(4)

        if edge == 0:       # +x edge
            x = hx
            y = np.random.uniform(-hy, hy)
        elif edge == 1:     # -x edge
            x = -hx
            y = np.random.uniform(-hy, hy)
        elif edge == 2:     # +y edge
            x = np.random.uniform(-hx, hx)
            y = hy
        else:               # -y edge
            x = np.random.uniform(-hx, hx)
            y = -hy

        z = hz  # top surface

        # Local point on the block
        local_point = np.array([x, y, z])

        # Convert block-local coordinate -> world coordinate
        world_point = geom_pos + geom_rot @ local_point

        return world_point
    def randomize_block_rotation(self):
        angle = np.random.uniform(-np.pi, np.pi)

        self.quat = np.array([
            np.cos(angle/2),
            0,
            0,
            np.sin(angle/2)
        ])
        geom = self.model.geom("block_geom")
        body_id = self.model.body("block").id

        self.model.body_quat[body_id] = self.quat

        mujoco.mj_forward(
            self.model,
            self.data
        )
        self.positions=self.get_top_edge("block_geom", step=0.01)
        self.get_first_point()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.previous_progress=0
        self.randomize_block_rotation()
        target = self.random_edge_point()
        target[2] += 0.005
        self.set_arm(default=target)
        return self._get_obs(), {}
    def closest_edge_point(self, point):
        # world coordinates
        block_pos = self.data.geom_xpos[self.model.geom("block_geom").id]
        # half-lengths
        half_size = self.model.geom_size[self.model.geom("block_geom").id]
        # clamp each coordinate
        return np.clip(
            point,
            block_pos - half_size,
            block_pos + half_size
        )
    def get_top_edge(self, geom_name, step):
        geom = self.model.geom(geom_name)
        geom_id = geom.id

        # MuJoCo box dimensions are half-extents
        width, length, height = 2 * geom.size

        # World-space position
        position = self.data.geom_xpos[geom_id]

        # World-space rotation matrix
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)

        # Local top corners
        corners = np.array([
            [-width / 2, -length / 2,  height / 2],
            [ width / 2, -length / 2,  height / 2],
            [ width / 2,  length / 2,  height / 2],
            [-width / 2,  length / 2,  height / 2],
        ])

        # Transform into world coordinates
        world_corners = corners @ rotation.T + position

        points = []

        for i in range(4):

            start = world_corners[i]
            end = world_corners[(i + 1) % 4]

            edge = end - start
            edge_length = np.linalg.norm(edge)

            distances = np.arange(0, edge_length, step)

            edge_points = (
                start +
                (distances[:, None] / edge_length) * edge
            )

            points.append(edge_points)

        points = np.vstack(points)

        # Close the loop
        points = np.vstack([
            points,
            world_corners[0]
        ])
        """import matplotlib.pyplot as plt 
        plt.scatter(points[:,0],points[:,1])
        for i in range(len(points)):
            plt.text(points[i, 0], points[i, 1], str(i))
        plt.show()"""
        return points
    def get_first_point(self):
        tip = self.data.site_xpos[
                    self.model.site("ee_site").id
                ]
        dist=[]
        tip = self.data.site_xpos[
                    self.model.site("ee_site").id
                ]
        for i in range(len(self.positions)):
            dist.append(np.linalg.norm(
            tip[:2] - self.positions[i][:2]
        ))
        dist=np.array(dist)
        self.current_target=np.argmax(dist)
    def edge_progress(self, tip):
        geom_id = self.model.geom("block_geom").id
        center = self.data.geom_xpos[geom_id]
        hx, hy, hz = self.model.geom_size[geom_id]
        edge_error = np.linalg.norm(
                    tip[:2] - self.positions[self.current_target][:2]
                )
        if edge_error<0.01: self.current_target+=1
        if self.current_target>len(self.positions): self.current_target=0
        return 1/edge_error
    def _reward(self): #
        # fingertip position
        tip = self.data.site_xpos[
            self.model.site("ee_site").id
        ]
        edge_point = self.closest_edge_point(tip) 

        edge_error = np.linalg.norm(
            tip[:2] - edge_point[:2]
        )
        progress = self.edge_progress(tip) 

        delta_progress = (
            progress -
            self.previous_progress
        )
        self.previous_progress = progress
        # contact force
        force = 0.2#self.get_contact_force() #TODO implement this
        contact_penalty = 0
        if force < 0.1:
            contact_penalty = 5

        reward = (
            -2.0 * edge_error
            +5.0 * delta_progress
            -contact_penalty
        )
        return reward
    def _get_obs(self):
        obs= self.get_nodes().flatten()
        if self.events:
            obs = np.concatenate([obs, self._event_obs()])   # + n_taxels values

        return obs.astype(np.float32)

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

    START_TIP  = [0.25, 0.0, 0.085]   # CHANGE: tip behind the spawn zone, at object mid-height
    SPAWN_X    = (0.34, 0.38)
    SPAWN_Y    = (-0.04, 0.04)

    TILT_LIMIT = 0.87                 # cos(~30 deg)
    HOLE_TOL   = 0.03                 # xy distance for "went through the correct hole"
    R_SUCCESS  = 100.0
    R_FALL     = -50.0

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactip_arm_shapes.xml"],
            obs_dim=12+213,               # tip3 + obj3 + sin/cos yaw 2 + to-hole 2 + yaw err 1 + onehot 4
            action_dim=3
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
            m.geom_conaffinity[g] = self._conaff0[n]  if on else 0
            m.body_gravcomp[b]    = 0.0 if on else 1.0   # parked bodies don't fall forever

    def _status(self):
        obj = self._obj_pos()
        dist = np.linalg.norm(obj[:2] - self._target_xy)
        if obj[2] < PLATE_TOP - 0.005:                    # dropped below the plate surface
            return "success" if dist < self.HOLE_TOL else "fail"   # wrong hole / off the edge = fail
        # tipping over the target hole's mouth is allowed, anywhere else it's game over
        if self._upright() < self.TILT_LIMIT and dist > 0.05:
            return "fail"
        return None

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)                  # resets data, step_count, set_arm()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        name = options.get("shape") or str(self._rng.choice(self.SHAPE_NAMES))
        self.shape = name
        idx = self.SHAPE_NAMES.index(name)
        self._target_xy = np.array([HOLE_X, HOLE_Y[idx]])

        self.set_arm(self.START_TIP)              # move arm first so it can't sweep the object
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

        for _ in range(50):                       # settle with the arm holding its pose
            self.data.ctrl[:6] = self.targets[:-1]
            mujoco.mj_step(self.model, self.data)

        self.current_position = self._tip_pos()   # base step() integrates from this

        obj, tip = self._obj_pos(), self._tip_pos()
        self._prev_goal = np.linalg.norm(obj[:2] - self._target_xy)
        self._prev_yaw  = abs(self._yaw_error())
        self._prev_tip  = np.linalg.norm(tip - self._push_point(obj))
        return self._get_obs(), {"shape": name}

    def _get_obs(self):
        tip = self.get_nodes().flatten()
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
            obs = np.concatenate([obs, self._event_obs()])   # + n_taxels values

        return obs.astype(np.float32)

    # called by base step() as self._reward()
    def _reward(self):
        status = self._status()
        if status == "success":
            return self.R_SUCCESS
        if status == "fail":
            return self.R_FALL

        tip, obj = self._tip_pos(), self._obj_pos()
        goal    = np.linalg.norm(obj[:2] - self._target_xy)
        yaw_abs = abs(self._yaw_error())
        tip_d   = np.linalg.norm(tip - self._push_point(obj))

        r = (10.0 * (self._prev_goal - goal)      # push toward the correct hole
             + 1.0 * (self._prev_yaw - yaw_abs)   # rotate toward the hole's orientation
             + 1.0 * (self._prev_tip - tip_d)     # weak: get the tip to the object
             - 0.01)                              # time penalty
        self._prev_goal, self._prev_yaw, self._prev_tip = goal, yaw_abs, tip_d
        return r

    # called by base step() as self._done()
    def _done(self):
        return self._status() is not None


class Incline(RobotArmEnv):
    # ---- geometry ----
    BALL_R = 0.02
    TIP_R  = 0.01            # approx radius of the contact tip
    TIP_H  = 0.02            # tip height above the surface along the normal (= ball centre height)

    # ---- friction randomisation: one "level" f in [0,1] maps to both coefficients ----
    F_SLIDE = (0.05, 1.0)    # sliding friction, log-interpolated
    F_ROLL  = (0.0, 0.008)   # rolling friction, linear (needs condim=6). ~0.0054 holds a ball on 15 deg
    OBSERVE_FRICTION = False # True = give the policy f (useful for debugging); False = hidden

    # ---- control ----
    ACTION_SCALE = 0.005     # metres per env step at |action| = 1
    FRAME_SKIP   = 10        # physics steps per env step
    S_MIN, Y_LIM = -0.06, 0.06

    # ---- reward ----
    SUCCESS_MARGIN = 0.04    # success when the ball is this close to the top
    PROGRESS_W = 30.0        # reward per metre of progress up the slope
    R_SUCCESS  = 100.0
    R_FAIL     = -50.0

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "hill_worldbody_tx.xml"],
            obs_dim=213+ 10 if self.OBSERVE_FRICTION else 9,
            action_dim=2
        )
        m, d = self.model, self.data
        self._rng = np.random.default_rng()
        self._ee_site   = m.site("ee_site").id
        self._ball_body = m.body("ball").id
        jid = m.joint("ball_free").id
        self._ball_qadr = m.jnt_qposadr[jid]
        self._ball_dadr = m.jnt_dofadr[jid]
        self._ramp_geom = m.geom("ramp_surface").id

        # read the ramp frame from the model (static body, so compute once)
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
            return "fail"                                    # rolled off the bottom / side / popped off
        return None

    def _push_dist(self, tip, ball):
        """Distance (in the slope plane) from the tip to the point just below the ball."""
        s_b, _, lat_b = self._slope_coords(ball)
        s_t, _, lat_t = self._slope_coords(tip)
        s_goal = s_b - (self.BALL_R + self.TIP_R + 0.01)
        return float(np.hypot(s_t - s_goal, lat_t - lat_b))

    # ---------- gym API ----------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)               # resets data, step_count, set_arm()
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}

        f = options["friction"] if "friction" in options else self._rng.uniform(0.0, 1.0)
        self._set_friction(f)

        # ball start near the bottom, tip just downhill of it
        s0 = self._rng.uniform(0.07, 0.10)
        y0 = self._rng.uniform(-0.02, 0.02)
        self._cmd_s = s0 - (self.BALL_R + self.TIP_R + 0.01)
        self._cmd_y = y0

        self.set_arm(self._tip_target(self._cmd_s, self._cmd_y))   # arm first, so it can't sweep the ball

        centre = self._S + s0 * self._D + (self.BALL_R + 1e-3) * self._N
        centre[1] += y0
        a, d = self._ball_qadr, self._ball_dadr
        self.data.qpos[a:a + 7] = [*centre, 1, 0, 0, 0]
        self.data.qvel[d:d + 6] = 0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(50):                    # settle with the arm holding its pose
            self.data.ctrl[:6] = self.targets[:-1]
            mujoco.mj_step(self.model, self.data)

        self.current_position = self._tip_pos()

        tip, ball = self._tip_pos(), self._ball_pos()
        self._prev_s   = self._slope_coords(ball)[0]
        self._prev_tip = self._push_dist(tip, ball)
        return self._get_obs(), {"friction": self.friction_level}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=float).ravel()[:2], -1.0, 1.0)
        self._cmd_s = float(np.clip(self._cmd_s + a[0] * self.ACTION_SCALE, self.S_MIN, self.S_MAX))
        self._cmd_y = float(np.clip(self._cmd_y + a[1] * self.ACTION_SCALE, -self.Y_LIM, self.Y_LIM))

        self.move_gripper_to(self._tip_target(self._cmd_s, self._cmd_y), iter=150)   # sets self.targets
        for _ in range(self.FRAME_SKIP):
            self.data.ctrl[:6] = self.targets[:-1]
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs        = self._get_obs()
        reward     = self._reward()
        terminated = self._done()
        truncated  = self.step_count >= self.max_steps
        self.current_position = self._tip_pos()
        s = self._slope_coords(self._ball_pos())[0]
        return obs, reward, terminated, truncated, {"friction": self.friction_level, "ball_s": s}

    def _get_obs(self):
        tip = self.get_nodes().flatten()
        ball =  self._ball_pos()
        vel = self.data.qvel[self._ball_dadr:self._ball_dadr + 3]   # free joint: linear vel, world frame
        s_b, _, lat_b = self._slope_coords(ball)
        s_t, _, lat_t = self._slope_coords(tip)
        obs = [tip, ball, vel, [s_b / self._L], [s_t - s_b, lat_t - lat_b]]
        if self.OBSERVE_FRICTION:
            obs.append([self.friction_level])
        return np.concatenate(obs).astype(np.float32)

    # called by step() as self._reward()
    def _reward(self):
        status = self._status()
        if status == "success":
            return self.R_SUCCESS
        if status == "fail":
            return self.R_FAIL

        tip, ball = self._tip_pos(), self._ball_pos()
        s = self._slope_coords(ball)[0]
        tip_d = self._push_dist(tip, ball)

        r = (self.PROGRESS_W * (s - self._prev_s)      # up the slope = good, rolling back = bad
             + 1.0 * (self._prev_tip - tip_d)          # weak: stay behind the ball
             - 0.01)                                   # time penalty
        self._prev_s, self._prev_tip = s, tip_d
        return r

    # called by step() as self._done()
    def _done(self):
        return self._status() is not None