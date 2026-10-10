from .base import TactileGymEnv
import numpy as np
from mujoco import mj_step, mj_resetData
import mujoco 
from dm_control import mujoco
from dm_control.utils import inverse_kinematics

class RobotArmEnv(TactileGymEnv):

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
            xml_subpath=["assets", "tactip_arm_peg.xml"],
            obs_dim=8,
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
        tip, cyl = self._tip_pos(), self._cyl_pos()
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
            xml_subpath=["assets", "tactip_arm_edge.xml"],
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
        return self.get_nodes().flatten()#super()._get_obs()['image']

    