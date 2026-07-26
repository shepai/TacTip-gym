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
        for i in range(150):
            self.move_gripper_to(default)
            self.data.ctrl[:6] = self.targets[:-1]
            mj_step(self.model, self.data)
            mujoco.mj_forward(
                self.model,
                self.data
            )
    def _reward(self):
        return np.random.randint(0,1)
    def step(self,action):
        mujoco.mj_forward(self.model, self.data)
        self.current_position+action #treat as vector instead of position
        self.move_gripper_to(action)
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
    def move_gripper_to(self, fingertip_coords):
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
            max_steps=750
        )

        # ONLY update sim state once
        self.targets = result.qpos[:7]
        #mj.mj_forward(self.model, self.data)

class Peg(RobotArmEnv):
    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactip_arm_peg.xml"],
            obs_dim=8,
            action_dim=3
        )
        self.set_arm()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #TODO place robot in position
        return self._get_obs(), {}
    def _reward(self): #TODO make gym for this
        peg_pos = self.data.site_xpos[
            self.model.site("peg_tip").id
        ]

        hole_pos = self.data.site_xpos[
            self.model.site("hole").id
        ]
        # horizontal alignment
        xy_error = np.linalg.norm(
            peg_pos[:2] -
            hole_pos[:2]
        )
        # insertion depth
        depth = (
            hole_pos[2]
            -
            peg_pos[2]
        )

        reward = (
            -10 * xy_error
            +5 * depth
        )
        # success condition
        if depth > 0.05 and xy_error < 0.005:
            reward += 100

        return reward
class Edge(RobotArmEnv):
    def __init__(self):
        self.previous_progress=0
        super().__init__(
            xml_subpath=["assets", "tactip_arm_edge.xml"],
            obs_dim=8,
            action_dim=3
        )
        #self.set_arm()
    def random_edge_point(self):
        geom_id = self.model.geom("block_geom").id

        center = self.data.geom_xpos[geom_id]
        hx, hy, hz = self.model.geom_size[geom_id]

        s = np.random.uniform(0,1)

        width = 2*hx
        height = 2*hy

        d = s * 2*(width+height)

        if d < width:
            p = np.array([-hx+d, -hy, hz])

        elif d < width+height:
            p = np.array([hx, -hy+(d-width), hz])

        elif d < 2*width+height:
            p = np.array([hx-(d-width-height), hy, hz])

        else:
            p = np.array([-hx, hy-(d-2*width-height), hz])


        # rotate local point into world frame
        R = self.data.geom_xmat[geom_id].reshape(3,3)

        return center + R @ p
    def randomize_block_rotation(self):
        angle = np.random.uniform(-np.pi, np.pi)

        quat = np.array([
            np.cos(angle/2),
            0,
            0,
            np.sin(angle/2)
        ])

        body_id = self.model.body("block").id

        self.model.body_quat[body_id] = quat

        mujoco.mj_forward(
            self.model,
            self.data
        )
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
    def edge_progress(self, tip):

        geom_id = self.model.geom("block_geom").id

        center = self.data.geom_xpos[geom_id]
        hx, hy, hz = self.model.geom_size[geom_id]

        p = tip - center

        x, y = p[:2]

        # clockwise edge lengths
        width = 2 * hx
        height = 2 * hy
        perimeter = 2 * (width + height)

        # distances to each edge
        edges = [
            (abs(y + hy), "bottom"),
            (abs(x - hx), "right"),
            (abs(y - hy), "top"),
            (abs(x + hx), "left"),
        ]

        _, edge = min(edges)

        if edge == "bottom":
            x = np.clip(x, -hx, hx)
            s = x + hx

        elif edge == "right":
            y = np.clip(y, -hy, hy)
            s = width + (y + hy)

        elif edge == "top":
            x = np.clip(x, -hx, hx)
            s = width + height + (hx - x)

        elif edge == "left":
            y = np.clip(y, -hy, hy)
            s = 2*width + height + (hy - y)

        return s / perimeter
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