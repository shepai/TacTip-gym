from .base import TactileGymEnv
import numpy as np
from mujoco import mj_step, mj_resetData
import mujoco 
from dm_control import mujoco
from dm_control.utils import inverse_kinematics

class RobotArmEnv(TactileGymEnv):

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactip_arm.xml"],
            obs_dim=8,
            action_dim=3
        )
        self.set_arm()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.step_count = 0
        self.set_arm()
        return self._get_obs(), {}
    def set_arm(self,default=[0.3,0.3,0.05]):
        for i in range(250):
            self.step(default)
    def _reward(self):
        return np.random.randint(0,1)
    def step(self,action):
        mujoco.mj_forward(self.model, self.data)
        self.move_gripper_to(action)
        self.data.ctrl[:6] = self.targets[:-1]
        mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

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
            max_steps=200
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
        super().__init__(
            xml_subpath=["assets", "tactip_arm_edge.xml"],
            obs_dim=8,
            action_dim=3
        )
        self.set_arm()
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        #TODO place robot in position
        return self._get_obs(), {}
    def _reward(self): #TODO make gym for this
        # fingertip position
        tip = self.data.site_xpos[
            self.model.site("ee_site").id
        ]
        edge_point = self.closest_edge_point(tip) #TODO implement this

        edge_error = np.linalg.norm(
            tip[:2] - edge_point[:2]
        )
        progress = self.edge_progress(tip) #TODO implement this

        delta_progress = (
            progress -
            self.previous_progress
        )
        self.previous_progress = progress
        # contact force
        force = self.get_contact_force() #TODO implement this
        contact_penalty = 0
        if force < 0.1:
            contact_penalty = 5

        reward = (
            -2.0 * edge_error
            +5.0 * delta_progress
            -0.01 * np.linalg.norm(self.action)
            -contact_penalty
        )
        return reward