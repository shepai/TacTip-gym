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
            obs_dim=7,
            action_dim=3
        )
    def _reward(self):
        return np.random.randint(0,1)
    def step(self,action):
        mujoco.mj_forward(self.model, self.data)
        dq=self.move_gripper_to(action)
        self.data.ctrl[:] = dq[:6]
        mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}
    def move_gripper_to(self, fingertip_coords):
        self.physics.data.qpos[:] = self.data.qpos[:]
        self.physics.data.qvel[:] = self.data.qvel[:]
        self.physics.forward()
        result = inverse_kinematics.qpos_from_site_pose(
            self.physics,
            site_name="attachment_site",
            joint_names=["joint"+str(i) for i in range(1,8)],
            target_pos=fingertip_coords,
            max_steps=200
        )

        # ONLY update sim state once
        self.targets = result.qpos[:7]
        #mj.mj_forward(self.model, self.data)