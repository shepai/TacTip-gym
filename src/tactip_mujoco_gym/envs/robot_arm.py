from .base import TactileGymEnv
import numpy as np
from mujoco import mj_step, mj_resetData
import mujoco 

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
        dq=self.kinematic_control(action)
        self.data.ctrl[:] = dq[:6]
        mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}
    def kinematic_control(self,target_pos, kp=2.0, damping=0.1):
        """
        Returns joint velocities (or position increments) using Jacobian IK.
        Works inside MuJoCo step() loop.
        """
        ee_site_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_SITE,
                "ee_site"
            )
        # current end-effector position
        x = self.data.site_xpos[ee_site_id].copy()

        # position error
        error = target_pos - x

        # task-space velocity command (P controller)
        xdot = kp * error

        # Jacobian (3 x nv)
        J = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, J, None, ee_site_id)

        # damped least squares IK
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(3), xdot)

        return dq