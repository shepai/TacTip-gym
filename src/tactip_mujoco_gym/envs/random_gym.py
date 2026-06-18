from .base import TactileGymEnv
import numpy as np
from mujoco import mj_step, mj_resetData
import mujoco

class RandomEnv(TactileGymEnv):
    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactile_env.xml"],
            obs_dim=9,
            action_dim=3
        )
    def _get_obs(self):
        self.renderer.update_scene(self.data, camera="sensor_cam")
        img = self.renderer.render()
        img = img.astype("float32") / 255.0
        obs = {
        "state": None,
        "image": img
    }
        return obs
    def step(self,action):
        mj_step(self.model, self.data)

        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}
    def _reward(self):
        return np.random.randint(0,10)