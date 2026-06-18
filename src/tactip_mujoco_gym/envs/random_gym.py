from .base import TactileGymEnv
import numpy as np


class RandomEnv(TactileGymEnv):

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactile_env.xml"],
            obs_dim=9,
            action_dim=3
        )

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
        ])

    def _reward(self):
        obj_pos = self.data.body("object").xpos
        goal = np.array([0.6, 0.0, 0.1])
        return -np.linalg.norm(obj_pos - goal)