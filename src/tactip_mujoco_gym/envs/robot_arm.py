from .base import TactileGymEnv
import numpy as np


class RobotArmEnv(TactileGymEnv):

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "tactip_arm.xml"],
            obs_dim=9,
            action_dim=7
        )
    def _reward(self):
        return np.random.randint(0,1)