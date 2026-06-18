from .base import TactileGymEnv


class RobotFootEnv(TactileGymEnv):

    def __init__(self):
        super().__init__(
            xml_subpath=["assets", "robot_foot", "scene.xml"],
            obs_dim=12,
            action_dim=6
        )

    def _get_obs(self):
        return np.concatenate([
            self.data.qpos,
            self.data.qvel,
        ])

    def _reward(self):
        # e.g. forward velocity / stability
        return self.data.qvel[0]