
from gymnasium.envs.registration import register

register(
    id="RobotArm-v0",
    entry_point="tactip_mujoco_gym.envs.robot_arm:RobotArmEnv",
)

register(
    id="RobotFoot-v0",
    entry_point="tactip_mujoco_gym.envs.quadruped:RobotFootEnv",
)

register(
    id="RandomTac-v0",
    entry_point="tactip_mujoco_gym.envs.random_gym:RandomEnv",
)