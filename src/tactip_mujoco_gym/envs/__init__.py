
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

# The arm tasks take keyword arguments through gym.make / make_vec_env(env_kwargs=...):
#   privileged=True   (default) tip + object state;  False = tactip markers + proprioception
#   events=False      add the tx_* taxel event counts to the observation
#   max_steps=None    override the episode length
# task specific: Peg/ShapeInsertion(start_touching=True), ShapeInsertion(spawn_yaw=0.1),
#                Incline(observe_friction=False)
register(
    id="TacArm-edge",
    entry_point="tactip_mujoco_gym.envs.robot_arm:Edge",
)

register(
    id="TacArm-peg",
    entry_point="tactip_mujoco_gym.envs.robot_arm:Peg",
)

register(
    id="TacArm-shape",
    entry_point="tactip_mujoco_gym.envs.robot_arm:ShapeInsertion",
)

register(
    id="TacArm-hill",
    entry_point="tactip_mujoco_gym.envs.robot_arm:Incline",
)

# Quadruped with a TacTip on each foot (Unitree Go2 from MuJoCo Menagerie; fetch it with
# tools/fetch_go2.py). Keyword arguments: feet="tactip" | "proxy" | "rigid",
# sensor_stiffness=10.0, terrain_level=None (curriculum) or 0..1, curriculum=True,
# command=None or (vx, vy, yaw_rate), obs_modes, obs_format, max_steps.
register(
    id="TacQuad-walk",
    entry_point="tactip_mujoco_gym.envs.tactip_quadruped:TacQuadruped",
)
