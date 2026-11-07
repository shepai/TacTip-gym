"""
The task catalogue. Every task is a Gymnasium environment:

    env = gym.make("TacArm-peg", obs_modes=["tactile"])          # after import tactip_mujoco_gym
    env = gym.make("tactip_mujoco_gym:TacArm-peg", ...)          # imports the package itself

`python -m tactip_mujoco_gym list` prints this table; `... describe <task>` builds the task
and shows every input it offers. Keyword arguments every robot accepts:

    obs_modes     inputs the policy gets: {"tactile": True, ...}, ["tactile", "vision"] or
                  "tactile+vision" (proprioception is always included)
    obs_format    "flat" (one vector) or "dict" (one key per input, for fusion networks)
    sensors       {key: readout spec}: change, add or remove sensor readouts
    sensor_xml    the sensor hardware mounted on the robot (default: the TacTip)
    overrides     {"CONSTANT": value}: any of the env's class constants, for this instance
    max_steps     episode length
plus the task's own options below.
"""
from gymnasium.envs.registration import register

ARM = "Unitree Z1 arm, sensor as the fingertip (1 mount: tip)"
QUAD = "Unitree Go2 quadruped, sensor on every foot (4 mounts: FL FR RL RR)"
HUMANOID = "RB-Y1 wheeled humanoid, sensors as palms (2 mounts: left right)"
ARM_INPUTS = ("privileged", "tactile", "tactile_image", "vision", "events")
ARM_KW = {"start_touching": "start in contact with the object (default True)"}

TASKS = {
    "TacArm-edge": dict(
        entry_point="tactip_mujoco_gym.envs.robot_arm:Edge", robot=ARM, inputs=ARM_INPUTS,
        summary="follow the top edge of a randomly rotated block for one full lap",
        actions="3: tip velocity x, y, z", options={}),
    "TacArm-peg": dict(
        entry_point="tactip_mujoco_gym.envs.robot_arm:Peg", robot=ARM, inputs=ARM_INPUTS,
        summary="push an upright cylinder into a hole",
        actions="2: tip velocity x, y", options=dict(ARM_KW)),
    "TacArm-shape": dict(
        entry_point="tactip_mujoco_gym.envs.robot_arm:ShapeInsertion", robot=ARM,
        inputs=ARM_INPUTS, summary="identify one of four shapes and push it into its own hole",
        actions="2: tip velocity x, y",
        options={**ARM_KW, "spawn_yaw": "start rotation range, rad (default 0.1)",
                 "spawn_toward_hole": "0 = start in the middle, 1 = in line with the hole"}),
    "TacArm-hill": dict(
        entry_point="tactip_mujoco_gym.envs.robot_arm:Incline", robot=ARM, inputs=ARM_INPUTS,
        summary="stop a ball rolling down a ramp of unknown friction, push it to the top",
        actions="2: along / across the slope",
        options={"observe_friction": "add the true friction to the privileged input"}),
    "TacQuad-walk": dict(
        entry_point="tactip_mujoco_gym.envs.tactip_quadruped:TacQuadruped", robot=QUAD,
        inputs=("privileged", "tactile", "tactile_image", "vision"),
        summary="follow velocity commands over tiles of changing friction, softness, bumps",
        actions="12: joint targets",
        options={"feet": "'tactip' | 'proxy' (sphere, no touch, fast) | 'rigid'",
                 "sensor_stiffness": "skin stiffness multiplier (default 10)",
                 "terrain_level": "0..1 fixed difficulty, None = curriculum",
                 "command": "None = random, or (vx, vy, yaw_rate)"}),
    "TacHumanoid-pick": dict(
        entry_point="tactip_mujoco_gym.envs.tactip_humanoid:TacHumanoidPick", robot=HUMANOID,
        inputs=("object", "privileged", "tactile", "tactile_image", "vision"),
        summary="squeeze-grasp a soft or rigid object, carry it to a target, put it down",
        actions="7 (bimanual) or 9 (independent)",
        options={"objects": "names to draw from (default all ten)",
                 "level": "0..1 fixed difficulty, None = curriculum",
                 "action_mode": "'bimanual' | 'independent'",
                 "sensor_stiffness": "skin stiffness multiplier (default 10)"}),
    "RobotArm-v0": dict(
        entry_point="tactip_mujoco_gym.envs.robot_arm:RobotArmEnv", robot=ARM,
        inputs=ARM_INPUTS, summary="the arm and sensor with no task (zero reward, no end)",
        actions="3: tip velocity x, y, z", options={}),
    "RandomTac-v0": dict(
        entry_point="tactip_mujoco_gym.envs.random_gym:RandomEnv",
        robot="the TacTip on its own, boxes dropped on it", inputs=(),
        summary="standalone sensor; observation {'image': in-sensor camera} (legacy API)",
        actions="none", options={}),
}

for _task_id, _task in TASKS.items():
    register(id=_task_id, entry_point=_task["entry_point"])


def list_tasks():
    """{task id: summary} for every registered task."""
    return {k: v["summary"] for k, v in TASKS.items()}
