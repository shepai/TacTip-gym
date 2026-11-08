"""
Teach the two-TacTip gripper to pick up objects of any shape, softness, weight and surface
friction, carry them to a target and put them down (TacGripper-pick).

The task, objects, curriculum and reward are TacHumanoid-pick's, and so is the training recipe
(humanoid_pick.py, in this folder): scripted demos from ScriptedPicker for a warm start, then
PPO with a fading behaviour-cloning term. Comparing the two robots' results shows what the
humanoid's body adds or costs; comparing OBS_MODES shows what touch adds:

    OBS_MODES                              what it tells you
    {"privileged": True}                   teacher: knows mass, friction, stiffness, squash
    {"object": True}                       sees where the object is, feels nothing
    {"object": True, "tactile": True}      sees + feels
    {"vision": True, "tactile": True}      overhead camera + touch (obs_format "dict", a GPU)

    python "examples/RL tasks/gripper_pick.py"

Logs: logs/<MODEL_NAME>/progress.csv (pick/success, pick/level, pick/squash, pick/fail_*,
reward/...), as for the humanoid. Demos are cached in demos/ (delete them to collect again).
"""
from humanoid_pick import PPO_KWARGS, PickLogger, ScriptedWarmStart, collect_demos, evaluate_objects

from tactip_mujoco_gym.RL_BASE import evaluate, run, visualise

ENV_NAME = "TacGripper-pick"
OBS_MODES = {"object": True, "tactile": True}
OBS_FORMAT = "flat"                    # "dict" for vision (CNN on the camera image)
ENV_KWARGS = {"sensor_stiffness": 10.0}
STEPS = 10_000_000
N_ENVS = 16
DEMOS = 400                            # scripted episodes for the warm start (0 = pure RL)
DEMO_LEVELS = (0.0, 0.5)               # difficulty range of the demos
MODEL_NAME = "gripper_" + ("+".join(k for k, on in OBS_MODES.items() if on) or "blind") + ("_demo" if DEMOS else "")
LOG_NAME = MODEL_NAME


if __name__ == "__main__":             # required: training and the demos use subprocesses
    env_kwargs = {**ENV_KWARGS, "obs_modes": OBS_MODES, "obs_format": OBS_FORMAT}
    callbacks = [PickLogger()]
    if DEMOS:
        # collect (or load) the demos before run() starts N_ENVS training processes
        collect_demos(env_kwargs, DEMOS, levels=DEMO_LEVELS, env_name=ENV_NAME)
        callbacks.insert(0, ScriptedWarmStart(env_kwargs, episodes=DEMOS, env_name=ENV_NAME,
                                              demo_levels=DEMO_LEVELS))
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, n_envs=N_ENVS, obs_modes=OBS_MODES,
        obs_format=OBS_FORMAT, env_kwargs=ENV_KWARGS, algo_kwargs=PPO_KWARGS,
        callbacks=callbacks, record_video=True, video_seconds=10, checkpoint_every=2_000_000)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=20)
    evaluate_objects(MODEL_NAME, env_name=ENV_NAME)
    visualise(ENV_NAME, MODEL_NAME)
