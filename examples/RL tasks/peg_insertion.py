from RL_BASE import run, evaluate, visualise

ENV_NAME = "TacArm-peg"
# Which inputs the policy gets (proprioception is always included):
#   privileged = true object / goal state, tactile = tactip markers, vision = hand camera
OBS_MODES = {"vision": False, "tactile": False, "privileged": True}
OBS_FORMAT = "flat"                # "flat" = one long vector; "dict" = one key per input (fusion)
MODEL_NAME = "TacArm-peg_v3_" + "+".join(k for k, on in OBS_MODES.items() if on)  # -> models/<MODEL_NAME>.zip
LOG_NAME = MODEL_NAME              # -> logs/<LOG_NAME>/progress.csv
STEPS = 3_000_000
ENV_KWARGS = {}

if __name__ == "__main__":         # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, obs_modes=OBS_MODES, obs_format=OBS_FORMAT,
        env_kwargs=ENV_KWARGS, record_video=True)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=50)
    visualise(ENV_NAME, MODEL_NAME)
