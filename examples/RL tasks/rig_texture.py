"""
Active texture identification on the Cartesian rig (TacRig-texture): the policy moves the
sensor over a block of the 3D Printable Tactile Dataset, presses and slides where it likes,
then answers which of the six textures it is.

For the paper's fixed protocol (presses on a grid, then a classifier) use
rig_texture_dataset.py instead; it is the baseline an active policy should beat on the same
sensor. ENV_KWARGS overrides PRINT_BLUR_MM / PRINT_NOISE_MM imitate a printer's smoothing and
roughness (train on ideal blocks, evaluate on "printed" ones to test robustness).
"""
from tactip_mujoco_gym.RL_BASE import evaluate, run, visualise

ENV_NAME = "TacRig-texture"
OBS_MODES = {"tactile": True}          # + "force" (the plate's force sensor), "tactile_image"
OBS_FORMAT = "flat"                    # "dict" for image inputs
ENV_KWARGS = {"textures": [1, 2, 3, 4, 5, 6]}
STEPS = 2_000_000
N_ENVS = 8
MODEL_NAME = "rig_texture_" + "+".join(k for k, on in OBS_MODES.items() if on)
LOG_NAME = MODEL_NAME

if __name__ == "__main__":             # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, n_envs=N_ENVS, obs_modes=OBS_MODES,
        obs_format=OBS_FORMAT, env_kwargs=ENV_KWARGS, record_video=True)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=100)
    printed = {**ENV_KWARGS, "overrides": {"PRINT_BLUR_MM": 0.2, "PRINT_NOISE_MM": 0.03}}
    print("\non printed blocks (smoothed, rough):")
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=100, env_kwargs=printed)
    visualise(ENV_NAME, MODEL_NAME)
