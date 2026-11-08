"""
Surface following with the arm (TacArm-surface): keep the sensor pressed in by the same depth,
and square to the surface, while it slides along a wavy strip.

    WAVE / TEXTURE        what the strip is
    "random" / None       uniform, chirp or multi-sine waves, smooth (default)
    "random" / "patches"  the same waves with the tactile dataset's textures, changing every 5 cm
    "uniform" / 4         one sine, one texture (square ridges both ways) all along

Train with OBS_MODES {"privileged": True} first (it should succeed quickly), then
{"tactile": True}. Run `python -m tactip_mujoco_gym.check_envs TacArm-surface` to see the
scripted follower before training.
"""
from tactip_mujoco_gym.RL_BASE import evaluate, run, visualise

ENV_NAME = "TacArm-surface"
OBS_MODES = {"tactile": True}          # privileged / tactile / tactile_image / vision
OBS_FORMAT = "flat"                    # "dict" for image inputs
ENV_KWARGS = {"wave": "random", "texture": None, "follow_normal": True}
STEPS = 3_000_000
MODEL_NAME = "surface_" + "+".join(k for k, on in OBS_MODES.items() if on) + (
    f"_{ENV_KWARGS['texture']}" if ENV_KWARGS["texture"] else "")
LOG_NAME = MODEL_NAME

if __name__ == "__main__":             # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, obs_modes=OBS_MODES, obs_format=OBS_FORMAT,
        env_kwargs=ENV_KWARGS, record_video=True)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=50)
    visualise(ENV_NAME, MODEL_NAME)
