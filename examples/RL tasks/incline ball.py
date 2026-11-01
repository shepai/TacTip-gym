from RL_BASE import run, evaluate, visualise

ENV_NAME = "TacArm-hill"
PRIVILEGED = True                  # False = learn from the tactip markers instead
MODEL_NAME = "TacArm-hill_v2" + ("" if PRIVILEGED else "_tactile")   # -> models/<MODEL_NAME>.zip
LOG_NAME = MODEL_NAME              # -> logs/<LOG_NAME>/progress.csv
STEPS = 5_000_000
ENV_KWARGS = {}

if __name__ == "__main__":         # required: training uses one subprocess per env
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS, privileged=PRIVILEGED, env_kwargs=ENV_KWARGS,
        record_video=True)
    evaluate(ENV_NAME, MODEL_NAME, n_episodes=50)
    visualise(ENV_NAME, MODEL_NAME)
