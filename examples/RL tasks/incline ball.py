from RL_BASE import*

#main(ENV_NAME = "TacArm-hill",MODEL_NAME = "ppo_incline",TOTAL_TIMESTEPS = 500_000)
ENV_NAME = "TacArm-hill"
MODEL_NAME = "TacArm-hill_v1"      # -> models/urchin_v1.zip
LOG_NAME = "TacArm-hill_v1"        # -> logs/urchin_v1/progress.csv
STEPS = 5_000_000
 
if __name__ == "__main__":
    run(ENV_NAME, MODEL_NAME, LOG_NAME, STEPS,record_video=True)
    visualise(ENV_NAME, MODEL_NAME)
 