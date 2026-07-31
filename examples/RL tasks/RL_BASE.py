import gymnasium as gym
import tactip_mujoco_gym          # Registers your environments
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import cv2
import time
import mujoco.viewer
def main(ENV_NAME = "EdgeFollow-v0",MODEL_NAME = "ppo_edge_follow",TOTAL_TIMESTEPS = 500_000):

    # Create environment
    env = gym.make(ENV_NAME)

    # Record episode statistics
    env = Monitor(env)

    # Create PPO model
    model = PPO(
        policy= "MlpPolicy",    # Dict observations
        env=env,

        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        device="cpu",
    )
    log_dir = "./sb3_logs/"
    # This tells SB3 to print to console, save to CSV, and save to JSON
    new_logger = configure(log_dir, ["stdout", "csv", "json"])
    model.set_logger(new_logger)

    # Train
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        progress_bar=True,
    )

    # Save model
    model.save(f"models/{MODEL_NAME}")

    env.close()

def show(ENV_NAME = "EdgeFollow-v0",MODEL_NAME = "ppo_edge_follow",):
    
     # -----------------------------
    # Load environment
    # -----------------------------
    env = gym.make(ENV_NAME)

    obs, info = env.reset()


    # -----------------------------
    # Load trained model
    # -----------------------------
    model = PPO.load(
        "models/"+MODEL_NAME
    )


    # IMPORTANT: use underlying MuJoCo model/data
    mujoco_model = env.unwrapped.model
    data = env.unwrapped.data

    # Passive viewer = you control simulation loop
    with mujoco.viewer.launch_passive(
        mujoco_model,
        data
    ) as viewer:

        while viewer.is_running():
            # -----------------------------
            # Policy chooses action
            # -----------------------------
            action, _ = model.predict(
                obs,
                deterministic=True
            )


            # Step simulation through Gym
            obs, reward, terminated, truncated, info = env.step(action)


            # Sync MuJoCo viewer
            viewer.sync()

            #cv2.waitKey(1)

            # Print reward occasionally
            print(
                f"Reward: {reward:.3f}",
                end="\r"
            )


            # Reset episode
            if terminated or truncated:

                print("\nEpisode finished")

                obs, info = env.reset()


            # Prevent maxing CPU
            time.sleep(
                env.unwrapped.model.opt.timestep
            )


    env.close()

if __name__ == "__main__":
    main()