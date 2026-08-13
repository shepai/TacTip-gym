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

def show(
    ENV_NAME="EdgeFollow-v0",
    MODEL_NAME="ppo_edge_follow",
    record=False,
    filename="recording.mp4",
):

    import cv2

    # -----------------------------
    # Load environment
    # -----------------------------
    env = gym.make(ENV_NAME)

    obs, info = env.reset()

    # -----------------------------
    # Load trained model
    # -----------------------------
    model = PPO.load(
        "models/" + MODEL_NAME
    )

    # IMPORTANT: use underlying MuJoCo model/data
    mujoco_model = env.unwrapped.model
    data = env.unwrapped.data

    # -----------------------------
    # Video recorder
    # -----------------------------
    video = None
    renderer = None

    if record:

        width = 640
        height = 480

        renderer = mujoco.Renderer(
            mujoco_model,
            height=height,
            width=width
        )

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        video = cv2.VideoWriter(
            filename,
            fourcc,
            30.0,
            (width, height)
        )

        if not video.isOpened():
            raise RuntimeError(
                f"Could not open video file: {filename}"
            )

    # -----------------------------
    # Passive viewer
    # -----------------------------
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

            # -----------------------------
            # Step simulation
            # -----------------------------
            obs, reward, terminated, truncated, info = env.step(
                action
            )

            # -----------------------------
            # Record MuJoCo frame
            # -----------------------------
            if record:

                renderer.update_scene(
                    data
                )

                frame = renderer.render()

                # MuJoCo gives RGB
                # OpenCV expects BGR
                frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_RGB2BGR
                )

                video.write(frame)

            # -----------------------------
            # Sync viewer
            # -----------------------------
            viewer.sync()

            print(
                f"Reward: {reward:.3f}",
                end="\r"
            )

            # -----------------------------
            # Reset episode
            # -----------------------------
            if terminated or truncated:

                print("\nEpisode finished")

                obs, info = env.reset()

            # -----------------------------
            # Prevent maxing CPU
            # -----------------------------
            time.sleep(
                env.unwrapped.model.opt.timestep
            )

    # -----------------------------
    # Close video
    # -----------------------------
    if renderer is not None:
        renderer.close()

    if video is not None:
        video.release()

        print(
            f"\nSaved recording to:\n{filename}"
        )

    env.close()