import gymnasium as gym
import tactip_mujoco_gym          # Registers your environments
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
import cv2
import time
import mujoco.viewer
import os 
import torch 
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
def run(env_name, model_name, log_name, total_timesteps, n_envs=8):
    os.makedirs("models", exist_ok=True)
    env = make_vec_env(env_name, n_envs=n_envs)
    model = PPO("MlpPolicy", env, n_steps=512, batch_size=1024, device="auto", verbose=1)
    model.set_logger(configure(f"logs/{log_name}", ["stdout", "csv"]))   # add "tensorboard" if installed
    model.learn(total_timesteps, progress_bar=True)
    model.save(f"models/{model_name}")
 
"""
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
 """

def visualise(env_name, model_name,
              record=False,          # save a video?
              log_dir=None,          # folder for the video (default: logs/<model_name>)
              video_name=None,       # default: <model_name>_<timestamp>.mp4
              video_seconds=None,    # stop after this many SIM seconds (None = until viewer closes)
              show=True,             # False = headless, no window (needs video_seconds)
              width=640, height=480, fps=30,
              track_body_id=1):      # camera follows this body; None = default free camera
    env = gym.make(env_name)
    raw = env.unwrapped
    model = PPO.load(f"models/{model_name}", device="cpu")
    dt = raw.frame_skip * raw.model.opt.timestep
    obs, _ = env.reset()

    # ---- optional video setup ----
    writer = renderer = cam = None
    if record:
        if not show and video_seconds is None:
            raise ValueError("Headless recording (show=False) needs video_seconds.")
        log_dir = log_dir or os.path.join("logs", model_name)
        os.makedirs(log_dir, exist_ok=True)
        name = video_name or f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        video_path = os.path.join(log_dir, name)

        renderer = mujoco.Renderer(raw.model, height=height, width=width)
        writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"cv2 could not open a video writer for {video_path}")

        if track_body_id is not None:
            cam = mujoco.MjvCamera()
            cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            cam.trackbodyid = track_body_id
            cam.distance, cam.azimuth, cam.elevation = 0.5, 135, -25
        else:
            cam = -1                                  # model's default camera

    sim_t, next_frame = 0.0, 0.0

    def advance():
        """One policy step + (optionally) grab a video frame."""
        nonlocal obs, sim_t, next_frame
        action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, _ = env.step(action)
        sim_t += dt
        if term or trunc:
            obs, _ = env.reset()
            print("new command:", raw.cmd)
        if writer is not None and sim_t >= next_frame:
            renderer.update_scene(raw.data, camera=cam)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))   # cv2 wants BGR
            next_frame += 1.0 / fps

    finished = lambda: video_seconds is not None and sim_t >= video_seconds

    try:
        if show:
            with mujoco.viewer.launch_passive(raw.model, raw.data) as v:
                while v.is_running() and not finished():
                    with v.lock():
                        advance()
                    v.sync()
                    time.sleep(dt)
        else:
            while not finished():                     # headless: runs faster than real time
                advance()
    finally:
        if writer is not None:
            writer.release()
            renderer.close()
            print(f"Video saved to {video_path}")
        env.close()