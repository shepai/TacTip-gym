import os

import cv2
import gymnasium as gym
import mujoco
from stable_baselines3.common.callbacks import BaseCallback


def write_video(frames, path, fps=30):
    if not frames:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))   # RGB -> BGR for OpenCV
    finally:
        writer.release()


class TrainingVideoCallback(BaseCallback):
    """Records a short video of the current policy every `every_percent` of training.

    Uses the training VecNormalize statistics to normalise observations - without that the
    videos show a policy fed un-normalised inputs, i.e. near-random behaviour, even when
    training is going well.
    """

    def __init__(self, env_name, total_timesteps, video_dir, every_percent=20, video_seconds=10,
                 env_kwargs=None, verbose=1):
        super().__init__(verbose)
        self.env_name = env_name
        self.env_kwargs = env_kwargs or {}
        self.total_timesteps = total_timesteps
        self.video_dir = video_dir
        self.every_percent = every_percent
        self.video_seconds = video_seconds
        self.next_percent = every_percent
        self.render_env = None
        self.renderer = None

    def _on_training_start(self):
        os.makedirs(self.video_dir, exist_ok=True)
        self.render_env = gym.make(self.env_name, **self.env_kwargs)
        raw = self.render_env.unwrapped
        self.renderer = mujoco.Renderer(raw.model, height=480, width=640)
        self.cam = mujoco.MjvCamera()
        core = mujoco.mj_name2id(raw.model, mujoco.mjtObj.mjOBJ_BODY, "core")
        if core >= 0:                                       # follow the sensor
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = core
        else:
            mujoco.mjv_defaultFreeCamera(raw.model, self.cam)
            self.cam.lookat[:] = [0.4, 0.0, 0.05]
        self.cam.distance, self.cam.azimuth, self.cam.elevation = 0.5, 135, -25

    def _on_step(self):
        progress = 100 * self.num_timesteps / self.total_timesteps
        while self.next_percent <= 100 and progress >= self.next_percent:
            self._record(self.next_percent)
            self.next_percent += self.every_percent
        return True

    def _normalise(self, obs):
        vn = self.model.get_vec_normalize_env()
        return vn.normalize_obs(obs) if vn is not None else obs

    def _record(self, percent):
        env = self.render_env
        raw = env.unwrapped
        obs, _ = env.reset()
        policy_dt = raw.FRAME_SKIP * raw.model.opt.timestep
        fps = max(1, round(1.0 / policy_dt))
        n_frames = max(1, round(self.video_seconds * fps))
        frames, successes, episodes = [], 0, 0
        for _ in range(n_frames):
            self.renderer.update_scene(raw.data, camera=self.cam)
            frames.append(self.renderer.render().copy())
            action, _ = self.model.predict(self._normalise(obs), deterministic=True)
            obs, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                episodes += 1
                successes += bool(info.get("is_success", False))
                obs, _ = env.reset()
        path = os.path.join(self.video_dir, f"progress_{percent:03d}pct_{self.num_timesteps}.mp4")
        write_video(frames, path, fps=fps)
        if self.verbose:
            print(f"Saved training video: {path}  ({successes}/{episodes} episodes succeeded)")

    def _on_training_end(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.render_env is not None:
            self.render_env.close()
            self.render_env = None
