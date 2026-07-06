import time
import gymnasium as gym
import tactip_mujoco_gym

import cv2
import numpy as np
import mujoco

def main():

    env = gym.make("RobotArm-v0")

    obs, info = env.reset()

    model = env.unwrapped.model
    data = env.unwrapped.data
    i=0
    reverse=False
    action=[0.3,0.3,0.05]
    action_init=action.copy()
    # MuJoCo offscreen renderer (for gym view)
    renderer = mujoco.Renderer(model, height=480, width=480)

    # Video writer (side-by-side = double width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        "arm.mp4",
        fourcc,
        30,
        (960, 480)  # 2 x 480 width
    )

    try:
        for _ in range(500):
            obs, reward, terminated, truncated, info = env.step(action)

            # -----------------------------
            # 1. Gym / MuJoCo view (left)
            # -----------------------------
            renderer.update_scene(data)
            gym_img = renderer.render()  # (H, W, 3), uint8

            # -----------------------------
            # 2. tactile camera (right)
            # -----------------------------
            tactile_img = obs["image"]

            # ensure correct format
            if tactile_img.dtype != np.uint8:
                tactile_img = (tactile_img * 255).astype(np.uint8)

            # resize both to same height
            gym_img = cv2.resize(gym_img, (480, 480))
            tactile_img = cv2.resize(tactile_img, (480, 480))

            # convert RGB → BGR for OpenCV
            gym_img = cv2.cvtColor(gym_img, cv2.COLOR_RGB2BGR)
            tactile_img = cv2.cvtColor(tactile_img, cv2.COLOR_RGB2BGR)

            # -----------------------------
            # 3. stack side-by-side
            # -----------------------------
            combined = np.hstack([gym_img, tactile_img])

            # write frame
            video.write(combined)

            if terminated or truncated:
                obs, info = env.reset()

            if not reverse: action[1]=action_init[1]+i
            else: action[1]=action_init[1]-i
            if i>=0.1:
                i=0
                reverse= not reverse
            i+=0.001
    finally:
        video.release()
        env.close()
        print("Saved rollout.mp4")

if __name__ == "__main__":
    main()