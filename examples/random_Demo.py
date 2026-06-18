import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()

import mujoco.viewer
import cv2



def main():

    env = gym.make("RandomTac-v0")

    obs, info = env.reset()

    # IMPORTANT: use underlying MuJoCo model/data
    model = env.unwrapped.model
    data = env.unwrapped.data

    # Passive viewer = you control simulation loop
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # Sample random action
            action = env.action_space.sample()

            # Step simulation via your Gym env
            obs, reward, terminated, truncated, info = env.step([])

            # Sync viewer with updated physics
            viewer.sync()

            # reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()
            img = obs["image"]
            cv2.imshow("sensor_cam", img)
            cv2.waitKey(1)
            # small sleep so it doesn't max CPU
            time.sleep(env.unwrapped.model.opt.timestep)


if __name__ == "__main__":
    main()