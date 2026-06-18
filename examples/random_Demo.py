import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()

import mujoco.viewer


def main():

    env = gym.make("RobotArm-v0")

    obs, info = env.reset()

    # IMPORTANT: use underlying MuJoCo model/data
    model = env.model
    data = env.data

    # Passive viewer = you control simulation loop
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # Sample random action
            action = env.action_space.sample()

            # Step simulation via your Gym env
            obs, reward, terminated, truncated, info = env.step(action)

            # Sync viewer with updated physics
            viewer.sync()

            # reset if episode ends
            if terminated or truncated:
                obs, info = env.reset()

            # small sleep so it doesn't max CPU
            time.sleep(env.model.opt.timestep)


if __name__ == "__main__":
    main()