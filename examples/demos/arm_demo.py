import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()
import mujoco.viewer
import cv2
import numpy as np


def main():

    env = gym.make("RobotArm-v0")

    obs, info = env.reset()

    # IMPORTANT: use underlying MuJoCo model/data
    model = env.unwrapped.model
    data = env.unwrapped.data
    i=0
    reverse=False
    action=np.array([0.3,0.3,0.04])
    action_init=action.copy()
    # Passive viewer = you control simulation loop
    with mujoco.viewer.launch_passive(model, data) as viewer:

        while viewer.is_running():

            # Sample random action

            # Step simulation via your Gym env
            obs, reward, terminated, truncated, info = env.step(action)
                
            # Sync viewer with updated physics
            viewer.sync()

            # reset if episode ends
            if terminated:
                obs, info = env.reset()
            img = env.unwrapped.get_tactile_image()          # camera inside the TacTip
            cv2.imshow("sensor_cam", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
            # small sleep so it doesn't max CPU
            time.sleep(env.unwrapped.model.opt.timestep)
            
            if not reverse: 
                action[1]=action_init[1]+i
                action[2]=action_init[2]+i
            else: 
                action[1]=action_init[1]-i
                action[2]=action_init[2]-i
            if i>=0.1:
                i=0
                reverse= not reverse
            i+=0.001
if __name__ == "__main__":
    main()