import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()
import mujoco
import cv2
import numpy as np

def main():
    # 1. Create the environment
    env = gym.make("TacArm-hill")
    obs, info = env.reset()

    # 2. Extract the underlying MuJoCo model and data
    mj_model = env.unwrapped.model
    mj_data = env.unwrapped.data

    # 3. Create a MuJoCo Renderer for offscreen rendering
    width, height = 640, 480
    renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # 4. Set up the Video Writer with your specific absolute path
    video_filename = "/home/dexter/Documents/GitHub/TacTip-gym/assets/tacarm_event.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30  
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    # 5. Configure scene options to toggle on Group 4 (Index 4)
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[4] = 1
    scene_option.sitegroup[4] = 1

    # 6. Initialize an abstract camera to control the pan properties
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    
    # Base camera configuration (adjust these baseline numbers depending on your scene's scale)
    cam.distance = 1.5       # Distance from the camera target look-at point
    cam.elevation = -25.0    # Vertical angle (looking slightly downwards)
    cam.lookat = np.array([0.0, 0.0, 0.1]) # Center target position of your object/arm

    total_steps = 300       # Extended execution length
    print(f"Recording started. Output will save to {video_filename}")

    try:
        for step in range(total_steps):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            # Slowly increase the azimuth angle every frame to create a smooth 360-pan effect
            # 0.3 degrees per step results in 300 degrees of total rotation over 1000 steps
            cam.azimuth = (step * 0.3) % 360

            # Update renderer using our custom abstract camera object instead of integer index
            renderer.update_scene(mj_data, camera=cam, scene_option=scene_option)
            
            # Render and convert format for OpenCV
            frame = renderer.render()
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_frame)

            # Force environment reset if it ends prematurely, ensuring it runs full duration
            if terminated or truncated:
                obs, info = env.reset()
                
    finally:
        # Clean up resources safely
        video_writer.release()
        env.close()
        print(f"Recording complete! File saved successfully.")

if __name__ == "__main__":
    main()
