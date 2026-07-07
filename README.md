# TacTip-gym
Mujoco simulation of a TacTip (based on the <a href="https://github.com/shepai/Robot_foot">robot foot repo</a>) with various robotic tasks.
This simulator uses a mesh of points with a flexible structure, so that we can simulate a soft silicone body. Check out our tutorial information page <a href="https://shepai.github.io/tutorials/softbodies.html">here</a> for information on how we did this. 

![TacTIp in simulator](https://raw.githubusercontent.com/shepai/TacTip-gym/refs/heads/main/assets/rollout-ezgif.com-video-to-gif-converter.gif)
![TacTIp in simulator](https://raw.githubusercontent.com/shepai/TacTip-gym/refs/heads/main/assets/armgif.gif)

## Installation 
Navigate to the folder path of download and use the command
```[bash]
pip install -e .
```

## Using 
Check out the examples, each environment is setup as a gym. Teere is the simple standalone TacTip with boxes called random. We set it up by calling it inwithin the paramters. The gyms we have made are:
- RandomTac-v0
- RobotArm-v0

```[python]
import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()

import mujoco.viewer
import cv2

env = gym.make("RandomTac-v0")
model = env.unwrapped.model
data = env.unwrapped.data

```
Then we can make a mujoco viewer window if we want to visualise it. Inside the main loop is the action (what you want to happen to the robot body), the step (passing commands through and making a physics step) and then updating the simulation. For the randomTac, there are no actions that make changes as its just the basic sensor. The robot chassis do have specific action spaces. 

The robot arm:

The observation space is a dictionary containing tactile sensor images.

| Key | Shape | Type | Range | Description |
|-----|-------|------|-------|-------------|
| tactile_image | (H, W, C) | uint8 | [0, 255] | Image captured from the tactile sensor |

```[python]
with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            # Sample random action
            action = env.action_space.sample()

            # Step simulation via your Gym env
            obs, reward, terminated, truncated, info = env.step([])

            # Sync viewer with updated physics
            viewer.sync()

            # reset if episode ends
            if terminated:
                obs, info = env.reset()
```


