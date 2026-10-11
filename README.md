# TacTip-gym

MuJoCo simulation of a TacTip tactile sensor (based on the <a href="https://github.com/shepai/Robot_foot">robot foot repo</a>) with a set of reinforcement learning tasks built on a robot arm.

The simulator models the TacTip's silicone skin as a mesh of points with a flexible structure, so the sensor deforms like a soft body. Our <a href="https://shepai.github.io/tutorials/softbodies.html">tutorial page</a> explains how this was built.

![TacTip in simulator](https://raw.githubusercontent.com/shepai/TacTip-gym/refs/heads/main/assets/example.gif)
![TacTip in simulator](https://raw.githubusercontent.com/shepai/TacTip-gym/refs/heads/main/assets/armgif.gif)

## Features

- Gymnasium-style API (`reset()` / `step()`), compatible with PPO and other standard RL libraries
- Soft-body TacTip simulation with marker-based tactile observations
- A 6-DoF robot arm controlled through the fingertip position using inverse kinematics
- Four manipulation tasks of increasing difficulty: edge following, peg pushing, shape insertion and incline pushing
- Randomised starts, shapes and surface friction, so policies cannot memorise a single scene
- An optional event-based tactile layer: a dense grid of touch sensors ("taxels") over every object and plane, read out as neuromorphic-style events

## Installation

Navigate to the folder where you downloaded the repo and run:

```bash
pip install -e .
```

## Environments

Every task builds on a shared `RobotArmEnv` base class, which owns the arm, the controller and the reset logic. Each task supplies its own scene, observation, reward and termination rule.

| Task | Goal | Observation | Action | Episode ends when |
|------|------|-------------|--------|-------------------|
| Edge following | Follow the top edges of a randomly rotated block | 213 (TacTip marker positions) | 3 (fingertip move, Z locked) | Step limit |
| `Peg` | Push an upright cylinder into a hole | 8 | 3 (Z locked) | Cylinder drops in, tips over or leaves the platform |
| `ShapeInsertion` | Push one of four shapes into its matching hole, in the right orientation | 15 | 3 (Z locked) | Shape drops through the correct hole, or fails |
| `Incline` | Push a ball up a ramp whose friction changes each episode | 12 (13 with friction visible) | 2 (along / across the slope) | Ball reaches the top or rolls off |

There are also two basic environments that are registered with Gymnasium:

- `RandomTac-v0`: the standalone TacTip with randomly placed boxes
- `RobotArm-v0`: the TacTip on the robot arm

### How the arm is controlled

The policy does not command joints. It outputs a movement of the fingertip, which is scaled and added to the current fingertip position:

```python
self.current_position += action * 0.5
```

The environment then solves inverse kinematics for joints 1 to 6, keeping the end-effector orientation fixed, and applies the resulting joint targets in MuJoCo. In the pushing tasks the Z component of the action is forced to zero, which reduces the problem to pushing in the horizontal plane. `Incline` is the exception: the fingertip height follows the slope, so its action is 2D.

### Edge following

The arm follows the top edges of a rectangular block. The block is randomly rotated each episode and the start point is chosen on one of its four top edges. The observation comes from `get_nodes().flatten()`, the TacTip marker positions embedded into a compact space.

Reward combines three terms: distance from the edge (`-2.0 * edge_error`), progress along the edge (`+5.0 * delta_progress`) and a contact penalty. The contact force is currently fixed at a placeholder value.

### Peg

An upright cylinder (4 cm wide) sits on a platform with a 6 cm hole (1 cm clearance per side). The cylinder starts at a random position behind the hole. The agent must push it into the hole without tipping it.

- **Observation (8):** fingertip position, cylinder position, vector from cylinder to hole
- **Reward:** `10 * (progress towards hole) + 1 * (progress towards push point) - 0.01 per step`
- **Success:** +100 when the cylinder drops into the hole
- **Fail:** -50 and the episode ends if the cylinder tilts more than about 30 degrees or leaves the platform

MuJoCo cannot cut a hole in a solid, so the platform is built from four box slabs around a gap.

### ShapeInsertion

The platform has four holes (circle, square, rectangle and triangle). Each episode one of the four shapes is shown, with a random start position and rotation. The agent must push it to the matching hole and rotate it so that it fits.

- **Hidden shapes:** all four shapes exist in the scene, but three are hidden each episode (transparent, collisions off, gravity compensated).
- **Generated holes:** `shape_config.py` writes the scene XML from one shared configuration, so the Python code and the geometry cannot drift apart. Run `python shape_config.py` to regenerate it.
- **Observation (15):** fingertip position, shape position, sine and cosine of yaw, vector to target hole, yaw error, one-hot shape identity.
- **Symmetry-aware yaw error:** the yaw error wraps by each shape's rotational symmetry (circle: none, square: 90 degrees, rectangle: 180 degrees, triangle: 120 degrees), so rotations that already fit are not penalised.
- **Reward:** progress towards the hole, progress towards the right orientation, a weak approach term, and a time penalty. +100 for dropping through the correct hole, -50 for a wrong hole, off the edge, or tipping away from the target.

Orientation is enforced by the physics: a square rotated 45 degrees cannot drop through the square hole.

### Incline

A ball starts at the bottom of a 15 degree ramp and must be pushed to the top. The ramp friction is randomised each episode:

- sliding friction from 0.05 to 1.0 (log scale)
- rolling friction from 0 to 0.008 (linear)

Low friction means the ball rolls back down as soon as the push stops, so the agent has to push continuously. High friction means the ball holds its place, so short pushes are enough. The friction is hidden from the policy by default, so it has to be inferred from how the ball moves.

- **Observation (12):** fingertip position, ball position, ball velocity, progress along the ramp, fingertip offset from the ball. Set `OBSERVE_FRICTION = True` to add the friction level as a 13th value (useful for checking the task is learnable).
- **Reward:** `30 * (progress up the ramp) + weak approach term - 0.01 per step`; +100 at the top, -50 if the ball leaves the ramp.
- **Evaluation:** fix the friction with `env.reset(options={"friction": 0.0})`.

## Quick start

```python
import time
import gymnasium as gym
import tactip_mujoco_gym  # triggers register()

import mujoco.viewer

env = gym.make("RandomTac-v0")
model = env.unwrapped.model
data = env.unwrapped.data
```

To see the simulation, open a MuJoCo viewer window. Inside the main loop you choose the action (what you want the robot to do), step the environment (which applies the command and advances the physics), and sync the viewer. For `RandomTac-v0` there are no actions that make changes, because it is just the basic sensor. The robot arm has a real action space.

For the robot arm, the observation is a dictionary containing tactile sensor images:

| Space | Type | Shape | Description |
|-------|------|-------|-------------|
| Observation | Dict | `{"image": (H, W, C)}` | Tactile sensor image observation |
| Action | Box | `(6,)` | Six continuous motor commands |

```python
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        # Sample random action
        action = env.action_space.sample()

        # Step simulation via your Gym env
        obs, reward, terminated, truncated, info = env.step([])

        # Sync viewer with updated physics
        viewer.sync()

        # Reset if episode ends
        if terminated:
            obs, info = env.reset()
```

You can view the observed image by adding this inside the loop:

```python
img = obs["image"]
cv2.imshow("sensor_cam", img)
cv2.waitKey(1)
# small sleep so it doesn't max CPU
time.sleep(env.unwrapped.model.opt.timestep)
```

## Event-based tactile layer (taxels)

On top of the TacTip, any object or plane in a scene can be covered in a dense grid of touch sensors, called taxels. Each taxel reports the normal force on its patch of surface. These can be read as raw forces or as **events**, which only fire when a taxel's force changes, like a neuromorphic or event-camera sensor. A steady load, such as an object resting on the platform, produces no events.

### 1. Generate the taxels

`add_events.py` (in `src/tactip_mujoco_gym/assets/scripts/`) reads a worldbody include file, adds thin touch-sensor sites over every geom, and writes a file of matching `<touch>` sensors:

```bash
python add_events.py shape_worldbody.xml --assets shape_assets.xml \
       --out shape_worldbody_tx.xml --sensors shape_sensors.xml
```

- Supported geoms: boxes, cylinders, spheres and the triangular prism mesh. Planes are skipped.
- The input must be a file with a `<mujoco>` root and `<body>` elements, which is the include-file pattern used by the scenes. Move inline bodies (such as the edge-following block) into their own file first.
- Spacing and covered faces are set by the `RULES` table at the top of the script. By default objects get 4 mm spacing over the whole surface, and platforms and ramps get 10 mm on the top face only.
- The script prints the taxel count per geom. Thousands of taxels slow the simulation, so check the cost and raise the spacing if needed.

Then use the `_tx` file in the scene in place of the original, and include the sensors:

```xml
<worldbody>
  <include file="shape_worldbody_tx.xml"/>
</worldbody>

<sensor>
  <include file="shape_sensors.xml"/>
</sensor>
```

### 2. Turn events on

```python
env = Peg()
env.set_event()        # find the taxels and grow the observation space
model = PPO("MlpPolicy", env)
```

Call `set_event()` **before** creating the policy, because it changes the observation size. `env.set_event(False)` turns it off again.

When events are on, the observation gains one value per taxel: the signed event count since the previous observation (+1 per force increase, -1 per decrease). Events fire when a taxel's force changes by at least `EVENT_THRESHOLD` (0.02 N by default).

### 3. Reading events and forces

| Method | Returns |
|--------|---------|
| `env.last_events` | Events used for the latest observation: `[taxel_index, polarity, time]` rows |
| `env.get_events()` | Events since the last call (do not use this alongside events in the observation, as it empties the buffer) |
| `env.get_taxel_forces()` | Current normal force on every taxel |
| `env.taxel_positions()` | World position of every taxel |
| `env.taxel_bodies()` | Name of the body each taxel is on |

Touch sensors give normal force only (no shear), so the layer produces a pressure map. For finer time resolution in an environment with frame skip, call `self._sample_taxels()` after each physics step.

### Viewing the taxels

The taxel sites are in site group 4, which the viewer hides by default. In the MuJoCo viewer, press **Shift + 4**, or tick group **4** under *Site groups* in the Rendering panel. From Python, set `viewer.opt.sitegroup[4] = 1`.

## Training tips

- **Curriculum:** the harder tasks work best when widened gradually. For `ShapeInsertion`, set `env.spawn_yaw = 0.0` and fix one shape with `env.reset(options={"shape": "square"})`, then add random rotation, then add all four shapes. For `Incline`, start with high friction and widen the range.
- **Hidden friction:** for `Incline` with friction hidden, use frame stacking or a recurrent policy, because a single frame cannot show how the ball is responding.
- **Progress rewards:** the pushing tasks reward the change in distance rather than the distance itself, so the agent is paid for real progress and not for hovering near the goal.
- **Time limits:** treat a time limit as `truncated`, not `terminated`, so the value function bootstraps correctly.

## Adding your own task

A new task needs a MuJoCo scene file, a class that inherits from `RobotArmEnv`, and these methods:

- `reset()`: place the arm and the objects, let them settle, and store any "previous distance" values the reward needs
- `_get_obs()`: build the observation (end with `return self._with_events(obs)` to support taxel events)
- `_reward()`: the reward for the step
- `_done()`: whether the episode has ended

Reset the arm before placing objects, so the arm cannot sweep them out of position.

## Links

- Project page: <a href="https://shepai.github.io">shepai.github.io</a>
- Soft body tutorial: <a href="https://shepai.github.io/tutorials/softbodies.html">shepai.github.io/tutorials/softbodies.html</a>
- Robot foot repo: <a href="https://github.com/shepai/Robot_foot">github.com/shepai/Robot_foot</a>