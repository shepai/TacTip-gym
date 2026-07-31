import gymnasium as gym
import numpy as np
from gymnasium import spaces
from mujoco import MjModel, MjData
from mujoco import mj_step, mj_resetData, mj_forward
from importlib.resources import files
from mujoco import renderer
from dm_control import mujoco

class TactileGymEnv(gym.Env):
    """
    Shared MuJoCo + Gym logic for all tactile environments.
    """

    def __init__(self, xml_subpath, obs_dim, action_dim):
        super().__init__()

        self.xml_path = files("tactip_mujoco_gym").joinpath(*xml_subpath)
        self.physics = mujoco.Physics.from_xml_path(str(self.xml_path))
        self.model = MjModel.from_xml_path(str(self.xml_path))
        self.data = MjData(self.model)

        self.action_space = spaces.Box(
            low=-0.02, high=0.02, shape=(action_dim,), dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.renderer = renderer.Renderer(self.model,height=128, width=128)
        self.step_count = 0
        self.max_steps = 200
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj_resetData(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}
    def step(self, action):
        mj_forward(self.model, self.data)
        self.data.ctrl[:-1] = action[:-1]
        mj_step(self.model, self.data)
        self.step_count += 1

        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps

        return obs, reward, terminated, truncated, {}
    def set_visibility(self, visible_ids):
        opt = self.renderer._scene_option
        for i in range(mujoco.mjNGROUP):
            if i in visible_ids:
                opt.geomgroup[i] = 1  # 1 means visible
                opt.sitegroup[i] = 1
                opt.jointgroup[i] = 1
                opt.tendongroup[i] = 1
                opt.actuatorgroup[i] = 1
            else:
                opt.geomgroup[i] = 0  # 0 means hidden
                opt.sitegroup[i] = 0
                opt.jointgroup[i] = 0
                opt.tendongroup[i] = 0
                opt.actuatorgroup[i] = 0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return opt
    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        self.set_visibility([2])
        self.renderer.update_scene(self.data, camera="sensor_cam")
        img = self.renderer.render()
        img = img.astype("float32") / 255.0
        obs = {
        "state": None,
        "image": img
    }
        opt_front=self.set_visibility([0, 1, 2])
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        return obs

    def _reward(self):
        raise NotImplementedError

    def _done(self):
        return False
    def get_nodes(self):
        nodes = []
        
        # 1. Get the world position and 3x3 rotation matrix of your tactile sensor body
        # (Replace "sensor_body_name" with the exact name of your sensor body from your XML)
        sensor_site_id = self.model.site("ee_site").id
        sensor_pos = self.data.site_xpos[sensor_site_id]
        sensor_rot = self.data.site_xmat[sensor_site_id].reshape(3, 3)

        for i in range(self.model.nsite):
            name = self.model.site(i).name

            if name.startswith("s_c"):
                # 2. Get the global position of the tactile node pin
                world_pos = self.data.site_xpos[i]
                
                # 3. Compute relative vector: World Vector = World Pin Pos - World Sensor Center Pos
                relative_world_vec = world_pos - sensor_pos
                
                # 4. Project into local space by multiplying by the inverse (transpose) rotation matrix
                local_pos = sensor_rot.T @ relative_world_vec
                
                nodes.append(local_pos)

        # 5. Explicitly cast to float32 to prevent Gymnasium warnings
        return np.array(nodes, dtype=np.float32)