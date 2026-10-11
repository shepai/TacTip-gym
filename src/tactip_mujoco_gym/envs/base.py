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

class TaxelMixin(TactileGymEnv):
    """
    Event-based readout of the touch-sensor grid made by add_taxels.py.

    A taxel emits an event when its force has changed by at least EVENT_THRESHOLD newtons since
    the last event on that taxel (or since reset). After an event the taxel's reference level is
    set to its current force, like a neuromorphic / dynamic-vision sensor. A steady load, such as
    an object resting on the platform, produces no events.

    Events are rows of [taxel_index, polarity, time]:
        taxel_index  index into taxel_names / taxel_positions()
        polarity     +1 force increased, -1 force decreased
        time         simulation time in seconds

    Usage in an env class:
        class Peg(TaxelMixin, RobotArmEnv):
            __init__ : call self._init_taxels() at the end
            reset    : call self._reset_taxels() at the very end, after the settling steps
        then call env.get_events() after env.step().
    """
    TAXEL_PREFIX = "tx_"
    EVENT_THRESHOLD = 0.02      # newtons
    def __init__(self, xml_subpath, obs_dim, action_dim):
            super().__init__(xml_subpath, obs_dim, action_dim)
            self.events=False 
    def set_event(self, on=True, threshold=None):
        if on:
            self._init_taxels(threshold)      # raises a clear error if the XML has no taxels
            self.events = True
            self._reset_taxels()
        else:
            self.events = False
        self._update_obs_space()
    def _update_obs_space(self):
        dim = self.base_obs_dim + (self.n_taxels if self.events else 0)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
    def _init_taxels(self, threshold=None):
        m = self.model
        ids = []
        for i in range(m.nsensor):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
            if name.startswith(self.TAXEL_PREFIX):
                ids.append(i)
        if not ids:
            raise RuntimeError(
                "No taxel sensors found. Run add_taxels.py and include the generated "
                "sensor file inside <sensor> in the scene XML.")
        ids = np.array(ids, dtype=int)
        self._tx_adr = m.sensor_adr[ids].astype(int)      # position of each reading in data.sensordata
        self._tx_site = m.sensor_objid[ids].astype(int)   # site each sensor is attached to
        self.taxel_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in ids]
        self.n_taxels = len(ids)
        self._tx_threshold = self.EVENT_THRESHOLD if threshold is None else float(threshold)
        self._tx_ref = np.zeros(self.n_taxels)
        self._tx_buf = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        """Take the current forces as the baseline and drop any stored events."""
        if self.events:
            self._tx_ref = self.data.sensordata[self._tx_adr].copy()
            self._tx_buf = []

    def _sample_taxels(self):
        """Compare against the reference level and store new events. Call after mj_step."""
        f = self.data.sensordata[self._tx_adr]
        diff = f - self._tx_ref
        idx = np.flatnonzero(np.abs(diff) >= self._tx_threshold)
        if idx.size:
            self._tx_buf.append(np.stack([
                idx.astype(float),
                np.sign(diff[idx]),
                np.full(idx.size, self.data.time),
            ], axis=1))
            self._tx_ref[idx] = f[idx]

    # ---------- public API ----------
    def get_events(self):
        """Events since the last call, as an (N, 3) array: [taxel_index, polarity, time]."""
        self._sample_taxels()
        if not self._tx_buf:
            return np.empty((0, 3))
        events = np.concatenate(self._tx_buf)
        self._tx_buf = []
        return events

    def get_taxel_forces(self):
        """Current normal force on every taxel (n_taxels,)."""
        return self.data.sensordata[self._tx_adr].copy()

    def taxel_positions(self):
        """World position of every taxel (n_taxels, 3). Moves with the object each step."""
        return self.data.site_xpos[self._tx_site].copy()

    def taxel_bodies(self):
        """Name of the body each taxel is on, as a list of strings."""
        ids = self.model.site_bodyid[self._tx_site]
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in ids]