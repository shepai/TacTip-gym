"""
Shared MuJoCo + Gymnasium plumbing for the tactip environments.

TactileGymEnv  - loads the model, owns the tactip marker read-out (get_nodes), the hand-camera
                 read-out (get_vision) and the observation-mode selection. The original
                 constructor signature (xml_subpath, obs_dim, action_dim) and the legacy
                 reset/step/_get_obs are kept so other envs built on it (RobotFoot, RandomTac)
                 behave exactly as before.
TaxelMixin     - optional event-based read-out of the tx_* touch sensors on the objects.

Observation modes (arm tasks)
-----------------------------
Proprioception (the arm's own state) is always included. On top of it, any combination of:

    "privileged"  true task state: object position, goal vector, ... (default)
    "tactile"     tactip marker displacements and their change since the last step
    "vision"      the hand camera image (VISION_CAMERA, VISION_WIDTH x VISION_HEIGHT)

    gym.make("TacArm-peg", obs_modes={"vision": True, "tactile": True, "privileged": False})
    env.unwrapped.set_obs_modes({"tactile": True})     # a dict is a full selection: others off
    env.unwrapped.set_obs_modes(vision=True)           # keywords change only those keys

The selected parts are concatenated into one flat vector (obs_format="flat", default), in the
order proprio, privileged, tactile, vision, events. obs_format="dict" returns them as separate
keys instead, ready for a fusion network (SB3: MultiInputPolicy).

Every change rebuilds env.observation_space, so call these before wrapping the env in a VecEnv /
VecNormalize, or pass obs_modes=... through env_kwargs (see RL_BASE.run).
The earlier privileged=True/False argument and set_privileged() still work:
True = privileged only, False = tactile only.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from importlib.resources import files

OBS_MODES = ("vision", "tactile", "privileged")
DEFAULT_OBS_MODES = {"vision": False, "tactile": False, "privileged": True}
OBS_FORMATS = ("flat", "dict")


class TactileGymEnv(gym.Env):
    """Shared MuJoCo + Gym logic for all tactile environments."""
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    ACTION_BOUND  = 0.02         # legacy envs: Box(-0.02, 0.02). The arm envs use 1.0 and scale
                                 # internally (with 0.02 SB3 clipped every arm action to 2%).
    MARKER_PREFIX = "s_c"        # tactip marker / pin sites
    SENSOR_SITE   = "ee_site"    # sensor frame the markers are expressed in
    SENSOR_CAM    = "sensor_cam"
    MARKER_SCALE  = 1e-3         # marker displacement is reported in mm (keeps obs ~O(1))
    MARKER_CLIP   = 20.0

    VISION_CAMERA    = "handcam"
    VISION_WIDTH     = 64        # keep small: in "flat" format every pixel is an input
    VISION_HEIGHT    = 64        # (>= 36 if you use SB3's default CNN in "dict" format)
    VISION_MODE      = "rgb"     # "rgb", "gray" or "depth"
    VISION_MAX_DEPTH = 1.0       # metres; depth is divided by this in "flat" format
    OBS_MODE_NAMES = OBS_MODES   # a task can add its own inputs (TacHumanoid adds "object")
    VISION_SHADOWS   = False     # shadow + reflection passes: most of the render time, little
    VISION_ANTIALIAS = False     # information at 64 px. Together ~6x faster with software GL.

    def __init__(self, xml_subpath, obs_dim, action_dim):
        super().__init__()
        self.xml_path = files("tactip_mujoco_gym").joinpath(*xml_subpath)
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)

        self.action_space = spaces.Box(low=-self.ACTION_BOUND, high=self.ACTION_BOUND,
                                       shape=(action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),
                                            dtype=np.float32)
        self._physics = None             # dm_control Physics, only built if something asks
        self._renderer = None            # renderers are created lazily: an OpenGL context per
        self._view_renderer = None       # env is slow and breaks headless SubprocVecEnv workers
        self.render_mode = None
        self.step_count = 0
        self.max_steps = 200
        self.obs_modes = dict(DEFAULT_OBS_MODES)
        self.obs_format = "flat"
        self._vision_renderer = None

        m = self.model
        self._ee_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, self.SENSOR_SITE)
        self._marker_ids = np.array(
            [i for i in range(m.nsite) if m.site(i).name.startswith(self.MARKER_PREFIX)], dtype=int)
        self.n_markers = len(self._marker_ids)
        self._nodes_rest = np.zeros((self.n_markers, 3), dtype=np.float32)
        self._nodes_last = np.zeros((self.n_markers, 3), dtype=np.float32)
        self._nodes_delta = np.zeros((self.n_markers, 3), dtype=np.float32)

    # ------------------------------------------------------------------ lazy resources
    @property
    def physics(self):
        if self._physics is None:
            from dm_control import mujoco as dm_mujoco
            self._physics = dm_mujoco.Physics.from_xml_path(str(self.xml_path))
        return self._physics

    @physics.setter
    def physics(self, value):
        self._physics = value

    @property
    def renderer(self):
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=128, width=128)
        return self._renderer

    @renderer.setter
    def renderer(self, value):
        self._renderer = value

    # ------------------------------------------------------------------ legacy API (unchanged)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:-1] = action[:-1]
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        obs = self._get_obs()
        reward = self._reward()
        terminated = self._done()
        truncated = self.step_count >= self.max_steps
        return obs, reward, terminated, truncated, {}

    def set_visibility(self, visible_ids):
        opt = self.renderer._scene_option
        for i in range(mujoco.mjNGROUP):
            on = 1 if i in visible_ids else 0
            opt.geomgroup[i] = opt.sitegroup[i] = opt.jointgroup[i] = on
            opt.tendongroup[i] = opt.actuatorgroup[i] = on
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        return opt

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)
        self.set_visibility([2])
        self.renderer.update_scene(self.data, camera=self.SENSOR_CAM)
        img = self.renderer.render().astype("float32") / 255.0
        obs = {"state": None, "image": img}
        self.set_visibility([0, 1, 2])
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        return obs

    def _reward(self):
        raise NotImplementedError

    def _done(self):
        return False

    # ------------------------------------------------------------------ observation modes
    @classmethod
    def _clean_obs_modes(cls, modes):
        unknown = set(modes) - set(cls.OBS_MODE_NAMES)
        if unknown:
            raise ValueError(f"Unknown observation mode(s) {sorted(unknown)}; "
                             f"choose from {list(cls.OBS_MODE_NAMES)}")
        return {k: bool(v) for k, v in modes.items()}

    @classmethod
    def resolve_obs_modes(cls, obs_modes=None, privileged=None):
        """obs_modes dict (missing keys = False) > legacy privileged flag > default."""
        if obs_modes is not None:
            if privileged is not None:
                import warnings
                warnings.warn("Both obs_modes and privileged were given; using obs_modes.", stacklevel=3)
            modes = {k: False for k in cls.OBS_MODE_NAMES}
            modes.update(cls._clean_obs_modes(obs_modes))
            return modes
        modes = {k: False for k in cls.OBS_MODE_NAMES}
        if privileged is not None:              # earlier API: privileged XOR tactile
            modes.update(tactile=not privileged, privileged=bool(privileged))
        else:
            modes.update(DEFAULT_OBS_MODES)
        return modes

    def set_obs_modes(self, obs_modes=None, **flags):
        """Choose which inputs make up the observation (proprioception is always included).

            env.set_obs_modes({"vision": True, "tactile": True})   # full selection, others off
            env.set_obs_modes(privileged=False)                    # change just these keys
        """
        modes = self.resolve_obs_modes(obs_modes) if obs_modes is not None else dict(self.obs_modes)
        modes.update(self._clean_obs_modes(flags))
        if modes["tactile"] and self.n_markers == 0:
            raise RuntimeError(f"No '{self.MARKER_PREFIX}*' marker sites in the model; "
                               "tactile observations are unavailable.")
        if modes["vision"] and mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, self.VISION_CAMERA) < 0:
            raise RuntimeError(f"Camera '{self.VISION_CAMERA}' not found in the model; "
                               "vision observations are unavailable.")
        self.obs_modes = modes
        self._update_obs_space()
        return self.observation_space

    def set_obs_format(self, obs_format="flat"):
        """"flat": one long vector. "dict": one key per selected input (for fusion networks)."""
        if obs_format not in OBS_FORMATS:
            raise ValueError(f"obs_format must be one of {OBS_FORMATS}, not {obs_format!r}")
        self.obs_format = obs_format
        self._update_obs_space()
        return self.observation_space

    def set_privileged(self, privileged=True):
        """Earlier API, kept for existing scripts: True = privileged state only,
        False = tactile markers only. Vision is left as it is."""
        return self.set_obs_modes(privileged=bool(privileged), tactile=not privileged)

    @property
    def privileged(self):
        return self.obs_modes["privileged"]

    @privileged.setter
    def privileged(self, value):
        self.obs_modes["privileged"] = bool(value)

    def _update_obs_space(self):
        """Size the observation space from an actual observation, so it can never drift out
        of sync with _get_obs (the old hand-written obs_dim values could)."""
        obs = self._get_obs()
        if isinstance(obs, dict):
            self.observation_space = spaces.Dict(
                {k: self._space_for(k, v) for k, v in obs.items()})
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    @staticmethod
    def _space_for(key, value):
        if value.dtype == np.uint8:                         # camera image (rgb / gray)
            return spaces.Box(0, 255, shape=value.shape, dtype=np.uint8)
        return spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)

    def _format_obs(self, parts):
        """parts: ordered {name: array}. Returns one flat float32 vector, or a dict."""
        if self.obs_format == "dict":
            return {k: (v if v.dtype == np.uint8 else v.astype(np.float32)) for k, v in parts.items()}
        flat = []
        for k, v in parts.items():
            if k == "vision":
                v = (v.astype(np.float32) / 255.0 if v.dtype == np.uint8
                     else np.clip(v / self.VISION_MAX_DEPTH, 0.0, 1.0))
            flat.append(np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(flat).astype(np.float32)

    # ------------------------------------------------------------------ hand camera read-out
    def get_vision(self):
        """Hand-camera image used for the "vision" input, shape (H, W, C).
        rgb: uint8 (H, W, 3); gray: uint8 (H, W, 1); depth: float32 metres (H, W, 1).
        Uses its own small renderer, separate from any viewing / video renderer."""
        if self.VISION_MODE not in ("rgb", "gray", "depth"):
            raise ValueError(f"VISION_MODE must be 'rgb', 'gray' or 'depth', not {self.VISION_MODE!r}")
        if self._vision_renderer is None:
            q = self.model.vis.quality
            samples = q.offsamples                  # read when the GL context is made
            if not self.VISION_ANTIALIAS:
                q.offsamples = 0
            try:
                self._vision_renderer = mujoco.Renderer(
                    self.model, height=self.VISION_HEIGHT, width=self.VISION_WIDTH)
            finally:
                q.offsamples = samples              # leave other renderers (videos) unchanged
            if not self.VISION_SHADOWS:
                flags = self._vision_renderer.scene.flags
                flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
                flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            if self.VISION_MODE == "depth":
                self._vision_renderer.enable_depth_rendering()
        r = self._vision_renderer
        r.update_scene(self.data, camera=self.VISION_CAMERA)
        img = r.render()
        if self.VISION_MODE == "depth":
            return img.astype(np.float32)[..., None]
        if self.VISION_MODE == "gray":
            gray = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            return np.clip(gray, 0, 255).astype(np.uint8)[..., None]
        return img.copy()

    # ------------------------------------------------------------------ tactip read-out
    def get_nodes(self):
        """Marker positions in the sensor (ee_site) frame, shape (n_markers, 3)."""
        p = self.data.site_xpos[self._marker_ids]
        c = self.data.site_xpos[self._ee_site]
        R = self.data.site_xmat[self._ee_site].reshape(3, 3)
        return ((p - c) @ R).astype(np.float32)          # row-wise R.T @ (p - c)

    def _capture_marker_rest(self):
        """Call with the sensor in free space: defines zero marker displacement."""
        self._nodes_rest = self.get_nodes()
        self._reset_marker_history()

    def _reset_marker_history(self):
        self._nodes_last = self.get_nodes()
        self._nodes_delta = np.zeros_like(self._nodes_last)

    def _update_marker_history(self):
        """Call once per env step, after the physics."""
        nodes = self.get_nodes()
        self._nodes_delta = nodes - self._nodes_last
        self._nodes_last = nodes

    def _tactile_obs(self):
        """Marker displacement from rest + change since the last step, both in mm."""
        disp = (self._nodes_last - self._nodes_rest) / self.MARKER_SCALE
        vel = self._nodes_delta / self.MARKER_SCALE
        out = np.concatenate([disp.ravel(), vel.ravel()])
        return np.clip(out, -self.MARKER_CLIP, self.MARKER_CLIP).astype(np.float32)

    def get_tactile_image(self):
        """The inside-the-sensor camera image (geom group 2 only), float32 in [0, 1]."""
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[2] = 1
        opt.sitegroup[:] = 0
        opt.sitegroup[2] = 1
        self.renderer.update_scene(self.data, camera=self.SENSOR_CAM, scene_option=opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
        img = self.renderer.render().astype(np.float32) / 255.0
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 1
        return img

    # ------------------------------------------------------------------ rendering
    def render(self):
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._view_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._view_cam)
            self._view_cam.lookat[:] = [0.4, 0.0, 0.05]
            self._view_cam.distance, self._view_cam.azimuth, self._view_cam.elevation = 0.9, 135, -25
        self._view_renderer.update_scene(self.data, camera=self._view_cam)
        return self._view_renderer.render().copy()

    def close(self):
        for r in (self._renderer, self._view_renderer, self._vision_renderer):
            if r is not None:
                r.close()
        self._renderer = self._view_renderer = self._vision_renderer = None
        super().close()


class TaxelMixin(TactileGymEnv):
    """
    Event-based readout of the touch-sensor grid made by add_taxels.py (tx_* sensors).

    A taxel emits an event when its force has changed by at least EVENT_THRESHOLD newtons since
    the last event on that taxel (or since reset); the reference level is then set to the
    current force, like a neuromorphic sensor. A steady load produces no events.

        env.unwrapped.set_event(True)     # adds one value per taxel to the observation:
                                          # signed event count since the previous step
        env.unwrapped.get_events()        # raw events, rows of [taxel_index, polarity, time]

    While events are off and every sensor in the model is a taxel, MuJoCo's sensor evaluation
    is switched off entirely (thousands of touch sensors are a big part of the step cost).
    """
    TAXEL_PREFIX = "tx_"
    EVENT_THRESHOLD = 0.02      # newtons

    def __init__(self, xml_subpath, obs_dim, action_dim):
        super().__init__(xml_subpath, obs_dim, action_dim)
        self.events = False
        self.n_taxels = 0
        self._tx_adr = np.zeros(0, dtype=int)
        self._tx_ref = np.zeros(0)
        self._tx_buf = []
        self._tx_counts = np.zeros(0, dtype=np.float32)
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_SENSOR, i) or ""
                 for i in range(self.model.nsensor)]
        self._only_taxel_sensors = self.model.nsensor > 0 and all(
            n.startswith(self.TAXEL_PREFIX) for n in names)
        self._set_sensor_eval(False)

    def _set_sensor_eval(self, on):
        if not self._only_taxel_sensors:
            return
        bit = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        if on:
            self.model.opt.disableflags &= ~bit
        else:
            self.model.opt.disableflags |= bit

    def set_event(self, on=True, threshold=None):
        if on:
            self._init_taxels(threshold)
            self.events = True
            self._set_sensor_eval(True)
            mujoco.mj_forward(self.model, self.data)
            self._reset_taxels()
        else:
            self.events = False
            self._set_sensor_eval(False)
        self._update_obs_space()
        return self.observation_space

    def _init_taxels(self, threshold=None):
        m = self.model
        ids = [i for i in range(m.nsensor)
               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or "").startswith(self.TAXEL_PREFIX)]
        if not ids:
            raise RuntimeError(
                "No taxel sensors found. Run add_taxels.py and include the generated "
                "sensor file inside <sensor> in the scene XML.")
        ids = np.array(ids, dtype=int)
        self._tx_adr = m.sensor_adr[ids].astype(int)       # position of each reading in sensordata
        self._tx_site = m.sensor_objid[ids].astype(int)    # site each sensor is attached to
        self.taxel_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) for i in ids]
        self.n_taxels = len(ids)
        self._tx_threshold = self.EVENT_THRESHOLD if threshold is None else float(threshold)
        self._tx_ref = np.zeros(self.n_taxels)
        self._tx_buf = []
        self._tx_counts = np.zeros(self.n_taxels, dtype=np.float32)

    def _reset_taxels(self):
        """Take the current forces as the baseline and drop stored events."""
        if self.events:
            self._tx_ref = self.data.sensordata[self._tx_adr].copy()
            self._tx_buf = []
            self._tx_counts[:] = 0.0

    def _sample_taxels(self):
        """Compare against the reference level and store new events. Call after mj_step."""
        if not self.events:
            return
        f = self.data.sensordata[self._tx_adr]
        diff = f - self._tx_ref
        idx = np.flatnonzero(np.abs(diff) >= self._tx_threshold)
        if idx.size:
            pol = np.sign(diff[idx])
            self._tx_buf.append(np.stack([idx.astype(float), pol,
                                          np.full(idx.size, self.data.time)], axis=1))
            self._tx_counts[idx] += pol
            self._tx_ref[idx] = f[idx]

    def _event_obs(self):
        """Signed event count per taxel since the previous env step (then cleared)."""
        out = np.clip(self._tx_counts, -5, 5).astype(np.float32)
        self._tx_counts = np.zeros_like(self._tx_counts)
        return out

    # ---------- public API ----------
    def get_events(self):
        """Events since the last call, as an (N, 3) array: [taxel_index, polarity, time]."""
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
