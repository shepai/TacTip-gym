"""
Shared MuJoCo + Gymnasium plumbing for the tactip environments.

TactileGymEnv  - loads the model and owns the observation pipeline: which inputs are
                 selected (obs_modes), the sensor readouts that compute them (sensors), the
                 places on the robot the sensor hardware is mounted (sensor_mounts) and the
                 flat / dict observation format. The original constructor signature
                 (xml_subpath, obs_dim, action_dim) and the legacy reset/step/_get_obs are kept
                 so RandomTac behaves exactly as before.
TaxelMixin     - adds the event-based tx_* taxel input ("events") and its API.

Observation inputs
------------------
Proprioception (the robot's own state) is always included. On top of it, any combination of

    task inputs   computed by the task itself: "privileged" (true task state: object pose,
                  goal vector, ...) and, for the humanoid, "object"
    sensor inputs computed by readouts (sensors/), one per key. Defaults on every robot:
                  "tactile"        TacTip marker displacements and their change (all mounts)
                  "tactile_image"  the camera inside each TacTip
                  "vision"         the robot's camera (VISION_CAMERA)
                  arm only: "events" (taxel events on the objects)

    gym.make("TacArm-peg", obs_modes={"vision": True, "tactile": True})   # missing keys = off
    gym.make("TacArm-peg", obs_modes=["vision", "tactile"])               # same, as a list
    gym.make("TacArm-peg", obs_modes=["tactile"], sensors={"tactile": {"clip": 10}})
    gym.make("TacArm-peg", obs_modes=["force"], sensors={"force": MyForceReadout()})
    env.unwrapped.set_obs_modes({"tactile": True})     # a dict / list is a full selection
    env.unwrapped.set_obs_modes(vision=True)           # keywords change only those keys

sensors={key: spec} replaces a default readout, tweaks it ({"width": 96} keeps the type),
adds a new key, or removes one (None). Specs: a registered name, "module:Class", a dict
{"type": ..., **kwargs} or a Sensor instance (copied per env). See sensors/base.py.

The selected parts are concatenated into one flat vector (obs_format="flat", default), in the
order proprio, task inputs, sensor inputs, or kept under their own keys (obs_format="dict"),
ready for a fusion network (SB3: MultiInputPolicy, or vt_encoders.VisuoTactilePolicy).

Every change rebuilds env.observation_space, so call these before wrapping the env in a
VecEnv / VecNormalize, or pass obs_modes=... through env_kwargs (see RL_BASE.run).
The earlier privileged=True/False argument and set_privileged() still work:
True = privileged only, False = tactile only.

overrides={"NAME": value} sets any of the env's UPPER_CASE class constants on this instance
(e.g. {"TIP_R": 0.03, "VISION_WIDTH": 96}), for config files and sensor swaps.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
import mujoco
from importlib.resources import files

from tactip_mujoco_gym.sensors import Camera, SensorUnavailable, TacTipMarkers, make_sensor

OBS_MODES = ("vision", "tactile", "privileged")          # the original three (kept for imports)
DEFAULT_OBS_MODES = {"vision": False, "tactile": False, "privileged": True}
OBS_FORMATS = ("flat", "dict")
RESERVED_KEYS = ("proprio",)


def _like(template, value):
    """Convert a config value (lists from JSON / YAML) to the type of the constant it replaces."""
    if isinstance(template, np.ndarray):
        return np.asarray(value, dtype=template.dtype)
    if isinstance(template, tuple) and isinstance(value, (list, tuple)):
        if len(template) == len(value):
            return tuple(_like(t, v) for t, v in zip(template, value, strict=True))
        return tuple(value)
    if isinstance(template, dict) and isinstance(value, dict):
        return {**template, **value}                   # e.g. a few REWARD_WEIGHTS
    if isinstance(template, bool):
        return bool(value)
    if isinstance(template, float) and isinstance(value, (int, float)):
        return float(value)
    return value


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
    TACTILE_IMAGE_WIDTH = 64     # "tactile_image": the camera inside the TacTip
    TACTILE_IMAGE_HEIGHT = 64

    VISION_CAMERA    = "handcam"
    VISION_WIDTH     = 64        # keep small: in "flat" format every pixel is an input
    VISION_HEIGHT    = 64        # (>= 36 if you use SB3's default CNN in "dict" format)
    VISION_MODE      = "rgb"     # "rgb", "gray" or "depth"
    VISION_MAX_DEPTH = 1.0       # metres; depth is divided by this in "flat" format
    OBS_MODE_NAMES = OBS_MODES   # (legacy; the inputs of an env are env.input_names)
    VISION_SHADOWS   = False     # shadow + reflection passes: most of the render time, little
    VISION_ANTIALIAS = False     # information at 64 px. Together ~6x faster with software GL.

    # inputs the task computes itself (in _obs_parts); everything else comes from a readout
    TASK_INPUTS = ("privileged",)
    # observation key -> default readout (sensors/); env kwarg sensors={...} changes these
    DEFAULT_SENSORS = {"tactile": "tactip_markers", "tactile_image": "tactip_camera",
                       "vision": "camera"}

    def __init__(self, xml_subpath, obs_dim, action_dim, model=None):
        super().__init__()
        self.xml_path = files("tactip_mujoco_gym").joinpath(*xml_subpath)
        self.model = model if model is not None else mujoco.MjModel.from_xml_path(str(self.xml_path))
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
        self._init_sensor_state()
        self._ee_site = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.SENSOR_SITE)

    # ------------------------------------------------------------------ configuration
    def _apply_overrides(self, overrides):
        """overrides={"NAME": value}: set UPPER_CASE class constants on this instance."""
        for name, value in (overrides or {}).items():
            if not name.isupper() or not hasattr(type(self), name):
                consts = sorted(k for k in dir(type(self)) if k.isupper() and not k.startswith("_"))
                raise ValueError(f"{type(self).__name__} has no constant {name!r} to override; "
                                 f"it has {consts}")
            setattr(self, name, _like(getattr(type(self), name), value))

    def _init_sensor_state(self):
        if not hasattr(self, "sensors"):
            self.sensors = {}
        if not hasattr(self, "sensor_mounts"):
            self.sensor_mounts = []
        self._active_sensors = []
        self._substep_sensors = []
        self._legacy_camera = None

    def _setup_sensors(self, sensors=None):
        """Build the readouts (DEFAULT_SENSORS, changed by `sensors`) and bind them to the
        model. Call once the model is compiled and self.sensor_mounts is set."""
        for s in getattr(self, "sensors", {}).values():
            s.close()
        self.sensors = {}
        self._init_sensor_state()
        specs = dict(self.DEFAULT_SENSORS)
        if isinstance(sensors, (list, tuple)):              # [spec, ...]: keys from the readouts
            built = [make_sensor(s) for s in sensors]
            sensors = {}
            for s in built:
                if not s.key:
                    raise ValueError(f"{s!r} has no default key: give sensors as {{key: spec}}")
                sensors[s.key] = s
        for key, spec in (sensors or {}).items():
            if key in RESERVED_KEYS or key in self.TASK_INPUTS:
                raise ValueError(f"{key!r} is computed by the task; pick another key for a sensor")
            if spec is None:
                specs.pop(key, None)
            elif isinstance(spec, dict) and "type" not in spec and key in specs:
                specs[key] = (spec, specs[key])               # tweak the default's arguments
            else:
                specs[key] = spec
        for key, spec in specs.items():
            if isinstance(spec, tuple):
                s = make_sensor(spec[0], default=spec[1])
            else:
                s = make_sensor(spec)
            s.active = False
            try:
                s.bind(self)
                s.available, s.reason = True, ""
            except SensorUnavailable as err:
                s.available, s.reason = False, str(err)
            self.sensors[key] = s

    def _calibrate_sensors(self, data=None):
        """Record every readout's rest state (call with the sensor in free space)."""
        for s in self.sensors.values():
            if s.available:
                s.calibrate(self, self.data if data is None else data)

    def _reset_sensors(self):
        for s in self._active_sensors:
            s.reset(self)

    def _step_sensors(self):
        for s in self._active_sensors:
            s.env_step(self)

    def _mj_step(self, n):
        """n physics steps; readouts that sample every physics step (taxel events) do."""
        if self._substep_sensors:
            for _ in range(n):
                mujoco.mj_step(self.model, self.data)
                for s in self._substep_sensors:
                    s.physics_step(self)
        else:
            mujoco.mj_step(self.model, self.data, n)

    def _sensor_obs(self):
        return {k: s.observe(self) for k, s in self.sensors.items() if s.active}

    def _sync_renderer(self, renderer, new):
        """Called by camera readouts before they render (TacQuad uploads its terrain)."""

    @property
    def input_names(self):
        """Every input that can be selected with obs_modes (proprioception is always on)."""
        return tuple(self.TASK_INPUTS) + tuple(self.sensors)

    def describe_inputs(self):
        """{input: (available, where it comes from or why it is unavailable)}"""
        out = {k: (True, "task") for k in self.TASK_INPUTS}
        for k, s in self.sensors.items():
            out[k] = (s.available, repr(s) if s.available else s.reason)
        return out

    @property
    def obs_modalities(self):
        """{observation key: encoder kind ("mlp" / "cnn")} for the selected sensor inputs
        that want one (used by RL_BASE.run with encoders=...)."""
        return {k: s.encoder for k, s in self.sensors.items() if s.active and s.encoder}

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
    def _clean_obs_modes(self, modes):
        names = self.input_names if self.sensors else self.OBS_MODE_NAMES
        unknown = set(modes) - set(names)
        if unknown:
            raise ValueError(f"Unknown observation mode(s) {sorted(unknown)}; "
                             f"choose from {list(names)}")
        return {k: bool(v) for k, v in modes.items()}

    def resolve_obs_modes(self, obs_modes=None, privileged=None):
        """obs_modes (missing keys = False) > legacy privileged flag > default (privileged).
        obs_modes may be a dict {name: bool}, a list of names, or "tactile+vision"."""
        names = self.input_names if self.sensors else self.OBS_MODE_NAMES
        if isinstance(obs_modes, str):
            obs_modes = [m for m in obs_modes.replace(",", "+").split("+") if m.strip()]
        if isinstance(obs_modes, (list, tuple, set)):
            obs_modes = {str(m).strip(): True for m in obs_modes}
        if obs_modes is not None:
            if privileged is not None:
                import warnings
                warnings.warn("Both obs_modes and privileged were given; using obs_modes.", stacklevel=3)
            modes = {k: False for k in names}
            modes.update(self._clean_obs_modes(obs_modes))
            return modes
        modes = {k: False for k in names}
        if privileged is not None:              # earlier API: privileged XOR tactile
            modes.update(tactile=not privileged, privileged=bool(privileged))
        else:
            modes.update({k: v for k, v in DEFAULT_OBS_MODES.items() if k in modes})
        return modes

    def set_obs_modes(self, obs_modes=None, **flags):
        """Choose which inputs make up the observation (proprioception is always included).

            env.set_obs_modes({"vision": True, "tactile": True})   # full selection, others off
            env.set_obs_modes(["vision", "tactile"])               # the same
            env.set_obs_modes(privileged=False)                    # change just these keys
        """
        modes = self.resolve_obs_modes(obs_modes) if obs_modes is not None else dict(self.obs_modes)
        modes.update(self._clean_obs_modes(flags))
        for k, on in modes.items():
            s = self.sensors.get(k)
            if on and s is not None and not s.available:
                raise RuntimeError(f"{s.reason}; {k} observations are unavailable.")
        self._select(modes)
        self._update_obs_space()
        return self.observation_space

    def _select(self, modes):
        self.obs_modes = dict(modes)
        for k, s in self.sensors.items():
            on = bool(modes.get(k, False))
            if on != s.active:
                s.active = on
                s.on_select(self, on)
        self._active_sensors = [s for s in self.sensors.values() if s.active]
        self._substep_sensors = [s for s in self._active_sensors if s.substeps]

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

    @property
    def events(self):
        """Whether the taxel "events" input is on."""
        return bool(self.obs_modes.get("events", False))

    def _update_obs_space(self):
        """Size the observation space from an actual observation, so it can never drift out
        of sync with _get_obs (the old hand-written obs_dim values could)."""
        obs = self._get_obs()
        if isinstance(obs, dict):
            self.observation_space = spaces.Dict(
                {k: self._space_for(k, v) for k, v in obs.items()})
        else:
            self.observation_space = spaces.Box(-np.inf, np.inf, shape=obs.shape, dtype=np.float32)

    def _space_for(self, key, value):
        s = self.sensors.get(key)
        if s is not None:
            return s.space(value)
        if value.dtype == np.uint8:                         # camera image (rgb / gray)
            return spaces.Box(0, 255, shape=value.shape, dtype=np.uint8)
        return spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)

    def _format_obs(self, parts):
        """parts: ordered {name: array}. Returns one flat float32 vector, or a dict."""
        if self.obs_format == "dict":
            return {k: (v if v.dtype == np.uint8 else v.astype(np.float32)) for k, v in parts.items()}
        flat = []
        for k, v in parts.items():
            s = self.sensors.get(k)
            flat.append(s.flatten(v) if s is not None else np.asarray(v, dtype=np.float32).ravel())
        return np.concatenate(flat).astype(np.float32)

    # ------------------------------------------------------------------ readout shortcuts
    def get_vision(self):
        """Image of the "vision" input, shape (H, W, C): uint8 rgb / gray, float32 depth (m).
        Uses its own small renderer, separate from any viewing / video renderer."""
        s = self.sensors.get("vision")
        if isinstance(s, Camera) and s.available:
            return s.observe(self)
        if self._legacy_camera is None:
            cam = Camera()
            try:
                cam.bind(self)
            except SensorUnavailable as err:
                raise RuntimeError(str(err)) from None
            self._legacy_camera = cam
        return self._legacy_camera.observe(self)

    def _markers(self):
        s = self.sensors.get("tactile")
        return s if isinstance(s, TacTipMarkers) and s.available else None

    @property
    def n_markers(self):
        s = self._markers()
        return s.n_markers if s is not None else 0

    def get_nodes(self):
        """Marker positions in their mount's sensor frame (ee_site), shape (n_markers, 3)."""
        s = self._markers()
        return s.nodes(self.data) if s is not None else np.zeros((0, 3), dtype=np.float32)

    @property
    def _nodes_rest(self):
        s = self._markers()
        return s.rest if s is not None else np.zeros((0, 3), dtype=np.float32)

    def _tactile_obs(self):
        """Marker displacement from rest + change since the last step, both in mm."""
        return self.sensors["tactile"].observe(self)

    # (earlier names for the readout life cycle, for subclasses written against them)
    def _capture_marker_rest(self):
        self._calibrate_sensors()

    def _reset_marker_history(self):
        s = self._markers()
        if s is not None:
            s.reset(self)

    def _update_marker_history(self):
        s = self._markers()
        if s is not None:
            s.env_step(self)

    def get_tactile_image(self):
        """The inside-the-sensor camera image (geom group 2 only), float32 in [0, 1]."""
        prefix = self.sensor_mounts[0].prefix if self.sensor_mounts else ""
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.geomgroup[2] = 1
        opt.sitegroup[:] = 0
        opt.sitegroup[2] = 1
        self.renderer.update_scene(self.data, camera=prefix + self.SENSOR_CAM, scene_option=opt)
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
        for s in list(getattr(self, "sensors", {}).values()):
            s.close()
        if getattr(self, "_legacy_camera", None) is not None:
            self._legacy_camera.close()
        super().close()


class TaxelMixin(TactileGymEnv):
    """
    Event-based readout of the touch-sensor grid made by add_events.py (tx_* sensors), as the
    "events" input (readout: sensors/taxels.py).

    A taxel emits an event when its force has changed by at least EVENT_THRESHOLD newtons since
    the last event on that taxel (or since reset); the reference level is then set to the
    current force, like a neuromorphic sensor. A steady load produces no events.

        env.unwrapped.set_event(True)     # = set_obs_modes(events=True): one value per taxel,
                                          # the signed event count since the previous step
        env.unwrapped.get_events()        # raw events, rows of [taxel_index, polarity, time]

    While events are off and every sensor in the model is a taxel, MuJoCo's sensor evaluation
    is switched off entirely (thousands of touch sensors are a big part of the step cost).
    """
    TAXEL_PREFIX = "tx_"
    EVENT_THRESHOLD = 0.02      # newtons
    DEFAULT_SENSORS = {**TactileGymEnv.DEFAULT_SENSORS, "events": "taxel_events"}

    def _taxels(self):
        s = self.sensors.get("events")
        if s is None or not s.available:
            raise RuntimeError(s.reason if s is not None else "this env has no 'events' input")
        return s

    def set_event(self, on=True, threshold=None):
        if on and threshold is not None:
            self._taxels().set_threshold(threshold)
        return self.set_obs_modes(events=bool(on))

    # ---------- public API ----------
    def get_events(self):
        """Events since the last call, as an (N, 3) array: [taxel_index, polarity, time]."""
        return self._taxels().events()

    def get_taxel_forces(self):
        """Current normal force on every taxel (n_taxels,)."""
        return self._taxels().forces(self)

    def taxel_positions(self):
        """World position of every taxel (n_taxels, 3). Moves with the object each step."""
        return self._taxels().positions(self)

    def taxel_bodies(self):
        """Name of the body each taxel is on, as a list of strings."""
        return self._taxels().bodies(self)

    @property
    def taxel_names(self):
        return self._taxels().names

    @property
    def n_taxels(self):
        s = self.sensors.get("events")
        return s.n_taxels if s is not None and s.available else 0
