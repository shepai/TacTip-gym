"""
Sensor readouts: the base class, mounts, and the registry.

A *readout* turns simulator state into one observation input ("tactile", "vision", ...). It
is separate from the sensor *hardware* (the MuJoCo bodies mounted on the robot, see
sensors/models.py), so the same hardware can be read in several ways (TacTip markers, the
camera inside the TacTip) and your own readout can be added without touching any env.

Life cycle, driven by the environment (every hook gets the env, so `env.model`, `env.data`,
`env.sensor_mounts` and the env's constants are all available):

    bind(env)               once, after the model is compiled: look up ids. Raise
                            SensorUnavailable(reason) if this model can't support the readout
                            (e.g. no marker sites); the env only complains if it is selected.
    calibrate(env, data)    record a rest / reference state (the sensor in free space)
    on_select(env, active)  the input was switched on or off (obs_modes)
    reset(env)              start of every episode, after the scene and robot are placed
    physics_step(env)       after every physics step, only while active and if substeps=True
    env_step(env)           after the physics of every env step
    observe(env)            -> np.ndarray, the observation (called once per env step)
    flatten(value)          how the value goes into the flat ("flat" obs_format) vector
    space(value)            the gymnasium space for the value (default: from its dtype)
    close()                 release renderers etc.

Register a readout so environments and config files can name it:

    from tactip_mujoco_gym.sensors import Sensor, register_sensor

    @register_sensor("wrist_force")
    class WristForce(Sensor):
        key = "force"                       # default observation key
        def __init__(self, scale=10.0):
            self.scale = scale              # store constructor args under the same name
        def bind(self, env): ...
        def observe(self, env): ...

    env = gym.make("TacArm-peg", sensors={"force": "wrist_force"}, obs_modes=["force"])

Readouts can also be named as "package.module:ClassName", or registered by an installed
package through the "tactip_mujoco_gym.sensors" entry-point group.
"""
import copy
import importlib
import inspect
from dataclasses import dataclass

import numpy as np
from gymnasium import spaces


class SensorUnavailable(RuntimeError):
    """The model doesn't have what this readout needs (raised by Sensor.bind)."""


@dataclass
class Mount:
    """One place on the robot where a copy of the sensor hardware sits.

    name    "tip" (arm), "FL" / "FR" / "RL" / "RR" (quadruped feet), "left" / "right" (hands)
    prefix  prefix of every element name in this copy of the sensor ("" for the arm's built-in
            TacTip, "FL_" for the quadruped's front-left foot, ...). A readout finds its sites
            and cameras as prefix + name, e.g. "FL_s_c0" or "left_sensor_cam".
    body    id of the copy's root body (its subtree is the sensor)
    frame   ("site", id) or ("body", id): the default frame readings are expressed in
    """
    name: str
    prefix: str
    body: int
    frame: tuple


def frame_pose(data, frame):
    """(position, 3x3 rotation) of a ("site", id) or ("body", id) frame."""
    kind, i = frame
    if kind == "site":
        return data.site_xpos[i], data.site_xmat[i].reshape(3, 3)
    return data.xpos[i], data.xmat[i].reshape(3, 3)


def subtree_bodies(model, root):
    """Ids of `root` and every body below it."""
    out, stack = [], [int(root)]
    while stack:
        b = stack.pop()
        out.append(b)
        stack.extend(int(c) for c in np.flatnonzero(model.body_parentid == b) if c != b)
    return out


def subtree_geoms(model, root, colliding=True):
    """Geom ids in the subtree of body `root` (only those that can collide, by default)."""
    mask = np.isin(model.geom_bodyid, subtree_bodies(model, root))
    if colliding:
        mask &= (model.geom_contype != 0) | (model.geom_conaffinity != 0)
    return np.flatnonzero(mask)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


class Sensor:
    """Base class for a readout. Subclasses set `key` and implement at least bind/observe."""
    key = None            # default observation key, e.g. "tactile"
    encoder = None        # how VisuoTactilePolicy encodes it: "mlp", "cnn" or None (pass through)
    substeps = False      # True: physics_step() runs after every physics step while active

    # state set by the environment (class defaults, so subclasses needn't call super().__init__)
    available = False
    reason = "not bound to an environment yet"
    active = False

    # ------------------------------------------------------------------ hooks
    def bind(self, env):
        """Look up model ids. Raise SensorUnavailable(reason) if it can't work in this model."""

    def calibrate(self, env, data):
        """Record the rest state from `data` (the sensor untouched)."""

    def on_select(self, env, active):
        """The input was switched on (active=True) or off."""

    def reset(self, env):
        """Start of an episode."""

    def physics_step(self, env):
        """After every physics step (only while active, and only if substeps=True)."""

    def env_step(self, env):
        """After the physics of every env step."""

    def observe(self, env):
        raise NotImplementedError

    def flatten(self, value):
        """Value as part of the flat observation vector."""
        return np.asarray(value, dtype=np.float32).ravel()

    def space(self, value):
        value = np.asarray(value)
        if value.dtype == np.uint8:
            return spaces.Box(0, 255, shape=value.shape, dtype=np.uint8)
        return spaces.Box(-np.inf, np.inf, shape=value.shape, dtype=np.float32)

    def close(self):
        """Release renderers or other resources."""

    # ------------------------------------------------------------------ description
    def spec(self):
        """A JSON-able description that make_sensor() turns back into this readout:
        {"type": registered name or "module:Class", <constructor arguments>}. Constructor
        arguments are read back from attributes of the same name; override this if yours
        are stored differently."""
        out = {"type": sensor_type_name(type(self))}
        try:
            params = inspect.signature(type(self).__init__).parameters
        except (TypeError, ValueError):
            return out
        for name, p in params.items():
            if name == "self" or p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
                continue
            if hasattr(self, name):
                out[name] = _jsonable(getattr(self, name))
        return out

    def __repr__(self):
        args = ", ".join(f"{k}={v!r}" for k, v in self.spec().items() if k != "type")
        return f"{type(self).__name__}({args})"

    def unbound_copy(self):
        """A fresh copy for a new environment (instances passed in are used as templates, so
        several envs in one process never share a readout's state)."""
        try:
            new = copy.deepcopy(self)
        except Exception:                       # e.g. holds a renderer: rebuild from its spec
            kw = self.spec()
            kw.pop("type")
            new = type(self)(**kw)
        for attr in ("available", "reason", "active"):
            new.__dict__.pop(attr, None)
        return new


# ====================================================================== registry
_REGISTRY = {}
_ENTRY_POINTS_LOADED = False
ENTRY_POINT_GROUP = "tactip_mujoco_gym.sensors"


def register_sensor(name, cls=None, overwrite=False):
    """Register a Sensor subclass under `name`. Works as a decorator or a plain call:

        @register_sensor("my_sensor")
        class MySensor(Sensor): ...

        register_sensor("my_sensor", MySensor)
    """
    def deco(c):
        if not (isinstance(c, type) and issubclass(c, Sensor)):
            raise TypeError(f"{c!r} is not a Sensor subclass")
        if name in _REGISTRY and _REGISTRY[name] is not c and not overwrite:
            raise ValueError(f"a sensor called {name!r} is already registered "
                             f"({_REGISTRY[name].__module__}.{_REGISTRY[name].__qualname__}); "
                             "pass overwrite=True to replace it")
        _REGISTRY[name] = c
        return c
    return deco(cls) if cls is not None else deco


def _load_entry_points():
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _ENTRY_POINTS_LOADED = True
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=ENTRY_POINT_GROUP)
    except Exception:                                     # pragma: no cover
        return
    for ep in eps:
        if ep.name in _REGISTRY:
            continue
        try:
            register_sensor(ep.name, ep.load())
        except Exception as err:                          # a broken plugin shouldn't break us
            print(f"[tactip_mujoco_gym] could not load sensor plugin {ep.name!r}: {err}")


def _import_path(path):
    mod, _, attr = path.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split("."):
        obj = getattr(obj, part)
    return obj


def sensor_class(name):
    """Readout class for a registered name or a "package.module:ClassName" path."""
    if isinstance(name, type) and issubclass(name, Sensor):
        return name
    if name in _REGISTRY:
        return _REGISTRY[name]
    if ":" in name:
        cls = _import_path(name)
        if not (isinstance(cls, type) and issubclass(cls, Sensor)):
            raise TypeError(f"{name} is not a Sensor subclass")
        return cls
    _load_entry_points()
    if name in _REGISTRY:
        return _REGISTRY[name]
    raise KeyError(f"unknown sensor {name!r}; registered: {sorted(_REGISTRY)} "
                   "(or give 'package.module:ClassName')")


def sensor_type_name(cls):
    """The registered name of a built-in readout, else the class's "module:Class" import
    path (which works in a fresh process, e.g. when a saved model is loaded, without the
    module that registered it having been imported)."""
    if cls.__module__.startswith("tactip_mujoco_gym."):
        for name, c in _REGISTRY.items():
            if c is cls:
                return name
    return f"{cls.__module__}:{cls.__qualname__}"


def make_sensor(spec, default=None):
    """Build a readout from any of:
        a Sensor instance           copied (instances are templates: every env gets its own)
        a Sensor subclass           instantiated with no arguments
        "name" / "module:Class"     instantiated with no arguments
        {"type": ..., **kwargs}     instantiated with kwargs; "type" may be left out when
                                    `default` (a type for the same slot) is given, which is
                                    how {"width": 96} tweaks a default camera
    """
    if isinstance(spec, Sensor):
        return spec.unbound_copy()
    if isinstance(spec, type) or isinstance(spec, str):
        return sensor_class(spec)()
    if isinstance(spec, dict):
        kw = dict(spec)
        kind = kw.pop("type", None)
        if kind is None:
            if default is None:
                raise ValueError(f"sensor spec {spec!r} needs a 'type'")
            if isinstance(default, Sensor):         # tweak a default instance's arguments
                base = default.spec()
                base.pop("type")
                base.update(kw)
                return type(default)(**base)
            kind = default.get("type") if isinstance(default, dict) else default
            if isinstance(default, dict):
                kw = {**{k: v for k, v in default.items() if k != "type"}, **kw}
        return sensor_class(kind)(**kw)
    raise TypeError(f"can't make a sensor from {spec!r}")


def list_sensors():
    """{name: first line of the class docstring} for every registered readout."""
    _load_entry_points()
    out = {}
    for name, cls in sorted(_REGISTRY.items()):
        doc = (inspect.getdoc(cls) or "").strip().splitlines()
        out[name] = doc[0] if doc else ""
    return out
