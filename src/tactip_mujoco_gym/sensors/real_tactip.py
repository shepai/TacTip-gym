"""
The real TacTip as a readout: the simulated TacTip, mapped through a trained sim2real
mapping (sim2real/mapping.py), so a policy in simulation sees what the real sensor would.

    env = gym.make("TacRig-v0", sensors={"real_tactip": {"model": "models/sim2real"}},
                   obs_modes=["real_tactip"])

    # or in place of the simulated camera image, for a policy that reads "tactile_image":
    env = gym.make("TacRig-v0", obs_modes=["tactile_image"],
                   sensors={"tactile_image": {"type": "real_tactip", "model": "models/sim2real"}})

Every environment has the "real_tactip" input. It is unavailable until it has a model: the
`model` argument, or the TACTIP_SIM2REAL_MODEL environment variable (handy with vectorised
envs and experiment configs). A mapping trained on the simulated camera image renders that
camera at the size the dataset used; one trained on markers reads the markers.
"""
import os

import numpy as np

from .base import Sensor, SensorUnavailable, register_sensor
from .tactip import TacTipCamera, TacTipMarkers

ENV_VAR = "TACTIP_SIM2REAL_MODEL"
_LOADED = {}          # (folder, file times, device) -> mapping: envs in one process share it


def _stamp(folder):
    return tuple(sorted((f, os.path.getmtime(os.path.join(folder, f))) for f in os.listdir(folder)
                        if f.startswith(("mapping.json", "weights."))))


def load_shared(folder, device="cpu"):
    """The mapping in `folder`, loaded once per process (reloaded if its files change)."""
    from ..sim2real.mapping import load_mapping
    key = (os.path.abspath(folder), _stamp(folder), str(device))
    if key not in _LOADED:
        _LOADED[key] = load_mapping(folder, device=device)
    return _LOADED[key]


@register_sensor("real_tactip")
class RealTacTip(Sensor):
    """The real TacTip's image, made from the simulated TacTip by a trained sim2real mapping, uint8 (H, W, C x mounts).

    model    folder of a trained mapping (python -m tactip_mujoco_gym.sim2real train-mapping);
             default: the TACTIP_SIM2REAL_MODEL environment variable
    device   torch device the mapping runs on (default "cpu"; "cuda" for a GPU)

    The image is RGB (grey, one channel, for a mapping trained with --gray) at the size the
    mapping was trained to make. With several mounted TacTips (feet, hands) each is mapped
    and the images are stacked along the channel axis, as tactip_camera does.
    """
    key = "real_tactip"
    encoder = "cnn"

    def __init__(self, model=None, device="cpu"):
        self.model = model
        self.device = device

    # ------------------------------------------------------------------ setup
    def bind(self, env):
        path = self.model or os.environ.get(ENV_VAR)
        if not path:
            raise SensorUnavailable(
                "no sim2real mapping: give one with sensors={'real_tactip': {'model': "
                f"'path/to/mapping'}}}} or the {ENV_VAR} environment variable (train one with "
                "python -m tactip_mujoco_gym.sim2real train-mapping)")
        from ..sim2real.mapping import read_config
        from ..sim2real.paired import Preprocess
        try:
            cfg = read_config(path)
        except FileNotFoundError as err:
            raise SensorUnavailable(str(err)) from None
        self._dir = cfg["dir"]
        self._prep = prep = Preprocess.from_dict(cfg["preprocess"])
        if prep.sim_input == "image":
            size = prep.sim_render or prep.sim_size
            inner = TacTipCamera(width=size, height=size, mode="rgb")
        else:
            inner = TacTipMarkers(velocity=False)
        inner.bind(env)                              # SensorUnavailable if there is no TacTip
        if prep.sim_input == "markers" and prep.n_markers and \
                any(c != prep.n_markers for c in inner.counts):
            raise SensorUnavailable(f"the mapping was trained on {prep.n_markers} markers; this "
                                    f"sensor has {inner.counts}")
        self._inner = inner
        self._mounts = max(1, len(getattr(env, "sensor_mounts", [])))
        self._mapping = None

    @property
    def mapping(self):
        """The loaded Sim2RealMapping (loaded on first use, shared within the process)."""
        if getattr(self, "_mapping", None) is None:
            self._mapping = load_shared(self._dir, self.device)
        return self._mapping

    # ------------------------------------------------------------------ life cycle
    def calibrate(self, env, data):
        self._inner.calibrate(env, data)

    def on_select(self, env, active):
        self._inner.on_select(env, active)
        if active:
            _ = self.mapping                         # load now: errors show when it is chosen

    def reset(self, env):
        self._inner.reset(env)

    def env_step(self, env):
        self._inner.env_step(env)

    def sim_input(self, env):
        """What the mapping is given: (mounts, H, W, 3) images or (mounts, n, 3) mm."""
        if self._prep.sim_input == "image":
            img = self._inner.observe(env)
            return np.stack(np.split(img, self._mounts, axis=-1))
        mm = np.asarray(self._inner.observe(env), np.float32) * self._inner._scale * 1000.0
        return mm.reshape(self._mounts, -1, 3)

    def observe(self, env):
        out = self.mapping.sim_to_real(self.sim_input(env))
        return out[0] if len(out) == 1 else np.concatenate(list(out), axis=-1)

    def flatten(self, value):
        return (np.asarray(value, dtype=np.float32) / 255.0).ravel()

    def close(self):
        inner = getattr(self, "_inner", None)
        if inner is not None:
            inner.close()

    def unbound_copy(self):
        return type(self)(model=self.model, device=self.device)
