"""
Event-based readout of the touch-sensor grids ("taxels", tx_* sensors) that
assets/scripts/add_events.py lays over the objects and surfaces of a scene.

A taxel emits an event when its force has changed by at least `threshold` newtons since the
last event on that taxel (or since reset); the reference level is then set to the current
force, like a neuromorphic sensor. A steady load produces no events.

While the input is off and every MuJoCo sensor in the model is a taxel, MuJoCo's sensor
evaluation is switched off entirely (thousands of touch sensors are a big part of the step
cost: ~3.4x faster physics in the peg scene).
"""
import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, register_sensor


@register_sensor("taxel_events")
class TaxelEvents(Sensor):
    """Signed event count per taxel (tx_* touch sensors) since the previous env step.

    prefix     sensor-name prefix of the taxels (default: env.TAXEL_PREFIX, "tx_")
    threshold  force change (N) that fires an event (default: env.EVENT_THRESHOLD, 0.02)
    clip       counts are clipped to +-clip (default 5)

    Taxels are sampled after every physics step, not just every env step.
    """
    key = "events"
    encoder = None
    substeps = True

    def __init__(self, prefix=None, threshold=None, clip=5.0):
        self.prefix = prefix
        self.threshold = threshold
        self.clip = clip

    def bind(self, env):
        m = env.model
        self._model = m
        prefix = self.prefix or getattr(env, "TAXEL_PREFIX", "tx_")
        names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or "" for i in range(m.nsensor)]
        self._only_taxel_sensors = m.nsensor > 0 and all(n.startswith(prefix) for n in names)
        self._set_sensor_eval(False)
        ids = np.array([i for i, n in enumerate(names) if n.startswith(prefix)], dtype=int)
        self._adr = m.sensor_adr[ids].astype(int) if ids.size else np.zeros(0, dtype=int)
        self._site = m.sensor_objid[ids].astype(int) if ids.size else np.zeros(0, dtype=int)
        self.names = [names[i] for i in ids]
        self.n_taxels = len(ids)
        self._ref = np.zeros(self.n_taxels)
        self._buf = []
        self._counts = np.zeros(self.n_taxels, dtype=np.float32)
        self._threshold = float(getattr(env, "EVENT_THRESHOLD", 0.02)
                                if self.threshold is None else self.threshold)
        if not ids.size:
            raise SensorUnavailable(
                "No taxel sensors found. Run assets/scripts/add_events.py and include the "
                "generated sensor file inside <sensor> in the scene XML.")

    def _set_sensor_eval(self, on):
        if not self._only_taxel_sensors:
            return
        bit = int(mujoco.mjtDisableBit.mjDSBL_SENSOR)
        if on:
            self._model.opt.disableflags &= ~bit
        else:
            self._model.opt.disableflags |= bit

    def set_threshold(self, threshold):
        self.threshold = threshold
        self._threshold = float(threshold)

    def on_select(self, env, active):
        if active:
            self._ref = np.zeros(self.n_taxels)
            self._buf = []
            self._counts = np.zeros(self.n_taxels, dtype=np.float32)
            self._set_sensor_eval(True)
            mujoco.mj_forward(env.model, env.data)
            self.reset(env)
        else:
            self._set_sensor_eval(False)

    def reset(self, env):
        """Take the current forces as the baseline and drop stored events."""
        if self.active:
            self._ref = env.data.sensordata[self._adr].copy()
            self._buf = []
            self._counts[:] = 0.0

    def physics_step(self, env):
        f = env.data.sensordata[self._adr]
        diff = f - self._ref
        idx = np.flatnonzero(np.abs(diff) >= self._threshold)
        if idx.size:
            pol = np.sign(diff[idx])
            self._buf.append(np.stack([idx.astype(float), pol,
                                       np.full(idx.size, env.data.time)], axis=1))
            self._counts[idx] += pol
            self._ref[idx] = f[idx]

    def observe(self, env):
        out = np.clip(self._counts, -self.clip, self.clip).astype(np.float32)
        self._counts = np.zeros_like(self._counts)
        return out

    # ------------------------------------------------------------------ extras
    def events(self):
        """Events since the last call, as an (N, 3) array: [taxel_index, polarity, time]."""
        if not self._buf:
            return np.empty((0, 3))
        out = np.concatenate(self._buf)
        self._buf = []
        return out

    def forces(self, env):
        """Current normal force on every taxel (n_taxels,)."""
        return env.data.sensordata[self._adr].copy()

    def positions(self, env):
        """World position of every taxel (n_taxels, 3)."""
        return env.data.site_xpos[self._site].copy()

    def bodies(self, env):
        """Name of the body each taxel is on."""
        ids = env.model.site_bodyid[self._site]
        return [mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_BODY, int(b)) for b in ids]
