"""
PressTip readout (Shepherd et al.; shepai.github.io/dev/PressTip.html): the force on each of
its pads under the silicone, the piezoelectric film's vibration signal, and the 3-axis
accelerometer. For the "presstip" sensor model (sensors/hardware.py), or any hardware that
names its parts the same way.
"""
from collections import deque

import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, frame_pose, register_sensor

G = 9.81


@register_sensor("presstip")
class PressTip(Sensor):
    """PressTip: pad forces, piezo vibration and acceleration, every mount.

    pad_prefix   the force pads: sites <prefix><pad_prefix>0, 1, ... (default "pad_")
    surface      the silicone geom whose contacts the pads feel (default "surface")
    board        site whose frame the pads lie in, z out of the board (default "board")
    press_joint  the silicone's press-in joint, which the piezo feels (default "press_z")
    accel        the accelerometer site (default "accel")
    sigma        how far a contact's force spreads through the silicone to the pads (m)
    f_ref        newtons per pad output unit
    piezo_tau    the piezo's high-pass time constant (s): it feels changes, not steady load
    piezo_ref    newtons (high-passed pad force, RMS over the env step) per piezo unit

    Output per mount: [pad forces (15), piezo (1), acceleration x, y, z in g (3)]. The
    accelerometer reads +1 g up at rest, like the real one, averaged over the env step.
    For plotting, `pads`, `piezo`, `accel` and `piezo_trace` (every physics step) hold the
    latest values per mount, and `layout` the pad centres (m, board frame).
    """
    key = "tactile"
    encoder = "mlp"
    substeps = True

    def __init__(self, pad_prefix="pad_", surface="surface", board="board", press_joint="press_z",
                 accel="accel", sigma=0.005, f_ref=2.0, piezo_tau=0.02, piezo_ref=0.05):
        self.pad_prefix = pad_prefix
        self.surface = surface
        self.board = board
        self.press_joint = press_joint
        self.accel = accel
        self.sigma = sigma
        self.f_ref = f_ref
        self.piezo_tau = piezo_tau
        self.piezo_ref = piezo_ref

    # ------------------------------------------------------------------ setup
    def bind(self, env):
        m = env.model
        mounts = list(getattr(env, "sensor_mounts", []))
        if not mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")

        def find(kind, name):
            return mujoco.mj_name2id(m, kind, name)

        S, J, GE = mujoco.mjtObj.mjOBJ_SITE, mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_GEOM
        self._owner = np.full(m.ngeom, -1, dtype=int)
        self._pads, self._frames, self._qadr, self._k, self._acc = [], [], [], [], []
        for k, mt in enumerate(mounts):
            pads = []
            while (sid := find(S, f"{mt.prefix}{self.pad_prefix}{len(pads)}")) >= 0:
                pads.append(sid)
            g = find(GE, mt.prefix + self.surface)
            if not pads or g < 0:
                raise SensorUnavailable(
                    f"no force pads ({mt.prefix + self.pad_prefix}0, ...) or silicone geom "
                    f"({mt.prefix + self.surface}) on the mounted sensor (sensor_xml='presstip')")
            self._owner[g] = k
            self._pads.append(np.array(pads))
            b = find(S, mt.prefix + self.board)
            self._frames.append(("site", b) if b >= 0 else tuple(mt.frame))
            j = find(J, mt.prefix + self.press_joint)
            self._qadr.append(int(m.jnt_qposadr[j]) if j >= 0 else -1)
            self._k.append(float(m.jnt_stiffness[j]) if j >= 0 else 0.0)
            self._acc.append(find(S, mt.prefix + self.accel))
        d = mujoco.MjData(m)                    # the pad layout, from the model at rest
        mujoco.mj_kinematics(m, d)
        self.layout = []
        for pads, frame in zip(self._pads, self._frames, strict=True):
            c, R = frame_pose(d, frame)
            self.layout.append(((d.site_xpos[pads] - c) @ R)[:, :2].copy())
        n = len(mounts)
        self._dt = float(m.opt.timestep)
        self._a = self.piezo_tau / (self.piezo_tau + self._dt)
        self._f6 = np.zeros(6)
        self._v6 = np.zeros(6)
        self.pads = [np.zeros(len(p), np.float32) for p in self._pads]
        self.piezo = np.zeros(n, np.float32)
        self.accel = np.zeros((n, 3), np.float32)
        self.piezo_trace = [deque(maxlen=2000) for _ in range(n)]
        self._hp = np.zeros(n)
        self._f_prev = np.zeros(n)
        self._v_prev = np.zeros((n, 3))
        self._sq = np.zeros(n)
        self._acc_sum = np.zeros((n, 3))
        self._count = 0

    # ------------------------------------------------------------------ physics-rate signals
    def _press_force(self, d, k):
        q = self._qadr[k]
        return -self._k[k] * float(d.qpos[q]) if q >= 0 else 0.0

    def _velocity(self, m, d, k):
        sid = self._acc[k]
        if sid < 0:
            return np.zeros(3)
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, sid, self._v6, 0)
        return self._v6[3:].copy()

    def _still(self, m, d, k):
        """The accelerometer's reading at rest: gravity, in its frame (g)."""
        sid = self._acc[k]
        if sid < 0:
            return np.zeros(3)
        R = d.site_xmat[sid].reshape(3, 3)
        return -(np.asarray(m.opt.gravity) @ R) / G

    def reset(self, env):
        m, d = env.model, env.data
        for k in range(len(self._pads)):
            self._f_prev[k] = self._press_force(d, k)
            self._v_prev[k] = self._velocity(m, d, k)
            self.accel[k] = self._still(m, d, k)
            self.piezo_trace[k].clear()
        self._hp[:] = 0.0
        self.piezo[:] = 0.0
        self._sq[:] = 0.0
        self._acc_sum[:] = 0.0
        self._count = 0

    def on_select(self, env, active):
        if active:
            self.reset(env)

    def physics_step(self, env):
        m, d = env.model, env.data
        g = np.asarray(m.opt.gravity)
        for k in range(len(self._pads)):
            f = self._press_force(d, k)
            self._hp[k] = self._a * (self._hp[k] + f - self._f_prev[k])
            self._f_prev[k] = f
            self.piezo_trace[k].append(self._hp[k])
            sid = self._acc[k]
            if sid >= 0:
                v = self._velocity(m, d, k)
                a = (v - self._v_prev[k]) / self._dt - g          # proper acceleration
                self._v_prev[k] = v
                self._acc_sum[k] += a @ d.site_xmat[sid].reshape(3, 3) / G
        self._sq += self._hp ** 2
        self._count += 1

    def env_step(self, env):
        if self._count:
            self.piezo[:] = np.sqrt(self._sq / self._count)
            self.accel[:] = self._acc_sum / self._count
        self._sq[:] = 0.0
        self._acc_sum[:] = 0.0
        self._count = 0

    # ------------------------------------------------------------------ reading
    def pad_forces(self, env):
        """Force on each pad (N), one array per mount: each contact on the silicone pushes
        the pads around where it acts, spread with a Gaussian of width `sigma`; contacts
        off the edge of the array reach the pads only in part."""
        m, d = env.model, env.data
        out = [np.zeros(len(p)) for p in self._pads]
        n = d.ncon
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            a = np.where(g1 >= 0, self._owner[np.maximum(g1, 0)], -1)
            b = np.where(g2 >= 0, self._owner[np.maximum(g2, 0)], -1)
            frames = [frame_pose(d, f) for f in self._frames]
            for i in np.flatnonzero((a >= 0) ^ (b >= 0)):
                k, sign = (int(b[i]), 1.0) if b[i] >= 0 else (int(a[i]), -1.0)
                mujoco.mj_contactForce(m, d, int(i), self._f6)
                f_world = sign * (d.contact.frame[i].reshape(3, 3).T @ self._f6[:3])
                c, R = frames[k]
                press = -float(f_world @ R[:, 2])                  # pushing towards the board
                if press <= 0.0:
                    continue
                xy = ((d.contact.pos[i] - c) @ R)[:2]
                w = np.exp(-((self.layout[k] - xy) ** 2).sum(1) / (2 * self.sigma ** 2))
                out[k] += press * w / (w.sum() + 0.2)
        return out

    def observe(self, env):
        forces = self.pad_forces(env)
        parts = []
        for k, f in enumerate(forces):
            self.pads[k] = f.astype(np.float32)
            parts += [f / self.f_ref, [self.piezo[k] / self.piezo_ref], self.accel[k]]
        return np.concatenate(parts).astype(np.float32)
