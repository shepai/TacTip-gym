"""
Whisker readout: how far each whisker is bent at its base and which way (the direction it
was touched from), whether it touches anything, and how much it vibrates (stick-slip on a
texture, flicking off an edge). For the "whiskers" sensor model (sensors/hardware.py), or any
hardware with joints named <prefix>w<i>_bx / _by at each whisker's base.
"""
from collections import deque

import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, frame_pose, register_sensor, subtree_geoms


@register_sensor("whiskers")
class Whiskers(Sensor):
    """Whiskers: per whisker, base bend (2 axes), contact and vibration, every mount.

    prefix      whisker i's base joints are <mount prefix><prefix><i>_bx and _by (default "w")
    tip_site    site whose frame `layout` is given in (default "sensor_tip")
    angle_ref   base bend (rad) per output unit
    vib_tau     high-pass time constant of the vibration signal (s)
    vib_ref     base angular speed (rad/s; RMS over the env step of the high-passed signal)
                per output unit

    Output: [bend x, bend y, contact (0 / 1), vibration] for each whisker, mounts in turn.
    The bend is the follicle's angle from rest (the whisker as built: its joints at zero,
    gravity compensated), so its direction says where the whisker was pushed from. For plotting, `bend`, `contact` and `vibration` hold the latest values
    per whisker, `vibration_trace` the signal at every physics step, and `layout` each
    whisker's base and rest tip (m, in its mount's tip frame).
    """
    key = "tactile"
    encoder = "mlp"
    substeps = True

    def __init__(self, prefix="w", tip_site="sensor_tip", angle_ref=0.1, vib_tau=0.01, vib_ref=2.0):
        self.prefix = prefix
        self.tip_site = tip_site
        self.angle_ref = angle_ref
        self.vib_tau = vib_tau
        self.vib_ref = vib_ref

    def bind(self, env):
        m = env.model
        mounts = list(getattr(env, "sensor_mounts", []))
        if not mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")
        J, S = mujoco.mjtObj.mjOBJ_JOINT, mujoco.mjtObj.mjOBJ_SITE
        qadr, dadr, owner, bases, tips, self.counts = [], [], [], [], [], []
        self._geom_whisker = np.full(m.ngeom, -1, dtype=int)
        frames = []
        for k, mt in enumerate(mounts):
            i = 0
            while True:
                name = f"{mt.prefix}{self.prefix}{i}"
                jx = mujoco.mj_name2id(m, J, name + "_bx")
                jy = mujoco.mj_name2id(m, J, name + "_by")
                if jx < 0 or jy < 0:
                    break
                qadr.append([m.jnt_qposadr[jx], m.jnt_qposadr[jy]])
                dadr.append([m.jnt_dofadr[jx], m.jnt_dofadr[jy]])
                self._geom_whisker[subtree_geoms(m, m.jnt_bodyid[jx])] = len(owner)
                owner.append(k)
                bases.append(mujoco.mj_name2id(m, S, name + "_base"))
                tips.append(mujoco.mj_name2id(m, S, name + "_tip"))
                i += 1
            if i == 0:
                raise SensorUnavailable(
                    f"no whiskers ({mt.prefix}{self.prefix}0_bx, ...) on the mounted sensor "
                    "(sensor_xml='whiskers')")
            self.counts.append(i)
            sid = mujoco.mj_name2id(m, S, mt.prefix + self.tip_site)
            frames.append(("site", sid) if sid >= 0 else tuple(mt.frame))
        self._qadr = np.array(qadr, dtype=int)
        self._dadr = np.array(dadr, dtype=int)
        self._owner = np.array(owner, dtype=int)
        n = len(owner)
        d = mujoco.MjData(m)                                    # layout at rest
        mujoco.mj_kinematics(m, d)
        self.layout = []
        for k, frame in enumerate(frames):
            c, R = frame_pose(d, frame)
            sel = np.flatnonzero(self._owner == k)
            b = np.array([d.site_xpos[bases[i]] if bases[i] >= 0 else c for i in sel])
            t = np.array([d.site_xpos[tips[i]] if tips[i] >= 0 else c for i in sel])
            self.layout.append({"base": (b - c) @ R, "tip": (t - c) @ R})
        self._rest = np.zeros((n, 2))
        self._a = self.vib_tau / (self.vib_tau + float(m.opt.timestep))
        self.bend = np.zeros((n, 2), np.float32)
        self.contact = np.zeros(n, np.float32)
        self.vibration = np.zeros(n, np.float32)
        self.vibration_trace = deque(maxlen=2000)
        self._hp = np.zeros((n, 2))
        self._w_prev = np.zeros((n, 2))
        self._sq = np.zeros(n)
        self._count = 0

    def reset(self, env):
        self._w_prev = env.data.qvel[self._dadr].copy()
        self._hp[:] = 0.0
        self._sq[:] = 0.0
        self._count = 0
        self.vibration[:] = 0.0
        self.vibration_trace.clear()

    def on_select(self, env, active):
        if active:
            self.reset(env)

    def physics_step(self, env):
        w = env.data.qvel[self._dadr]
        self._hp = self._a * (self._hp + w - self._w_prev)
        self._w_prev = w.copy()
        mag = (self._hp ** 2).sum(1)
        self._sq += mag
        self._count += 1
        self.vibration_trace.append(np.sqrt(mag).astype(np.float32))

    def env_step(self, env):
        if self._count:
            self.vibration[:] = np.sqrt(self._sq / self._count)
        self._sq[:] = 0.0
        self._count = 0

    def touching(self, env):
        """1 for each whisker touching anything, else 0."""
        d = env.data
        out = np.zeros(len(self._owner), np.float32)
        n = d.ncon
        if n:
            for g in (d.contact.geom1[:n], d.contact.geom2[:n]):
                w = np.where(g >= 0, self._geom_whisker[np.maximum(g, 0)], -1)
                out[w[w >= 0]] = 1.0
        return out

    def observe(self, env):
        self.bend = ((env.data.qpos[self._qadr] - self._rest) / self.angle_ref).astype(np.float32)
        self.contact = self.touching(env)
        vib = self.vibration / self.vib_ref
        return np.column_stack([self.bend, self.contact, vib]).ravel().astype(np.float32)
