"""
A custom sensor readout, end to end: the net contact force on each mounted sensor and where
it acts, as a tactile input. Pair it with rigid_pad.xml (a markerless rigid pad) or use it on
the TacTip as a "force-only" baseline.

    python examples/custom_sensor/pad_contact.py          # (from the repo root) try it

Use it like any built-in readout:

    import gymnasium as gym
    import tactip_mujoco_gym
    from examples.custom_sensor.pad_contact import PadContact

    env = gym.make("TacArm-peg", sensor_xml="examples/custom_sensor/rigid_pad.xml",
                   sensors={"contact": PadContact()}, obs_modes=["contact"])

or by name in a config file (sensors: {contact: "examples.custom_sensor.pad_contact:PadContact"};
the "module:Class" form also works inside SubprocVecEnv workers).

What a readout gets: env.model / env.data (MuJoCo), env.sensor_mounts (one Mount per copy of
the sensor hardware: its name prefix, root body and reference frame) and the env's constants.
"""
import mujoco
import numpy as np

from tactip_mujoco_gym.sensors import (Sensor, SensorUnavailable, frame_pose, register_sensor,
                                       subtree_geoms)


@register_sensor("pad_contact")
class PadContact(Sensor):
    """Net contact force on each mounted sensor (N, in its tip frame) and the force-weighted
    contact point (mm from the tip): 6 values per mount.

    tip_site     site on the sensor whose frame the readings are in (default "sensor_tip";
                 the mount's frame if the sensor has none)
    force_scale  newtons per output unit
    """
    key = "contact"          # default observation key
    encoder = "mlp"          # VisuoTactilePolicy gives it an MLP encoder (None = pass through)

    def __init__(self, tip_site="sensor_tip", force_scale=1.0):
        self.tip_site = tip_site               # constructor arguments stored under the same
        self.force_scale = force_scale         # names, so spec() can save and rebuild it

    def bind(self, env):
        m = env.model
        if not env.sensor_mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")
        self._owner = np.full(m.ngeom, -1, dtype=int)       # geom -> mount index
        self._frames = []
        for k, mt in enumerate(env.sensor_mounts):
            geoms = subtree_geoms(m, mt.body)
            if geoms.size == 0:
                raise SensorUnavailable(f"the sensor on mount {mt.name!r} has no colliding geoms")
            self._owner[geoms] = k
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, mt.prefix + self.tip_site)
            self._frames.append(("site", sid) if sid >= 0 else tuple(mt.frame))
        self._f6 = np.zeros(6)

    def observe(self, env):
        m, d = env.model, env.data
        n_mounts = len(self._frames)
        force = np.zeros((n_mounts, 3))
        moment_pt = np.zeros((n_mounts, 3))
        weight = np.zeros(n_mounts)
        for i in range(d.ncon):
            c = d.contact[i]
            a = self._owner[c.geom1] if c.geom1 >= 0 else -1      # (-1: a flex contact)
            b = self._owner[c.geom2] if c.geom2 >= 0 else -1
            if (a >= 0) == (b >= 0):                    # not touching one of our sensors
                continue
            k, sign = (b, 1.0) if b >= 0 else (a, -1.0)   # contact force acts on geom2
            mujoco.mj_contactForce(m, d, i, self._f6)
            f_world = sign * (c.frame.reshape(3, 3).T @ self._f6[:3])
            force[k] += f_world
            moment_pt[k] += abs(self._f6[0]) * c.pos
            weight[k] += abs(self._f6[0])
        out = []
        for k, frame in enumerate(self._frames):
            p, R = frame_pose(d, frame)
            centre = (moment_pt[k] / weight[k] - p) @ R if weight[k] > 1e-9 else np.zeros(3)
            out += [(force[k] @ R) / self.force_scale, centre * 1e3]
        return np.concatenate(out).astype(np.float32)


if __name__ == "__main__":
    import os
    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401
    from tactip_mujoco_gym.check_envs import peg_policy

    here = os.path.dirname(os.path.abspath(__file__))
    env = gym.make("TacArm-peg", sensor_xml=os.path.join(here, "rigid_pad.xml"),
                   sensors={"contact": PadContact()}, obs_modes=["contact"], obs_format="dict",
                   overrides={"TIP_R": 0.02})
    raw = env.unwrapped
    print("inputs:", {k: ok for k, (ok, _) in raw.describe_inputs().items()})
    obs, _ = env.reset(seed=0)
    for t in range(60):
        obs, r, term, trunc, info = env.step(peg_policy(raw))
        if t % 10 == 0:
            print(f"step {t:3d}  force (N) {np.round(obs['contact'][:3], 2)}  "
                  f"centre (mm) {np.round(obs['contact'][3:], 1)}  reward {r:.2f}")
        if term or trunc:
            print("episode over:", "success" if info["is_success"] else info["fail_reason"])
            break
    env.close()
