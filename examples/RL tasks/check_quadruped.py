"""
Checks for TacQuad-walk before training - run it on YOUR sensor and after changing constants.

    python check_quadruped.py                      # default sensor_stiffness (10)
    python check_quadruped.py --stiffness 20

Reports:
  * the sensor as attached: tip axis, reach, radius, markers, mass; standing height
  * speed (env steps/s, one process) for feet = tactip / proxy / rigid
  * standing with zero action for 3 s on hard, icy, soft, bouncy and rough ground:
    does it stay up, how far does it sag, how deep do the feet sink
  * TacTip loading while standing: marker displacement and how many loaded pins are at the
    end of their travel. Mostly saturated -> raise sensor_stiffness; tiny signal -> lower it
  * an open-loop trot (no learning): proves the legs can step and shows the tactile signal
    following the foot force (stance vs swing)
"""
import argparse
import time

import numpy as np

from tactip_mujoco_gym.envs.tactip_quadruped import TacQuadruped

GROUNDS = {
    "hard":   dict(feel={"friction": 1.0, "softness": 0.0, "dampratio": 1.0}, level=0.0),
    "ice":    dict(feel={"friction": 0.15, "softness": 0.0, "dampratio": 1.0}, level=0.0),
    "soft":   dict(feel={"friction": 0.8, "softness": 1.0, "dampratio": 1.0}, level=0.0),
    "bouncy": dict(feel={"friction": 0.8, "softness": 0.6, "dampratio": 0.3}, level=0.0),
    "rough":  dict(feel=None, level=1.0),
}


def trot(t, f=3.0, thigh=0.5, knee=0.9):
    """Open-loop diagonal trot in action units (FL+RR, then FR+RL)."""
    a = np.zeros(12)
    for k, ph in enumerate([0, np.pi, np.pi, 0]):
        th = 2 * np.pi * f * t + ph
        a[3 * k + 1] = thigh * np.sin(th)
        a[3 * k + 2] = -knee * max(0.0, -np.sin(th))
    return a


def steps_per_second(env, n=100):
    env.reset(seed=0)
    t = time.time()
    for i in range(n):
        env.step(trot(i * env.dt))
    return n / (time.time() - t)


def pin_saturation(env):
    """Fraction of loaded sensor slide/hinge joints sitting at a travel limit (within 10%)."""
    m, d = env.model, env.data
    sat, loaded = 0, 0
    for leg in ("FL", "FR", "RL", "RR"):
        for j in range(m.njnt):
            name = m.joint(j).name
            if not name.startswith(leg + "_") or not m.jnt_limited[j] or name.endswith("_joint"):
                continue
            lo, hi = m.jnt_range[j]
            q = d.qpos[m.jnt_qposadr[j]]
            if abs(q) > 0.1 * (hi - lo):
                loaded += 1
                sat += min(q - lo, hi - q) < 0.1 * (hi - lo)
    return sat, loaded


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stiffness", type=float, default=10.0, help="sensor_stiffness")
    args = ap.parse_args()

    env = TacQuadruped(feet="tactip", sensor_stiffness=args.stiffness, obs_modes={})
    i = env.foot_info
    print(f"sensor     : '{i['root']}', {i['n_markers']} markers/foot, mass {1e3 * i['mass']:.0f} g, "
          f"tip axis {np.round(i['axis'], 2)}, reach {1e3 * i['reach']:.0f} mm, "
          f"radius {1e3 * i['radius']:.0f} mm, sensor frame "
          f"{'ee_site' if i['has_site'] else 'root body (no ee_site)'}")
    print(f"robot      : {env.robot_mass:.1f} kg, straight standing height {env.stand_height:.3f} m, "
          f"timestep {env.TIMESTEP * 1e3:.1f} ms x {env.FRAME_SKIP} = {1 / env.dt:.0f} Hz policy, "
          f"{env.model.nbody} bodies")

    speeds = {"tactip": steps_per_second(env)}
    for feet in ("proxy", "rigid"):
        e = TacQuadruped(feet=feet, obs_modes={})
        speeds[feet] = steps_per_second(e)
        e.close()
    print("speed      : " + ", ".join(f"{k} {v:.0f}" for k, v in speeds.items())
          + " env steps/s (one process; SubprocVecEnv runs one per core)")

    print("\nstanding still for 3 s (zero action):")
    for name, g in GROUNDS.items():
        env.set_terrain_level(g["level"])
        env.reset(seed=1, options={"feel": g["feel"]} if g["feel"] else None)
        if name == "rough":                        # stand on the nearest rough tile
            t = int(np.flatnonzero(env._tile_rough)[0])
            n = env.N_TILES
            xy = env._origin + (np.array([t % n, t // n]) + 0.5) * env.TILE
            env.data.qpos[:2] = xy
            env.data.qpos[2] += env.terrain_height(*xy) + 0.05
        ended = ""
        for _ in range(int(3 / env.dt)):
            _, _, term, _, info = env.step(np.zeros(12))
            if term:
                ended = info["fail_reason"]
                break
        base = env.data.xpos[env._base]
        _, _, _, grav = env._base_state()
        h = base[2] - env.terrain_height(base[0], base[1])
        feet = env._foot_pos()
        sink = env.terrain_height(feet[:, 0], feet[:, 1]) - env._lowest_foot_z(env.data)
        disp = np.linalg.norm(env.get_foot_nodes(), axis=2).max(axis=1) * 1e3
        sat, loaded = pin_saturation(env)
        print(f"  {name:<7}: {'FELL (' + ended + ')' if ended else 'stands'}, height {h:.3f} m, "
              f"tilt {np.degrees(np.arccos(np.clip(-grav[2], -1, 1))):4.1f} deg, "
              f"sink {1e3 * sink.max():4.1f} mm, max marker shift {disp.max():.1f} mm, "
              f"loaded pins at their limit {sat}/{loaded}")

    print("\nopen-loop trot on hard flat ground, 4 s (not a learned gait):")
    for feet in ("proxy", "tactip"):
        e = env if feet == "tactip" else TacQuadruped(feet=feet, obs_modes={})
        e.set_terrain_level(0.0)
        e.reset(seed=0, options={"feel": GROUNDS["hard"]["feel"]})
        p0, R0 = e.data.xpos[e._base].copy(), e.data.xmat[e._base].reshape(3, 3).copy()
        force, disp = [], []
        for k in range(int(4 / e.dt)):
            e.step(trot(k * e.dt))
            if feet == "tactip":
                force.append(e._foot_force)
                disp.append(np.linalg.norm(e.get_foot_nodes(), axis=2).max(axis=1) * 1e3)
        fwd = (e.data.xpos[e._base] - p0) @ R0[:, 0]
        line = f"  {feet:<7}: moved {fwd:+.2f} m"
        if feet == "tactip":
            F, D = np.array(force).ravel(), np.array(disp).ravel()
            c = np.corrcoef(F, D)[0, 1]
            line += (f"; marker shift {D[F < 1].mean():.2f} mm in swing, {D[F > 20].mean():.2f} mm in "
                     f"stance; correlation with foot force {c:.2f}")
        print(line)
    env.close()


if __name__ == "__main__":
    main()
