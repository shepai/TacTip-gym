"""
Sanity checks for the tactip arm environments - run this on YOUR model before training.

    python check_envs.py                 # all tasks
    python check_envs.py TacArm-peg      # one task

It reports, per task:
  * controller: tip drift with zero action, achieved vs commanded speed, IK residual
  * reset: how often the episode starts with the sensor touching the object
  * scripted policy (uses privileged state): success rate -> proves the task is solvable
    with your geometry; if this is ~0, fix the geometry constants before training RL
  * random policy return, and env steps per second (single process)

Constants most likely to need adjusting for your sensor (class attributes in robot_arm.py):
  TARGET_QUAT  sensor orientation the IK holds (unchanged from your code: [1, 0, 0, 0])
  TIP_R        effective radius of the sensor tip (used for push points)
  PUSH_Z       tip height when pushing peg / shapes
  Incline.TIP_H, HOME_TIP, WS_LO / WS_HI
"""
import sys
import time

import gymnasium as gym
import numpy as np

import tactip_mujoco_gym  # noqa: F401
from tactip_mujoco_gym.envs.robot_arm import SHAPES


def unit(v):
    return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------------------------------------------------------- scripted policies
def peg_policy(raw):
    tip, cyl = raw._tip_pos(), raw._cyl_pos()
    d = unit(raw.HOLE_XY - cyl[:2])
    err = cyl[:2] - d * (raw.CYL_R + raw.TIP_R - 0.002) - tip[:2]
    return np.clip(0.35 * d + (err - d * (err @ d)) / raw.ACTION_SCALE, -1, 1)


def shape_policy(raw):
    tip, obj = raw._tip_pos(), raw._obj_pos()
    d = unit(raw._target_xy - obj[:2])
    err = obj[:2] - d * (SHAPES[raw.shape]["bound_r"] + raw.TIP_R - 0.002) - tip[:2]
    speed = 0.3 if np.linalg.norm(raw._target_xy - obj[:2]) > 0.04 else 0.12
    return np.clip(speed * d + 1.5 * (err - d * (err @ d)) / raw.ACTION_SCALE, -1, 1)


def incline_policy(raw):
    s_b, _, y_b = raw._slope_coords(raw._ball_pos())
    s_t, _, y_t = raw._slope_coords(raw._tip_pos())
    v = raw.data.qvel[raw._ball_dadr:raw._ball_dadr + 3]
    gap = s_b - s_t - (raw.BALL_R + raw.TIP_R)
    a_y = np.clip((y_b - y_t + 0.03 * v[1]) / raw.ACTION_SCALE, -1, 1)
    a_s = np.clip(gap / raw.ACTION_SCALE * 0.5 + 0.25, -1, 1) if gap > 0.004 else 0.25
    return np.array([a_s, a_y])


def edge_policy(raw):
    tip = raw._tip_pos()
    _, lat, _, tan = raw._edge_coords(tip[:2])
    n_out = np.array([tan[1], -tan[0]])
    xy = 0.5 * tan - np.clip(lat / raw.ACTION_SCALE[0], -1, 1) * n_out
    dz = tip[2] - raw._top_z - raw._z_ref
    return np.clip([xy[0], xy[1], -dz / raw.ACTION_SCALE[2]], -1, 1)


TASKS = {
    "TacArm-peg":   (peg_policy,     lambda r: r._cyl_body,             {}),
    "TacArm-shape": (shape_policy,   lambda r: r._body[r.shape],        {"spawn_toward_hole": 1.0}),
    "TacArm-hill":  (incline_policy, lambda r: r._ball_body,            {}),
    "TacArm-edge":  (edge_policy,    lambda r: r._block_body,           {}),
}


def rollout(env, policy, seed):
    raw = env.unwrapped
    env.reset(seed=seed)
    R, n = 0.0, 0
    while True:
        a = policy(raw) if policy else env.action_space.sample()
        _, r, te, tr, info = env.step(a)
        R += r
        n += 1
        if te or tr:
            return R, bool(info["is_success"]), n


def controller_check(raw):
    """Free-space checks above the workspace (temporarily widening the workspace box)."""
    lo, hi = raw.WS_LO, raw.WS_HI
    raw.WS_LO, raw.WS_HI = np.full(3, -np.inf), np.full(3, np.inf)
    try:
        start = raw.HOME_TIP.copy()
        raw._place_arm(start)
        p0 = raw._tip_pos()
        for _ in range(100):
            raw._set_target(raw._target)
            raw._drive_arm(raw.FRAME_SKIP)
        drift = np.linalg.norm(raw._tip_pos() - p0) * 1e3
        scale = np.atleast_1d(raw.ACTION_SCALE)[0]
        raw._place_arm(start)
        p0 = raw._tip_pos()
        n = 20
        for _ in range(n):
            raw._set_target(raw._target + np.array([scale, 0, 0]))
            raw._drive_arm(raw.FRAME_SKIP)
        v = (raw._tip_pos() - p0)[0] / (n * raw.FRAME_SKIP * raw.model.opt.timestep)
        v_cmd = scale / (raw.FRAME_SKIP * raw.model.opt.timestep)
        return drift, v, v_cmd, raw._ik_err * 1e3
    finally:
        raw.WS_LO, raw.WS_HI = lo, hi


def main(names):
    for name in names:
        policy, body_fn, script_kw = TASKS[name]
        print(f"\n=== {name}")
        env = gym.make(name)
        raw = env.unwrapped
        drift, v, v_cmd, ik = controller_check(raw)
        flag = "" if drift < 1.0 and abs(v - v_cmd) < 0.15 * v_cmd else "   <-- check controller"
        print(f"controller : zero-action drift {drift:.2f} mm/2s, speed {v:.3f} m/s "
              f"(commanded {v_cmd:.3f}), IK residual {ik:.3f} mm{flag}")

        touching = []
        for s in range(10):
            env.reset(seed=s)
            touching.append(raw._touching(body_fn(raw)))
        if name != "TacArm-hill":
            print(f"reset      : starts touching in {np.mean(touching):.0%} of episodes")

        t = time.time()
        rand = [rollout(env, None, 100 + s) for s in range(5)]
        steps = sum(x[2] for x in rand)
        print(f"random     : return {np.mean([x[0] for x in rand]):7.1f}, success "
              f"{np.mean([x[1] for x in rand]):.0%}   ({steps / (time.time() - t):.0f} steps/s)")
        env.close()

        env = gym.make(name, **script_kw)
        res = [rollout(env, policy, 200 + s) for s in range(10)]
        print(f"scripted   : return {np.mean([x[0] for x in res]):7.1f}, success "
              f"{np.mean([x[1] for x in res]):.0%}" + (f"   (env kwargs {script_kw})" if script_kw else ""))
        env.close()


if __name__ == "__main__":
    main(sys.argv[1:] or list(TASKS))
