"""
Sanity checks for the tactip arm environments - run this on YOUR model before training.

    python check_envs.py                 # all tasks
    python check_envs.py TacArm-peg      # one task
    python check_envs.py TacArm-peg --sensor-xml my_sensor.xml     # with your sensor
    python check_envs.py TacArm-peg --env-kwargs '{"overrides": {"TIP_R": 0.02}}'

It reports, per task:
  * controller: tip drift with zero action, achieved vs commanded speed, IK residual
  * reset: how often the episode starts with the sensor touching the object, how deep it
    starts inside it, and anything else the sensor touches (e.g. the platform)
  * peg only: the sensor radius measured at PUSH_Z, compared with TIP_R
  * zero action: how long an episode lasts if the arm holds still; a few steps means the
    reset itself leaves the object falling / tipping (geometry constants)
  * episode lengths and why episodes fail (info["fail_reason"])
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


def surface_policy(raw):
    st = raw._current()
    gx, _ = raw._base.gradient(st["tip"][0], st["tip"][1])
    a_z = np.clip((-st["err"] + gx * raw.SPEED) / raw.Z_SCALE, -1, 1)   # + feed-forward of the slope
    if not raw.follow_normal:
        return np.array([a_z])
    return np.array([a_z, np.clip((st["pitch_goal"] - st["pitch"]) / raw.PITCH_RATE, -1, 1)])


TASKS = {
    "TacArm-peg":   (peg_policy,     lambda r: r._cyl_body,             {}),
    "TacArm-shape": (shape_policy,   lambda r: r._body[r.shape],        {"spawn_toward_hole": 1.0}),
    "TacArm-hill":  (incline_policy, lambda r: r._ball_body,            {}),
    "TacArm-edge":  (edge_policy,    lambda r: r._block_body,           {}),
    "TacArm-surface": (surface_policy, lambda r: r._surface_body,       {}),
}


def zero_policy(raw):
    return np.zeros(raw.action_space.shape)


def rollout(env, policy, seed):
    """Returns (return, success, steps, outcome) with outcome "success", "timeout" or the
    fail reason."""
    raw = env.unwrapped
    env.reset(seed=seed)
    R, n = 0.0, 0
    while True:
        a = policy(raw) if policy else env.action_space.sample()
        _, r, te, tr, info = env.step(a)
        R += r
        n += 1
        if te or tr:
            ok = bool(info["is_success"])
            outcome = "success" if ok else (info.get("fail_reason") or "fail") if te else "timeout"
            return R, ok, n, outcome


def summarise(label, results, extra=""):
    outcomes = {}
    for x in results:
        outcomes[x[3]] = outcomes.get(x[3], 0) + 1
    ends = ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]))
    print(f"{label:<11}: return {np.mean([x[0] for x in results]):7.1f}, success "
          f"{np.mean([x[1] for x in results]):4.0%}, {np.mean([x[2] for x in results]):5.1f} steps "
          f"[{ends}]{extra}")
    return np.mean([x[2] for x in results])


def reset_contacts(raw, body):
    """(touching, deepest penetration in mm, contact force N, other geoms the arm touches)."""
    m, d = raw.model, raw.data
    idx = raw._contacts_with(body)
    pen = max((-d.contact.dist[i] * 1e3 for i in idx), default=0.0)
    others = set()
    for i in range(d.ncon):
        g1, g2 = int(d.contact.geom1[i]), int(d.contact.geom2[i])
        for ga, gb in ((g1, g2), (g2, g1)):
            if raw._arm_geom[ga] and not raw._arm_geom[gb] and m.geom_bodyid[gb] != body:
                others.add(m.geom(gb).name or f"geom {gb} (body {m.body(m.geom_bodyid[gb]).name})")
    return idx.size > 0, pen, raw._contact_force(body), others


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


def main(names, extra=None):
    extra = dict(extra or {})
    for name in names:
        policy, body_fn, script_kw = TASKS[name]
        print(f"\n=== {name}" + (f"  {extra}" if extra else ""))
        env = gym.make(name, **extra)
        raw = env.unwrapped
        drift, v, v_cmd, ik = controller_check(raw)
        flag = "" if drift < 1.0 and abs(v - v_cmd) < 0.15 * v_cmd else "   <-- check controller"
        print(f"controller : zero-action drift {drift:.2f} mm/2s, speed {v:.3f} m/s "
              f"(commanded {v_cmd:.3f}), IK residual {ik:.3f} mm{flag}")

        touching, pens, forces, others, radii = [], [], [], set(), []
        for s in range(10):
            env.reset(seed=s)
            body = body_fn(raw)
            t_, pen, f, oth = reset_contacts(raw, body)
            touching.append(t_)
            pens.append(pen)
            forces.append(f)
            others |= oth
            if name == "TacArm-peg" and t_:
                radii.append(np.linalg.norm(raw._tip_pos()[:2] - raw._cyl_pos()[:2]) - raw.CYL_R)
        if name != "TacArm-hill":
            flag = "   <-- sensor starts inside the object" if max(pens) > 3.0 else ""
            print(f"reset      : starts touching in {np.mean(touching):.0%} of episodes, "
                  f"penetration up to {max(pens):.1f} mm, force {np.mean(forces):.2f} N{flag}")
        if others:
            print(f"             the arm/sensor also touches: {', '.join(sorted(others))}"
                  "   <-- e.g. PUSH_Z too low for your sensor")
        if radii:
            r = 1e3 * float(np.mean(radii))
            flag = "" if abs(r - 1e3 * raw.TIP_R) < 3.0 else f"   <-- set TIP_R = {r / 1e3:.3f}"
            print(f"             sensor radius measured at PUSH_Z: {r:.1f} mm "
                  f"(TIP_R = {1e3 * raw.TIP_R:.1f} mm){flag}")

        zero = [rollout(env, zero_policy, 300 + s) for s in range(5)]
        n0 = summarise("zero action", zero)
        if n0 < 10 and name != "TacArm-hill":
            print("             <-- episodes end with the arm holding still: the reset leaves the "
                  "object unstable (geometry constants / XML don't match)")

        t = time.time()
        rand = [rollout(env, None, 100 + s) for s in range(5)]
        steps = sum(x[2] for x in rand)
        summarise("random", rand, f"   ({steps / (time.time() - t):.0f} steps/s)")
        env.close()

        env = gym.make(name, **{**extra, **script_kw})
        res = [rollout(env, policy, 200 + s) for s in range(10)]
        summarise("scripted", res, f"   (env kwargs {script_kw})" if script_kw else "")
        env.close()


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser(description="Sanity checks for the arm tasks.")
    ap.add_argument("tasks", nargs="*", help=f"default: all of {list(TASKS)}")
    ap.add_argument("--sensor-xml", default=None, help="sensor hardware (name or MJCF file)")
    ap.add_argument("--sensor-stiffness", type=float, default=None)
    ap.add_argument("--env-kwargs", default="{}", help="other env arguments as JSON")
    a = ap.parse_args(sys.argv[1:])
    kw = json.loads(a.env_kwargs)
    if a.sensor_xml is not None:
        kw["sensor_xml"] = a.sensor_xml
    if a.sensor_stiffness is not None:
        kw["sensor_stiffness"] = a.sensor_stiffness
    main(a.tasks or list(TASKS), kw)
