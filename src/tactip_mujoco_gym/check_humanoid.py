"""
Checks for TacHumanoid-pick before training - run it after changing the robot, sensor or
constants.

    python check_humanoid.py                    # property mixes at level 0, then a level sweep
    python check_humanoid.py --object soft_ball --episodes 3
    python check_humanoid.py --no-sweep         # just the property mixes

Reports the model size and speed, then runs the scripted bimanual pick-and-place
(tactip_humanoid.ScriptedPicker, which reads the simulator's state): square up to the object,
approach, squeeze until the grip force suits its weight and friction, lift, carry (turning the
base if the target is to the side), lower, release. If the script can't do it, a learned
policy won't either. Each run prints the outcome, the peak grip force and squash, and how far
the TacTip markers moved while holding.

    1) every object, light + grippy and heavy + slippery, at level 0
    2) every object with random properties at levels 0.5 and 1 (any angle, up to 1 kg,
       slippery, target off to the side)
"""
import argparse
import time

import numpy as np

from tactip_mujoco_gym.envs.tactip_humanoid import N_PADS, OBJECT_NAMES, SOFT_OBJECTS, ScriptedPicker, TacHumanoidPick


def run_episode(env, options, seed=0, max_steps=None):
    obs, _ = env.reset(seed=seed, options=options)
    pol = ScriptedPicker(env)
    raw = env.unwrapped
    peak_f, peak_strain, held_shift, R = 0.0, 0.0, [], 0.0
    phases, steps = [], 0
    for _ in range(max_steps or raw.max_steps):
        obs, r, term, trunc, info = env.step(pol())
        steps += 1
        R += r
        peak_f = max(peak_f, raw._hand_force.max())
        peak_strain = max(peak_strain, raw._obj[4])
        if pol.phase in ("lift", "carry"):
            held_shift.append(np.linalg.norm(raw.get_hand_nodes(), axis=2).max() * 1e3)
        if not phases or phases[-1] != pol.phase:
            phases.append(pol.phase)
        if term or trunc:
            break
    outcome = "success" if info["is_success"] else (info["fail_reason"] or f"timeout in {pol.phase}")
    return dict(outcome=outcome, steps=steps, ret=R, peak_force=peak_f, peak_strain=peak_strain,
                shift=np.mean(held_shift) if held_shift else 0.0, phases=phases)


def report(name, label, r, sps):
    print(f"{name:<14} {label:<16}: {r['outcome']:<26} {r['steps']:4d} steps, return {r['ret']:6.1f}, "
          f"grip up to {r['peak_force']:5.1f} N, squash {r['peak_strain']:.2f}, "
          f"TacTip shift while held {r['shift']:.1f} mm ({sps:.0f} steps/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object", default=None, choices=OBJECT_NAMES)
    ap.add_argument("--episodes", type=int, default=1, help="per object and property mix")
    ap.add_argument("--no-sweep", action="store_true", help="skip the level 0.5 / 1 sweep")
    args = ap.parse_args()

    env = TacHumanoidPick(obs_modes={"tactile": True}, level=0.0)
    m = env.model
    print(f"model      : {m.nbody} bodies, {m.nv} dofs, {len(SOFT_OBJECTS)} soft objects ({N_PADS} shared pads), "
          f"{env.n_markers} TacTip markers ({env.n_markers // 2} per hand), "
          f"observation {env.observation_space.shape[0]} values")
    names = [args.object] if args.object else OBJECT_NAMES
    mixes = [("light, grippy", dict(mass=0.1, friction=1.0)),
             ("heavy, slippery", dict(mass=0.8, friction=0.35))]
    results = []
    print("\n1) level 0, set properties")
    for name in names:
        for label, props in mixes:
            for ep in range(args.episodes):
                t = time.time()
                r = run_episode(env, dict(object=name, **props), seed=ep)
                results.append(r["outcome"] == "success")
                report(name, label, r, r["steps"] / (time.time() - t))
    if not args.no_sweep:
        for level in (0.5, 1.0):
            print(f"\n2) level {level}, random properties")
            for name in names:
                for ep in range(args.episodes):
                    t = time.time()
                    r = run_episode(env, dict(object=name, level=level), seed=100 + 7 * names.index(name) + ep)
                    results.append(r["outcome"] == "success")
                    p = env._props
                    label = (f"{p['mass']:.2f}kg mu {p['friction']:.2f}"
                             + (f" {p['stiffness']:.0f}N/m" if np.isfinite(p["stiffness"]) else ""))
                    report(name, label, r, r["steps"] / (time.time() - t))
    print(f"\nscripted success: {np.mean(results):.0%} of {len(results)} episodes")


if __name__ == "__main__":
    main()
