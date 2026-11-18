"""
The "real_tactip" input in the gym: the simulated rig presses a texture block and strokes
across it, and a window shows the simulated TacTip next to what a trained sim2real mapping
says the real TacTip would see.

Train a mapping first (from rig_dataset sessions):

    python -m tactip_mujoco_gym.sim2real train-mapping data/ --out models/sim2real

Then:

    python examples/sim2real/real_tactip_demo.py --model models/sim2real --texture 3
    python examples/sim2real/real_tactip_demo.py --model models/sim2real --save strip.png   # no window

In the window, q or Esc quits.
"""
import argparse

import gymnasium as gym
import numpy as np

import tactip_mujoco_gym  # noqa: F401  (registers the tasks)
from tactip_mujoco_gym.sim2real import SimRig
from tactip_mujoco_gym.sim2real.paired import draw_markers, grid, save_png


def path(rig, depth, stroke, n):
    """Waypoints (raw mm): down into the block, along x, back up."""
    rig.approach(1.0)                               # 1 mm above the surface under the tip
    x, y, z = rig.pos
    top = z - 1.0
    down = [(x, y, top - d) for d in np.linspace(-1.0, depth, n)]
    along = [(x + s, y, top - depth) for s in np.linspace(0.0, stroke, n)]
    up = [(x + stroke, y, top - d) for d in np.linspace(depth, -1.0, max(2, n // 2))]
    return down + along + up


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="models/sim2real", help="a trained mapping's folder")
    ap.add_argument("--texture", type=int, default=1, help="dataset texture block, 1..20")
    ap.add_argument("--depth", type=float, default=1.5, help="press depth (mm)")
    ap.add_argument("--stroke", type=float, default=6.0, help="stroke length along x (mm)")
    ap.add_argument("--steps", type=int, default=30, help="waypoints per part of the path")
    ap.add_argument("--device", default="cpu", help="where the mapping runs (cpu, cuda, ...)")
    ap.add_argument("--save", default=None, help="write a strip of frames to this PNG instead of a window")
    a = ap.parse_args()

    env = gym.make("TacRig-v0", textures=[a.texture],
                   sensors={"real_tactip": {"model": a.model, "device": a.device}})
    rig = SimRig(env=env)
    raw = rig.e
    real = raw.sensors["real_tactip"]
    if not real.available:
        raise SystemExit(f"real_tactip: {real.reason}")
    raw.set_obs_modes(["tactile", "real_tactip"])   # keeps the mapping's inputs up to date
    m = real.mapping
    print(f"{m}; the window shows the simulated TacTip | real_tactip")
    rest = m.info.get("marker_rest_mm")

    def picture():
        sim = real.sim_input(raw)[0]
        if m.sim_input == "markers":
            sim = draw_markers(sim, rest, size=256) if rest is not None else np.zeros((256, 256, 3), np.uint8)
        return grid([[sim, real.observe(raw)]], tile=256, labels=["simulated TacTip", "real_tactip"])

    rig.reset(a.texture, verbose=False)
    strip = []
    cv2 = None
    if not a.save:
        import cv2
    try:
        waypoints = path(rig, a.depth, a.stroke, a.steps)
        for k, p in enumerate(waypoints):
            rig.move_to(*p)
            pic = picture()
            if a.save:
                if k % max(1, len(waypoints) // 8) == 0:
                    strip.append(pic)
                continue
            cv2.imshow("real_tactip", pic[..., ::-1])
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        env.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    if a.save and strip:
        save_png(a.save, np.concatenate(strip, axis=0))
        print(f"wrote {a.save}")


if __name__ == "__main__":
    main()
