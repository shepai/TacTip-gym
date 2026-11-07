"""
Watch the quadruped environment (TacQuad-walk) in the MuJoCo viewer.

    python demo_quadruped.py                        # open-loop trot, TacTip feet, mixed terrain
    python demo_quadruped.py --policy stand         # hold the standing pose
    python demo_quadruped.py --feel ice             # every tile icy (also: hard, soft, bouncy)
    python demo_quadruped.py --level 1              # hardest terrain mix (rough, slippery, soft)
    python demo_quadruped.py --feet proxy           # sphere feet (fast) / --feet rigid
    python demo_quadruped.py --model quad_tactip_tactile    # a policy trained by quadruped_walk.py
    python demo_quadruped.py --video demo.mp4 --seconds 10  # no window: record a video instead

Keys in the viewer:  R reset   P next policy (trot / stand / random)   L next terrain level
The console shows the command, the speed, the force on each foot and how far each TacTip's
markers have moved. On macOS run it with `mjpython` instead of `python`.
"""
import argparse
import time

import numpy as np

from tactip_mujoco_gym.envs.tactip_quadruped import TacQuadruped

FEELS = {
    "hard":   {"friction": 1.0, "softness": 0.0, "dampratio": 1.0},
    "ice":    {"friction": 0.15, "softness": 0.0, "dampratio": 1.0},
    "soft":   {"friction": 0.8, "softness": 1.0, "dampratio": 1.0},
    "bouncy": {"friction": 0.8, "softness": 0.6, "dampratio": 0.3},
}
POLICIES = ("trot", "stand", "random")
LEVELS = (0.0, 0.5, 1.0)


def trot(t, f=3.0, thigh=0.5, knee=0.9):
    """Open-loop diagonal trot (FL+RR, then FR+RL). Not a learned gait: it shuffles forward."""
    a = np.zeros(12)
    for k, ph in enumerate([0, np.pi, np.pi, 0]):
        th = 2 * np.pi * f * t + ph
        a[3 * k + 1] = thigh * np.sin(th)
        a[3 * k + 2] = -knee * max(0.0, -np.sin(th))
    return a


class Demo:
    def __init__(self, args):
        self.args = args
        kwargs = dict(feet=args.feet, terrain_level=args.level, sensor_stiffness=args.stiffness)
        if args.policy in POLICIES:
            kwargs["command"] = (0.5, 0.0, 0.0)
        self.model_policy = None
        if args.model:                           # trained policy: rebuild its env and normaliser
            from tactip_mujoco_gym.RL_BASE import load_policy
            model, env, normalise, _ = load_policy("TacQuad-walk", args.model, env_kwargs={
                "terrain_level": args.level})
            self.model_policy = lambda obs: model.predict(normalise(obs), deterministic=True)[0]
            self.env = env.unwrapped
            self.policy = "model"
        else:
            self.env = TacQuadruped(obs_modes={}, **kwargs)
            self.policy = args.policy
        self.rng = np.random.default_rng(0)
        self.flags = {"reset": False}
        self.reset()

    def reset(self):
        feel = FEELS.get(self.args.feel) if self.args.feel else None
        self.obs, _ = self.env.reset(options={"feel": feel} if feel else None)
        self.k = 0
        print(f"\n[reset] terrain level {self.env.level:.2f}, policy {self.policy}, "
              f"feet {self.env.feet}" + (f", every tile {self.args.feel}" if feel else ""))

    def action(self):
        if self.policy == "model":
            return self.model_policy(self.obs)
        if self.policy == "trot":
            return trot(self.k * self.env.dt)
        if self.policy == "random":
            return self.rng.uniform(-1, 1, 12)
        return np.zeros(12)

    def step(self):
        self.obs, _, term, trunc, info = self.env.step(self.action())
        self.k += 1
        if self.k % 25 == 0:
            self.status()
        if term or trunc:
            why = info["fail_reason"] or ("time up" if self.k >= self.env.max_steps else "left the arena")
            print(f"\n[episode over after {self.k * self.env.dt:.1f} s: {why}]")
            self.reset()

    def status(self):
        env = self.env
        _, lin, _, _ = env._base_state()
        feet = env._foot_pos()
        tiles = env._tile_index(feet[:, 0], feet[:, 1])
        line = (f"\rt {self.k * env.dt:5.1f}s  cmd {np.round(env._command, 2)}  "
                f"speed {lin[0]:+.2f} m/s  foot N {np.round(env._foot_force).astype(int)}  "
                f"ground mu {env._tile_friction[tiles].mean():.2f} soft {env._tile_softness[tiles].mean():.2f}")
        if env.n_markers:
            shift = np.linalg.norm(env.get_foot_nodes(), axis=2).max(axis=1) * 1e3
            line += f"  TacTip mm {np.round(shift, 1)}"
        print(line + "   ", end="", flush=True)

    def key(self, keycode):
        c = chr(keycode) if 32 <= keycode < 127 else ""
        if c == "R":
            self.flags["reset"] = True
        elif c == "P" and self.policy != "model":
            self.policy = POLICIES[(POLICIES.index(self.policy) + 1) % len(POLICIES)]
            print(f"\n[policy: {self.policy}]")
        elif c == "L":
            i = int(np.argmin([abs(lv - self.env.level) for lv in LEVELS]))
            self.env.set_terrain_level(LEVELS[(i + 1) % len(LEVELS)])
            self.flags["reset"] = True
            print(f"\n[terrain level -> {self.env.level:.1f} from the next reset]")

    # ------------------------------------------------------------------ outputs
    def view(self):
        import mujoco.viewer          # here, so --video also works on machines without a display
        env = self.env
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=self.key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = env._base
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.8, 135, -20
            while viewer.is_running():
                t0 = time.time()
                with viewer.lock():
                    if self.flags["reset"]:
                        self.flags["reset"] = False
                        self.reset()
                    self.step()
                env.sync_renderer(viewer)            # new terrain after a reset
                viewer.sync()
                time.sleep(max(0.0, env.dt / self.args.speed - (time.time() - t0)))

    def record(self, path, seconds):
        env = self.env
        n = int(seconds / env.dt)
        frames = []
        for _ in range(n):
            self.step()
            frames.append(env.render())
        fps = int(round(1 / env.dt))
        try:
            import cv2
            h, w = frames[0].shape[:2]
            out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
            for f in frames:
                out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            out.release()
        except ImportError:
            import imageio
            imageio.mimsave(path, frames, fps=fps)
        print(f"\nsaved {path} ({len(frames)} frames, {fps} fps)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="trot", choices=POLICIES)
    ap.add_argument("--model", default=None, help="name of a model saved by quadruped_walk.py")
    ap.add_argument("--feet", default="tactip", choices=["tactip", "proxy", "rigid"])
    ap.add_argument("--stiffness", type=float, default=10.0, help="sensor_stiffness")
    ap.add_argument("--level", type=float, default=0.5, help="terrain level 0..1")
    ap.add_argument("--feel", default=None, choices=list(FEELS), help="same feel on every tile")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--video", default=None, help="record to this .mp4 instead of opening a window")
    ap.add_argument("--seconds", type=float, default=10.0, help="video length")
    args = ap.parse_args()

    demo = Demo(args)
    if args.video:
        demo.record(args.video, args.seconds)
    else:
        demo.view()


if __name__ == "__main__":
    main()