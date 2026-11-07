"""
Watch the wheeled humanoid with TacTip hands (TacHumanoid-pick) in the MuJoCo viewer.

    python demo_humanoid.py                           # scripted pick-and-place, random objects
    python demo_humanoid.py --object soft_ball        # one object (see --help for the names)
    python demo_humanoid.py --mass 0.9 --friction 0.3 # heavy and slippery: watch it squeeze harder
    python demo_humanoid.py --stiffness 200           # very soft (N/m per pad; 4000 = firm)
    python demo_humanoid.py --level 1                 # hardest mix: any angle, target off to the side
    python demo_humanoid.py --policy random           # or: hold
    python demo_humanoid.py --model humanoid_object+tactile_demo   # a policy from humanoid_pick.py
    python demo_humanoid.py --video demo.mp4 --seconds 12          # no window: record a video

Keys in the viewer:  R reset   P next policy (scripted / hold / random)   O next object
                     L next level
The console shows the object and its properties, the scripted phase, the grip force on each
hand, how squashed the object is and how far each TacTip's markers have moved. On macOS run it
with `mjpython` instead of `python`.
"""
import argparse
import time

import numpy as np

from tactip_mujoco_gym.envs.tactip_humanoid import (
    OBJECT_NAMES,
    ScriptedPicker,
    TacHumanoidPick,
)

POLICIES = ("scripted", "hold", "random")
LEVELS = (0.0, 0.5, 1.0)


class Demo:
    def __init__(self, args):
        self.args = args
        self.model_policy = None
        if args.model:                           # trained policy: rebuild its env and normaliser
            from tactip_mujoco_gym.RL_BASE import load_policy
            model, env, normalise, _ = load_policy("TacHumanoid-pick", args.model,
                                                   env_kwargs={"level": args.level})
            self.model_policy = lambda obs: model.predict(normalise(obs), deterministic=True)[0]
            self.env = env.unwrapped
            self.policy = "model"
        else:
            self.env = TacHumanoidPick(obs_modes={}, level=args.level,
                                       sensor_stiffness=args.sensor_stiffness)
            self.policy = args.policy
        self.object = args.object
        self.rng = np.random.default_rng(0)
        self.flags = {"reset": False}
        self.reset()

    def reset(self):
        opts = {k: getattr(self.args, k) for k in ("mass", "friction", "stiffness")
                if getattr(self.args, k) is not None}
        if self.object:
            opts["object"] = self.object
        self.obs, _ = self.env.reset(options=opts)
        self.picker = ScriptedPicker(self.env)
        self.k = 0
        e = self.env
        p = e._props
        soft = np.isfinite(p["stiffness"])
        print(f"\n[reset] {e.obj}: mass {p['mass']:.2f} kg, friction {p['friction']:.2f}, "
              + (f"pads {p['stiffness']:.0f} N/m" if soft else "rigid")
              + f"; level {e.level:.2f}, policy {self.policy}")

    def action(self):
        if self.policy == "model":
            return self.model_policy(self.obs)
        if self.policy == "scripted":
            return self.picker()
        if self.policy == "random":
            return self.rng.uniform(-1, 1, self.env.action_space.shape)
        return np.zeros(self.env.action_space.shape)

    def step(self):
        self.obs, _, term, trunc, info = self.env.step(self.action())
        self.k += 1
        if self.k % 10 == 0:
            self.status()
        if term or trunc:
            why = "placed it!" if info["is_success"] else (info["fail_reason"] or "time up")
            print(f"\n[episode over after {self.k * self.env.dt:.1f} s: {why}]")
            self.reset()

    def status(self):
        e = self.env
        _, _, h, _, squash, _ = e._obj
        shift = np.linalg.norm(e.get_hand_nodes(), axis=2).max(axis=1) * 1e3
        phase = f"{self.picker.phase:<8}" if self.policy == "scripted" else ""
        print(f"\rt {self.k * e.dt:5.1f}s {phase} grip N {np.round(e._hand_force, 1)}  "
              f"squash {squash:.2f}  height {h * 100:4.1f} cm  TacTip mm {np.round(shift, 1)}   ",
              end="", flush=True)

    def key(self, keycode):
        c = chr(keycode) if 32 <= keycode < 127 else ""
        if c == "R":
            self.flags["reset"] = True
        elif c == "P" and self.policy != "model":
            self.policy = POLICIES[(POLICIES.index(self.policy) + 1) % len(POLICIES)]
            self.flags["reset"] = True
            print(f"\n[policy: {self.policy}]")
        elif c == "O":
            i = OBJECT_NAMES.index(self.object) if self.object else -1
            self.object = OBJECT_NAMES[(i + 1) % len(OBJECT_NAMES)]
            self.flags["reset"] = True
        elif c == "L":
            i = int(np.argmin([abs(lv - self.env.level) for lv in LEVELS]))
            self.env.level = LEVELS[(i + 1) % len(LEVELS)]
            self.flags["reset"] = True
            print(f"\n[level -> {self.env.level:.1f}]")

    # ------------------------------------------------------------------ outputs
    def view(self):
        import mujoco
        import mujoco.viewer  # here, so --video also works on machines without a display
        env = self.env
        with mujoco.viewer.launch_passive(env.model, env.data, key_callback=self.key) as viewer:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = env.model.body(env.TRACK_BODY).id
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.3, 150, -35
            while viewer.is_running():
                t0 = time.time()
                with viewer.lock():
                    if self.flags["reset"]:
                        self.flags["reset"] = False
                        self.reset()
                    self.step()
                viewer.sync()
                time.sleep(max(0.0, env.dt / self.args.speed - (time.time() - t0)))

    def record(self, path, seconds):
        env = self.env
        frames = []
        for _ in range(int(seconds / env.dt)):
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
    ap.add_argument("--policy", default="scripted", choices=POLICIES)
    ap.add_argument("--model", default=None, help="name of a model saved by humanoid_pick.py")
    ap.add_argument("--object", default=None, choices=OBJECT_NAMES)
    ap.add_argument("--mass", type=float, default=None, help="kg")
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--stiffness", type=float, default=None, help="soft objects: N/m per pad")
    ap.add_argument("--level", type=float, default=0.0, help="difficulty 0..1 (fixed)")
    ap.add_argument("--sensor-stiffness", type=float, default=10.0, help="TacTip skin")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed (1 = real time)")
    ap.add_argument("--video", default=None, help="record to this .mp4 instead of opening a window")
    ap.add_argument("--seconds", type=float, default=12.0, help="video length")
    args = ap.parse_args()

    demo = Demo(args)
    if args.video:
        demo.record(args.video, args.seconds)
    else:
        demo.view()


if __name__ == "__main__":
    main()
