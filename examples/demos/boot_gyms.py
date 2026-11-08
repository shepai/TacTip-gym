"""
Boot the gyms: build each task, reset it, run it for a while with a scripted policy (or random
actions), and print what it offers. With --view, watch one in the MuJoCo viewer; with
--video, record one.

    python examples/demos/boot_gyms.py                             # every task: what boots
    python examples/demos/boot_gyms.py TacArm-surface TacRig-texture TacGripper-pick
    python examples/demos/boot_gyms.py TacArm-surface --view       # watch (mjpython on macOS)
    python examples/demos/boot_gyms.py TacGripper-pick --video gripper.mp4 --seconds 15
    python examples/demos/boot_gyms.py TacArm-surface --view --kwargs '{"wave": "chirp", "texture": "patches"}'
    python examples/demos/boot_gyms.py TacRig-texture --obs tactile+force --policy random

Scripted policies (they read the simulator's state, so they work with any --obs):
    arm tasks          check_envs.py's controllers (TacArm-surface: depth + pitch follower)
    humanoid, gripper  ScriptedPicker: square up, squeeze, lift, carry, put down
    rig                press to 1.9 N, slide 5 mm, lift; TacRig-texture then answers (with the
                       true texture: it shows the mechanics, it isn't a classifier)
    quadruped          stand still (zero action)
Tasks whose robot files aren't in assets/ (Z1 meshes, Go2, RB-Y1) are reported as missing.
"""
import argparse
import json
import time
import traceback

import gymnasium as gym
import numpy as np

import tactip_mujoco_gym
from tactip_mujoco_gym import TASKS


# ----------------------------------------------------------------------------- policies
class RigScript:
    """Press until the plate reads FORCE, slide along x keeping that force, lift, answer."""
    FORCE = 1.9

    def __init__(self, env):
        self.raw = env.unwrapped
        self.phase, self.t = "down", 0

    def __call__(self):
        r = self.raw
        if r.step_count == 0:                         # a new episode
            self.phase, self.t = "down", 0
        a = np.zeros(r.action_space.shape)
        if r._decide_index is not None:
            a[r._decide_index] = -1.0                  # not yet
        self.t += 1
        f = r.plate_force
        if self.phase == "down":
            a[2] = -1.0 if f < 0.05 else -0.25
            if f >= self.FORCE:
                self.phase, self.t = "slide", 0
        elif self.phase == "slide":                    # 0.1 mm per step, force held by z
            a[0] = 0.25
            a[2] = float(np.clip(0.5 * (f - self.FORCE), -0.5, 0.5))
            if self.t > 50:
                self.phase, self.t = "lift", 0
        elif self.phase == "lift":
            a[2] = 1.0
            if self.t > 20:
                self.phase = "answer"
        elif r._decide_index is not None:
            a[r._decide_index] = 1.0
            a[r._decide_index + 1 + r.texture_index] = 1.0
        return a

def gif(task, obs, kwargs, policy, path, seconds):
    import imageio.v2 as imageio

    env = make(task, obs, kwargs)
    raw = env.unwrapped
    env.reset(seed=0)

    act = make_policy(task, env, policy)
    dt = env_dt(raw)
    frames = []

    for _ in range(int(seconds / dt)):
        _, _, term, trunc, _ = env.step(act())
        frames.append(raw.render())

        if term or trunc:
            env.reset()

    env.close()

    imageio.mimsave(
        path,
        frames,
        duration=dt,
        loop=0
    )

    print(f"saved {len(frames)} frames to {path}")
def make_policy(task, env, kind):
    """A function of no arguments returning the next action."""
    raw = env.unwrapped
    if kind == "random":
        return env.action_space.sample
    if kind == "hold" or task.startswith("TacQuad") or task in ("RobotArm-v0", "RandomTac-v0"):
        return lambda: np.zeros(env.action_space.shape)
    if task.startswith("TacRig"):
        return RigScript(env)
    if task in ("TacHumanoid-pick", "TacGripper-pick"):
        from tactip_mujoco_gym.envs.tactip_humanoid import ScriptedPicker
        state = {}

        def picker():
            if raw.step_count == 0 or "p" not in state:     # one picker per episode
                state["p"] = ScriptedPicker(env)
            return state["p"]()
        return picker
    from tactip_mujoco_gym.check_envs import TASKS as ARM_SCRIPTS
    if task in ARM_SCRIPTS:
        script = ARM_SCRIPTS[task][0]
        return lambda: script(raw)
    return env.action_space.sample


def env_dt(raw):
    dt = getattr(raw, "dt", None)
    if dt:
        return float(dt)
    return float(getattr(raw, "FRAME_SKIP", 1) * raw.model.opt.timestep)


def make(task, obs, kwargs):
    kw = dict(kwargs)
    if obs:
        kw["obs_modes"] = obs
    return gym.make(task if ":" in task else f"tactip_mujoco_gym:{task}", **kw)


# ----------------------------------------------------------------------------- modes
def boot(tasks, obs, kwargs, policy, steps, verbose):
    """Build, reset and run each task; one line per task."""
    print(f"{'task':<18} {'build':>6} {'obs':>14} {'act':>4} {'steps/s':>8} {'return':>8}  episode")
    for task in tasks:
        t0 = time.time()
        try:
            env = make(task, obs, kwargs)
        except Exception as err:                                  # noqa: BLE001
            print(f"{task:<18} not built: {type(err).__name__}: {str(err).splitlines()[0][:110]}")
            if verbose:
                traceback.print_exc()
            continue
        build = time.time() - t0
        try:
            raw = env.unwrapped
            obs0, info = env.reset(seed=0)
            act = make_policy(task, env, policy)
            shape = ({k: v.shape for k, v in obs0.items()} if isinstance(obs0, dict)
                     else np.shape(obs0))
            R, outcome, n, t1 = 0.0, "running", 0, time.time()
            for n in range(1, steps + 1):
                _, r, term, trunc, info = env.step(act())
                R += float(r)
                if term or trunc:
                    outcome = ("success" if info.get("is_success") else
                               info.get("fail_reason") or ("time limit" if trunc else "ended"))
                    break
            rate = n / max(time.time() - t1, 1e-9)
            shape_s = str(shape) if not isinstance(shape, dict) else f"dict({len(shape)})"
            print(f"{task:<18} {build:5.1f}s {shape_s:>14} {env.action_space.shape[0] if env.action_space.shape else 0:>4} "
                  f"{rate:8.0f} {R:8.1f}  {outcome} after {n} steps")
            if verbose:
                avail = {k: ok for k, (ok, _) in raw.describe_inputs().items()} if hasattr(raw, "describe_inputs") else {}
                print(f"{'':<18} inputs: " + ", ".join(f"{k}{'' if ok else ' (unavailable)'}" for k, ok in avail.items()))
                if isinstance(shape, dict):
                    print(f"{'':<18} observation: {shape}")
        except Exception as err:                                  # noqa: BLE001
            print(f"{task:<18} built, then failed: {type(err).__name__}: {err}")
            if verbose:
                traceback.print_exc()
        finally:
            env.close()


def view(task, obs, kwargs, policy):
    """Watch the task in MuJoCo's passive viewer; episodes restart when they end."""
    import mujoco.viewer
    env = make(task, obs, kwargs)
    raw = env.unwrapped
    env.reset(seed=0)
    act = make_policy(task, env, policy)
    dt = env_dt(raw)
    sync = getattr(raw, "sync_renderer", None)               # heightfields redrawn per episode
    episode, R = 0, 0.0
    with mujoco.viewer.launch_passive(raw.model, raw.data) as viewer:
        while viewer.is_running():
            t = time.time()
            _, r, term, trunc, info = env.step(act())
            R += float(r)
            if term or trunc:
                episode += 1
                why = "success" if info.get("is_success") else (info.get("fail_reason") or "time limit")
                print(f"episode {episode}: return {R:.1f}, {why}")
                R = 0.0
                env.reset()
            if sync is not None:
                sync(viewer)
            viewer.sync()
            time.sleep(max(0.0, dt - (time.time() - t)))
    env.close()


def video(task, obs, kwargs, policy, path, seconds):
    from tactip_mujoco_gym.RL_BASE import write_video
    env = make(task, obs, kwargs)
    raw = env.unwrapped
    env.reset(seed=0)
    act = make_policy(task, env, policy)
    dt = env_dt(raw)
    frames = []
    for _ in range(int(seconds / dt)):
        _, _, term, trunc, _ = env.step(act())
        frames.append(raw.render())
        if term or trunc:
            env.reset()
    env.close()
    write_video(frames, path, fps=max(1, round(1 / dt)))
    print(f"saved {len(frames)} frames to {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tasks", nargs="*", help=f"default: all of {list(TASKS)}")
    ap.add_argument("--policy", default="scripted", choices=("scripted", "random", "hold"))
    ap.add_argument("--obs", default=None, help='inputs, e.g. "tactile" or "tactile+force" (default: the task\'s)')
    ap.add_argument("--kwargs", default="{}", help="task options as JSON")
    ap.add_argument("--steps", type=int, default=150, help="steps per task when booting")
    ap.add_argument("--view", action="store_true", help="watch the (first) task in the viewer")
    ap.add_argument("--video", default=None, help="record the (first) task to this .mp4")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--gif", default=None, help="record the (first) task to this .gif")
    ap.add_argument("-v", "--verbose", action="store_true", help="inputs, shapes, tracebacks")
    a = ap.parse_args()
    tasks = a.tasks or list(TASKS)
    kwargs = json.loads(a.kwargs)
    if a.view:
        view(tasks[0], a.obs, kwargs, a.policy)
    elif a.video:
        video(tasks[0], a.obs, kwargs, a.policy, a.video, a.seconds)
    elif a.gif:
        gif(tasks[0], a.obs, kwargs, a.policy, a.gif, a.seconds)
    else:
        boot(tasks, a.obs, kwargs, a.policy, a.steps, a.verbose)


if __name__ == "__main__":
    main()
