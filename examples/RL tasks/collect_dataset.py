"""
Collect paired vision / tactile / privileged datasets from the TacArm environments.

    python collect_dataset.py --env TacArm-peg --frames 50000 --workers 4
    python collect_dataset.py --env all --frames 50000 --workers 4
    python collect_dataset.py --env TacArm-peg --teacher TacArm-peg_v3_privileged   # trained SB3 teacher
    python collect_dataset.py --env TacArm-peg --overwrite      # replace an earlier collection

Run `python check_envs.py <env>` on your model first: if the scripted teacher can't solve the
task, or episodes end after a few steps, the data will be mostly reset states.
While it runs it prints overall progress every 30 s and each worker's frame count. A worker
that makes no progress for 5 minutes prints its stack trace (where it is stuck); Ctrl-C stops
all workers and keeps the shards saved so far.

Writes datasets/<env>/shard_w<worker>_<n>.npz plus datasets/<env>/meta.json.
Every frame holds the state BEFORE the action, and what happened next:
    proprio, privileged, tactile, vision    observation parts (as with obs_format="dict")
    teacher_action   what the teacher does in this state (labels for behaviour cloning / DAgger)
    action           what was actually executed (teacher + noise, or random)
    reward, terminated, truncated, success
    touching, contact_force                 sensor-object contact (extra labels)
    shape_id         TacArm-shape only: index into ShapeInsertion.SHAPE_NAMES (-1 otherwise)
    episode, t       episode id (unique within the dataset) and step index

Behaviour, drawn per episode so the data covers good and bad states:
    70%  teacher + Gaussian noise (std drawn from 0..NOISE_MAX)
    20%  random actions
    10%  teacher, no noise
The default teacher is the scripted privileged controller from check_envs.py. A trained
privileged-only SB3 model (from RL_BASE.run) can be used instead with --teacher.
"""
import argparse
import faulthandler
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

ALL_INPUTS = {"vision": True, "tactile": True, "privileged": True}
TASKS = ["TacArm-peg", "TacArm-hill", "TacArm-shape", "TacArm-edge"]
NOISE_MAX = 0.6
BEHAVIOUR = (("teacher_noisy", 0.7), ("random", 0.2), ("teacher", 0.1))
# a worker with no progress for STALL_SECONDS prints where it is stuck;
# the main process prints overall progress every REPORT_SECONDS
STALL_SECONDS = int(os.environ.get("TACTIP_STALL_SECONDS", 300))
REPORT_SECONDS = int(os.environ.get("TACTIP_REPORT_SECONDS", 30))
# per-worker counters shared with the main process
FRAMES, EPISODES, SUCCESSES, FAILS, EP_STEPS = range(5)
N_COUNTERS = 5


def contact_body(raw):
    """Body the sensor is meant to touch in each task."""
    for attr in ("_cyl_body", "_ball_body", "_block_body"):
        if hasattr(raw, attr):
            return getattr(raw, attr)
    return raw._body[raw.shape]


def scripted_teacher(env_name):
    from check_envs import edge_policy, incline_policy, peg_policy, shape_policy
    policy = {"TacArm-peg": peg_policy, "TacArm-shape": shape_policy,
              "TacArm-hill": incline_policy, "TacArm-edge": edge_policy}[env_name]
    return lambda raw, obs: np.asarray(policy(raw), dtype=np.float32)


def sb3_teacher(env_name, model_name, model_dir):
    """A policy trained by RL_BASE.run on privileged-only, flat observations."""
    from RL_BASE import load_policy
    model, env, normalise, _ = load_policy(env_name, model_name, model_dir)
    raw = env.unwrapped
    only_privileged = raw.obs_modes == {"vision": False, "tactile": False, "privileged": True}
    if not only_privileged or raw.obs_format != "flat" or raw.events:
        raise ValueError(f"{model_name} must be trained on privileged-only flat observations "
                         f"(got {raw.obs_modes}, format {raw.obs_format}, events {raw.events})")
    env.close()

    def act(raw, obs):
        flat = np.concatenate([obs["proprio"], obs["privileged"]]).astype(np.float32)
        return np.asarray(model.predict(normalise(flat), deterministic=True)[0], dtype=np.float32)
    return act


GL_PROBE = """
import mujoco
m = mujoco.MjModel.from_xml_string("<mujoco/>")
r = mujoco.Renderer(m, 8, 8)
r.render()
try:
    from OpenGL import GL
    print(GL.glGetString(GL.GL_RENDERER).decode())
except Exception:
    print("unknown GPU")
r.close()
"""
SOFTWARE_GL = ("llvmpipe", "softpipe", "swrast", "swiftshader", "software")


def probe_gl(backend):
    """Make a MuJoCo renderer with MUJOCO_GL=backend ("" = MuJoCo's default) in a fresh process.
    Returns (works, renderer name or error text)."""
    env = os.environ.copy()
    env.pop("MUJOCO_GL", None)
    if backend:
        env["MUJOCO_GL"] = backend
    try:
        r = subprocess.run([sys.executable, "-c", GL_PROBE], env=env, capture_output=True,
                           text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return False, "timed out creating an OpenGL context"
    if r.returncode == 0 and r.stdout.strip():
        return True, r.stdout.strip().splitlines()[-1]
    return False, (r.stderr.strip().splitlines() or ["unknown error"])[-1]


def choose_gl(requested, n_workers):
    """Pick the rendering backend for the workers and set MUJOCO_GL (inherited by them)."""
    if requested != "auto":
        candidates = [requested]
    elif os.environ.get("MUJOCO_GL"):
        candidates = [os.environ["MUJOCO_GL"]]          # respect an explicit choice
    else:
        # EGL renders offscreen without a window and is reliable from many processes
        candidates = (["egl", "", "osmesa"] if sys.platform.startswith("linux") else [""])
    errors = []
    for backend in candidates:
        ok, name = probe_gl(backend)
        if ok:
            break
        errors.append(f"  MUJOCO_GL={backend or '(default)'}: {name}")
    else:
        raise SystemExit("No working OpenGL backend for MuJoCo, which the vision input needs:\n"
                         + "\n".join(errors) + "\nOn Linux install EGL (e.g. apt install libegl1) "
                         "or run on a machine with a display.")
    if backend:
        os.environ["MUJOCO_GL"] = backend
    else:
        os.environ.pop("MUJOCO_GL", None)
    software = any(k in name.lower() for k in SOFTWARE_GL)
    if software:
        # Mesa's CPU renderer: one thread pool per worker otherwise
        os.environ.setdefault("LP_NUM_THREADS", str(max(1, (os.cpu_count() or 2) // n_workers)))
    print(f"rendering  : MUJOCO_GL={backend or '(default)'} on {name}"
          + ("  (software rendering on the CPU - slow; check your GPU drivers if you have one)"
             if software else ""), flush=True)


def _save_shard(buf, path):
    arrays = {}
    for k, v in buf.items():
        a = np.asarray(v)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        arrays[k] = a
    np.savez_compressed(path, **arrays)


def worker(env_name, n_frames, seed, out_dir, worker_id, shard_size, env_kwargs, teacher,
           model_dir, progress):
    # If this worker stops making progress, dump its stack so the hang can be located.
    # Re-armed after every frame, so it only fires on a real stall.
    faulthandler.enable()
    faulthandler.dump_traceback_later(STALL_SECONDS)

    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401  (registers the environments)

    env = gym.make(f"tactip_mujoco_gym:{env_name}", obs_modes=ALL_INPUTS, obs_format="dict",
                   **env_kwargs)
    raw = env.unwrapped
    act_teacher = sb3_teacher(env_name, teacher, model_dir) if teacher else scripted_teacher(env_name)
    rng = np.random.default_rng(seed)
    names, probs = zip(*BEHAVIOUR, strict=True)
    shape_names = getattr(raw, "SHAPE_NAMES", None)

    def new_episode(first=False):
        o, _ = env.reset(seed=seed if first else None)
        mode = names[rng.choice(len(names), p=probs)]
        return o, mode, rng.uniform(0.0, NOISE_MAX)

    obs, mode, noise = new_episode(first=True)
    buf, frames, shard, ep, t, t0 = {}, 0, 0, 0, 0, time.time()
    base = worker_id * N_COUNTERS
    outcomes = {}

    def outcome_text():
        return ", ".join(f"{k} {v}" for k, v in sorted(outcomes.items(), key=lambda kv: -kv[1]))
    while frames < n_frames:
        a_teacher = np.clip(act_teacher(raw, obs), -1.0, 1.0)
        if mode == "random":
            a = rng.uniform(-1.0, 1.0, size=env.action_space.shape)
        elif mode == "teacher_noisy":
            a = np.clip(a_teacher + noise * rng.standard_normal(a_teacher.shape), -1.0, 1.0)
        else:
            a = a_teacher
        body = contact_body(raw)
        row = {k: v for k, v in obs.items()}
        row.update(teacher_action=a_teacher, action=a.astype(np.float32),
                   touching=raw._touching(body), contact_force=raw._contact_force(body),
                   shape_id=shape_names.index(raw.shape) if shape_names else -1,
                   episode=worker_id * 1_000_000 + ep, t=t)

        obs, r, terminated, truncated, info = env.step(a)
        row.update(reward=r, terminated=terminated, truncated=truncated,
                   success=bool(info.get("is_success", False)))
        for k, v in row.items():
            buf.setdefault(k, []).append(v)
        frames += 1
        t += 1
        progress[base + FRAMES] = frames
        if terminated or truncated:
            ep += 1
            progress[base + EPISODES] = ep
            progress[base + EP_STEPS] += t
            if info.get("is_success", False):
                progress[base + SUCCESSES] += 1
                end = "success"
            elif terminated:
                progress[base + FAILS] += 1
                end = info.get("fail_reason") or "fail"
            else:
                end = "timeout"
            outcomes[end] = outcomes.get(end, 0) + 1
            if ep == 50:
                print(f"[{env_name} w{worker_id}] first 50 episodes: {outcome_text()}", flush=True)
            t = 0
            obs, mode, noise = new_episode()
        faulthandler.dump_traceback_later(STALL_SECONDS)     # still alive: push the alarm back

        if len(buf["t"]) >= shard_size or frames >= n_frames:
            path = os.path.join(out_dir, f"shard_w{worker_id:02d}_{shard:03d}.npz")
            _save_shard(buf, path)
            print(f"[{env_name} w{worker_id}] saved {os.path.basename(path)} "
                  f"({frames}/{n_frames} frames, {frames / (time.time() - t0):.1f} frames/s; "
                  f"episode ends: {outcome_text() or 'none yet'})", flush=True)
            buf, shard = {}, shard + 1
    faulthandler.cancel_dump_traceback_later()
    env.close()


def _summary(progress, n_workers):
    c = np.array(progress[:], dtype=np.int64).reshape(n_workers, N_COUNTERS)
    eps = int(c[:, EPISODES].sum())
    mean_len = c[:, EP_STEPS].sum() / eps if eps else float("nan")
    return c, eps, mean_len, int(c[:, SUCCESSES].sum()), int(c[:, FAILS].sum())


def monitor(env_name, procs, progress, n_frames_total, t0):
    """Wait for the workers, printing progress, flagging stalled workers and suspicious data."""
    n = len(procs)
    last_frames = np.zeros(n, dtype=np.int64)
    last_change = np.full(n, time.time())
    warned_short, warned_stall = False, set()
    next_report = time.time() + REPORT_SECONDS
    while any(p.is_alive() for p in procs):
        time.sleep(2)
        c, eps, mean_len, succ, fails = _summary(progress, n)
        now = time.time()
        moved = c[:, FRAMES] != last_frames
        last_change[moved] = now
        last_frames = c[:, FRAMES].copy()
        if now >= next_report:
            next_report = now + REPORT_SECONDS
            total = int(c[:, FRAMES].sum())
            ep_txt = (f"{eps} episodes (mean {mean_len:.1f} steps, {100 * succ / eps:.0f}% success, "
                      f"{100 * fails / eps:.0f}% fail)" if eps else "no finished episodes yet")
            print(f"[{env_name}] {total}/{n_frames_total} frames, {total / (now - t0):.1f} frames/s, "
                  f"{ep_txt} | per worker: {' '.join(str(int(f)) for f in c[:, FRAMES])}", flush=True)
        if not warned_short and eps >= 50 and mean_len < 10:
            warned_short = True
            print(f"[{env_name}] WARNING: episodes end after {mean_len:.1f} steps on average "
                  f"({100 * fails / eps:.0f}% fail). The data will be mostly reset states. "
                  f"Run `python check_envs.py {env_name}`: the sensor geometry constants "
                  "(TIP_R, PUSH_Z) probably don't match your TacTip.", flush=True)
        for w, p in enumerate(procs):
            if p.is_alive() and now - last_change[w] > STALL_SECONDS and w not in warned_stall:
                warned_stall.add(w)
                print(f"[{env_name}] WARNING: worker {w} has made no progress for "
                      f"{now - last_change[w]:.0f} s; its stack trace is printed above.", flush=True)


def write_meta(env_name, out_dir, args, env_kwargs):
    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401

    env = gym.make(f"tactip_mujoco_gym:{env_name}", obs_modes=ALL_INPUTS, obs_format="dict",
                   **env_kwargs)
    raw = env.unwrapped
    shards = sorted(f for f in os.listdir(out_dir) if f.endswith(".npz"))
    n, eps, succ, touch = 0, 0, 0, 0
    for f in shards:
        d = np.load(os.path.join(out_dir, f))
        n += len(d["t"])
        eps += int((d["terminated"] | d["truncated"]).sum())
        succ += int(d["success"].sum())
        touch += int(d["touching"].sum())
    meta = dict(
        env_name=env_name, env_kwargs=env_kwargs, frames=n, finished_episodes=eps,
        successes=succ, touching_fraction=touch / max(n, 1),
        teacher=args.teacher or "scripted (check_envs.py)", behaviour=dict(BEHAVIOUR),
        noise_max=NOISE_MAX, seed=args.seed,
        shapes={k: list(sp.shape) for k, sp in env.observation_space.spaces.items()},
        action_dim=int(env.action_space.shape[0]),
        n_markers=int(raw.n_markers),
        vision=dict(camera=raw.VISION_CAMERA, width=raw.VISION_WIDTH,
                    height=raw.VISION_HEIGHT, mode=raw.VISION_MODE),
        shape_names=list(getattr(raw, "SHAPE_NAMES", [])),
    )
    env.close()
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{env_name}: {n} frames, {eps} finished episodes, {succ} successes, "
          f"sensor touching in {100 * meta['touching_fraction']:.0f}% of frames -> {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--env", default="all", help="TacArm-peg / -hill / -shape / -edge, or 'all'")
    ap.add_argument("--frames", type=int, default=50_000, help="frames per environment")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--shard-size", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-kwargs", default="{}", help='JSON, e.g. \'{"spawn_yaw": 0.5}\'')
    ap.add_argument("--teacher", default=None,
                    help="name of a privileged-only model saved by RL_BASE.run (default: scripted)")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--gl", default="auto",
                    help="MuJoCo rendering backend: auto (EGL if it works), egl, glfw, osmesa")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete shards from an earlier collection of the same env first")
    args = ap.parse_args()

    # Set before the workers start (they inherit the environment): one maths thread per
    # worker, otherwise N workers x N BLAS threads fight over the CPU.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(var, "1")
    choose_gl(args.gl, args.workers)

    env_kwargs = json.loads(args.env_kwargs)
    names = TASKS if args.env == "all" else [args.env]
    ctx = mp.get_context("spawn")          # fresh processes: safe with OpenGL renderers
    for env_name in names:
        out_dir = os.path.join(args.out, env_name)
        os.makedirs(out_dir, exist_ok=True)
        old = [f for f in os.listdir(out_dir) if f.endswith(".npz") or f == "meta.json"]
        if old and args.overwrite:
            for f in old:
                os.remove(os.path.join(out_dir, f))
            print(f"[{env_name}] removed {len(old)} old files from {out_dir}", flush=True)
        elif old:
            raise SystemExit(f"{out_dir} already has data from an earlier run; "
                             "add --overwrite to replace it, or use --out")
        per = -(-args.frames // args.workers)
        progress = ctx.Array("q", args.workers * N_COUNTERS)
        procs = [ctx.Process(target=worker, args=(env_name, per, args.seed + 1000 * w, out_dir, w,
                                                   args.shard_size, env_kwargs, args.teacher,
                                                   args.model_dir, progress))
                 for w in range(args.workers)]
        print(f"[{env_name}] {args.workers} workers x {per} frames -> {out_dir}", flush=True)
        t0 = time.time()
        for p in procs:
            p.start()
        try:
            monitor(env_name, procs, progress, per * args.workers, t0)
        except KeyboardInterrupt:
            for p in procs:
                p.terminate()
                p.join()
            if any(f.endswith(".npz") for f in os.listdir(out_dir)):
                write_meta(env_name, out_dir, args, env_kwargs)
            raise SystemExit("interrupted: workers stopped (shards saved so far are kept)") from None
        for p in procs:
            p.join()
        failed = [w for w, p in enumerate(procs) if p.exitcode != 0]
        if failed:
            raise SystemExit(f"worker(s) {failed} failed while collecting {env_name}; "
                             "see their error messages above")
        write_meta(env_name, out_dir, args, env_kwargs)


if __name__ == "__main__":
    main()
