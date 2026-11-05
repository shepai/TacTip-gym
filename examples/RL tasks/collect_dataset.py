"""
Collect paired vision / tactile / privileged datasets from the TacArm environments.

    python collect_dataset.py --env TacArm-peg --frames 50000 --workers 4
    python collect_dataset.py --env all --frames 50000 --workers 4
    python collect_dataset.py --env TacArm-peg --teacher TacArm-peg_v3_privileged   # trained SB3 teacher

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
import json
import multiprocessing as mp
import os
import time

import numpy as np

ALL_INPUTS = {"vision": True, "tactile": True, "privileged": True}
TASKS = ["TacArm-peg", "TacArm-hill", "TacArm-shape", "TacArm-edge"]
NOISE_MAX = 0.6
BEHAVIOUR = (("teacher_noisy", 0.7), ("random", 0.2), ("teacher", 0.1))


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


def _save_shard(buf, path):
    arrays = {}
    for k, v in buf.items():
        a = np.asarray(v)
        if a.dtype == np.float64:
            a = a.astype(np.float32)
        arrays[k] = a
    np.savez_compressed(path, **arrays)


def worker(env_name, n_frames, seed, out_dir, worker_id, shard_size, env_kwargs, teacher, model_dir):
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
        if terminated or truncated:
            ep += 1
            t = 0
            obs, mode, noise = new_episode()

        if len(buf["t"]) >= shard_size or frames >= n_frames:
            path = os.path.join(out_dir, f"shard_w{worker_id:02d}_{shard:03d}.npz")
            _save_shard(buf, path)
            print(f"[{env_name} w{worker_id}] {frames}/{n_frames} frames, {ep} episodes, "
                  f"{frames / (time.time() - t0):.0f} frames/s -> {os.path.basename(path)}", flush=True)
            buf, shard = {}, shard + 1
    env.close()


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
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="datasets")
    ap.add_argument("--shard-size", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env-kwargs", default="{}", help='JSON, e.g. \'{"spawn_yaw": 0.5}\'')
    ap.add_argument("--teacher", default=None,
                    help="name of a privileged-only model saved by RL_BASE.run (default: scripted)")
    ap.add_argument("--model-dir", default="models")
    args = ap.parse_args()

    env_kwargs = json.loads(args.env_kwargs)
    names = TASKS if args.env == "all" else [args.env]
    ctx = mp.get_context("spawn")          # fresh processes: safe with OpenGL renderers
    for env_name in names:
        out_dir = os.path.join(args.out, env_name)
        os.makedirs(out_dir, exist_ok=True)
        old = [f for f in os.listdir(out_dir) if f.endswith(".npz")]
        if old:
            raise SystemExit(f"{out_dir} already has {len(old)} shards; delete them or use --out")
        per = -(-args.frames // args.workers)
        procs = [ctx.Process(target=worker, args=(env_name, per, args.seed + 1000 * w, out_dir, w,
                                                   args.shard_size, env_kwargs, args.teacher,
                                                   args.model_dir))
                 for w in range(args.workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if any(p.exitcode != 0 for p in procs):
            raise SystemExit(f"a worker failed while collecting {env_name}")
        write_meta(env_name, out_dir, args, env_kwargs)


if __name__ == "__main__":
    main()
