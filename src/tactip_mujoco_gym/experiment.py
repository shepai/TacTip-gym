"""
Experiment configs: one YAML / JSON file says what is being tested - the task, the robot's
options, the sensor hardware, the readouts, which inputs the policy gets, how they are fused,
and how to train - so a comparison is a set of files (or variants in one file), not edits to
code.

    python -m tactip_mujoco_gym train configs/peg_sensing.yaml               # every variant
    python -m tactip_mujoco_gym train configs/peg_sensing.yaml --variant tactile
    python -m tactip_mujoco_gym train configs/peg_sensing.yaml --dry-run     # show, don't run
    python -m tactip_mujoco_gym evaluate configs/peg_sensing.yaml --variant tactile
    python -m tactip_mujoco_gym train configs/sensing_trials.yaml --skip-trained   # many tasks
    python -m tactip_mujoco_gym summary configs/sensing_trials.yaml               # table + chart

Keys (all optional except env):

    name           model / log name (default: <env>_<inputs>); variants append _<variant>
    description    free text
    env            task id (python -m tactip_mujoco_gym list)
    env_kwargs     task / robot options, e.g. {start_touching: false} or {feet: proxy}
    sensor_xml     sensor hardware: a registered name ("tactip"), a standalone MJCF file
    sensor_stiffness   skin stiffness multiplier
    sensors        readouts: {key: spec}; a spec is a registered name, "module:Class", or
                   {type: ..., <arguments>}; {width: 96} without a type tweaks the default;
                   null removes it
    obs_modes      inputs the policy gets: [tactile, vision] (proprioception always included)
    obs_format     flat | dict
    overrides      {CONSTANT: value} for the env's class constants
    policy         {encoders: null | scratch | <checkpoint>, freeze_encoders: true}
    train          {algo: PPO, steps: 3000000, n_envs: 8, seed: 0, algo_kwargs: {...},
                    record_video: true, video_every_percent: 5, video_seconds: 7,
                    checkpoint_every: 500000, log_every: 100000, seeds: N}
                   seeds: N independent training runs (seeds seed .. seed+N-1, names end
                   _s<seed>); algo_kwargs.policy_kwargs.activation_fn may be a torch.nn
                   name ("ELU")
    evaluate       {episodes: 50, deterministic: true, seed: 1000}: episode k of every
                   policy is reset(seed=seed + k), so policies are tested on the same episodes
    imports        modules to import first (e.g. the module that registers your readout)
    variants       {name: {<any keys above>}} merged over the base (dicts merge, the rest
                   replaces), one run each
    tasks          {task id: {<any keys above but variants>}} (or a list of ids): the same
                   variants on several tasks, each entry merged over the base (its own steps,
                   env_kwargs ...); runs are named <name>_<task>_<variant>. A variant whose
                   inputs a task doesn't offer is skipped for that task.

Evaluating saves every test episode to results/<name>/episodes.csv and summarises them
(success rate with a 95% CI per task and variant) in summary.csv, a table and success.png
(see results.py).

In Python:

    from tactip_mujoco_gym.experiment import load_config, make_env, train
    cfg = load_config("configs/peg_sensing.yaml")
    env = make_env(cfg, variant="tactile")
"""
import copy
import importlib
import json
import os
import traceback

ENV_KEYS = ("sensor_xml", "sensor_stiffness", "sensors", "overrides")
TOP_KEYS = {"name", "description", "env", "env_kwargs", *ENV_KEYS, "obs_modes", "obs_format",
            "policy", "train", "evaluate", "imports", "variants", "tasks"}
TRAIN_KEYS = {
    "algo", "steps", "n_envs", "seed", "algo_kwargs",
    "record_video", "video_every_percent", "video_seconds",
    "checkpoint_every", "seeds", "log_every"
}
POLICY_KEYS = {"encoders", "freeze_encoders"}
EVAL_KEYS = {"episodes", "deterministic", "seed"}


# ----------------------------------------------------------------------------- loading
def modes_dict(obs_modes):
    """obs_modes as {name: bool} (it may be given as a list or "tactile+vision")."""
    if obs_modes is None:
        return None
    if isinstance(obs_modes, str):
        obs_modes = [m for m in obs_modes.replace(",", "+").split("+") if m.strip()]
    if isinstance(obs_modes, (list, tuple, set)):
        return {str(m).strip(): True for m in obs_modes}
    return {k: bool(v) for k, v in dict(obs_modes).items()}


def load_config(source):
    """A config dict from a .yaml / .yml / .json file, or a dict (copied). Checked for
    unknown keys, so a typo fails here rather than being silently ignored."""
    if isinstance(source, dict):
        cfg = copy.deepcopy(source)
        base = os.getcwd()
    else:
        path = str(source)
        with open(path) as f:
            text = f.read()
        if path.endswith((".yaml", ".yml")):
            try:
                import yaml
            except ImportError:
                raise ImportError("YAML configs need PyYAML: pip install pyyaml "
                                  "(or write the config as .json)") from None
            cfg = yaml.safe_load(text) or {}
        else:
            cfg = json.loads(text)
        base = os.path.dirname(os.path.abspath(path))
        cfg.setdefault("_file", os.path.abspath(path))
    cfg.setdefault("_dir", base)
    if isinstance(cfg.get("tasks"), (list, tuple)):
        cfg["tasks"] = {t: {} for t in cfg["tasks"]}
    check_config(cfg)
    return cfg


def check_config(cfg, where="config"):
    unknown = set(cfg) - TOP_KEYS - {"_file", "_dir"}
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {sorted(unknown)}; allowed: {sorted(TOP_KEYS)}")
    for key, allowed in (("train", TRAIN_KEYS), ("policy", POLICY_KEYS), ("evaluate", EVAL_KEYS)):
        bad = set(cfg.get(key) or {}) - allowed
        if bad:
            raise ValueError(f"{where}: unknown {key} key(s) {sorted(bad)}; allowed: {sorted(allowed)}")
    if "variants" in cfg and where != "config":
        raise ValueError(f"{where}: variants can't be nested")
    if "tasks" in cfg and where != "config":
        raise ValueError(f"{where}: tasks go at the top level")
    tasks = cfg.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise ValueError(f"{where}: tasks is a mapping {{task id: {{options}}}} or a list of ids")
    for name, t in tasks.items():
        check_config(t or {}, where=f"task {name!r}")
    variants = cfg.get("variants") or {}
    if isinstance(variants, list):
        variants = {v.get("name") or f"v{i}": v for i, v in enumerate(variants)}
    for name, v in variants.items():
        check_config({k: x for k, x in v.items() if k != "name"}, where=f"variant {name!r}")


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def variants(cfg, only=None):
    """[(variant name or None, resolved config)]: one entry per variant (or just the base
    config if it has none); `only` picks some by name."""
    vs = cfg.get("variants") or {}
    if isinstance(vs, list):
        vs = {v.get("name") or f"v{i}": {k: x for k, x in v.items() if k != "name"}
              for i, v in enumerate(vs)}
    base = {k: v for k, v in cfg.items() if k != "variants"}
    if not vs:
        if only:
            raise ValueError(f"the config has no variants (asked for {only})")
        return [(None, base)]
    names = list(vs) if not only else ([only] if isinstance(only, str) else list(only))
    missing = [n for n in names if n not in vs]
    if missing:
        raise ValueError(f"no variant(s) {missing}; the config has {list(vs)}")
    return [(n, _merge(base, vs[n])) for n in names]


def task_configs(cfg, tasks=None):
    """[(task id or None, config for it)]: one per entry of `tasks` (each merged over the
    base, with env = its id unless it sets one), or just the base config without tasks."""
    tmap = cfg.get("tasks") or {}
    base = {k: v for k, v in cfg.items() if k != "tasks"}
    if not tmap:
        if tasks:
            raise ValueError(f"the config has no tasks (asked for {tasks})")
        return [(None, base)]
    names = list(tmap) if not tasks else ([tasks] if isinstance(tasks, str) else list(tasks))
    missing = [t for t in names if t not in tmap]
    if missing:
        raise ValueError(f"no task(s) {missing}; the config has {list(tmap)}")
    return [(t, _merge(base, {"env": t, **(tmap[t] or {})})) for t in names]


def _unavailable(cfg):
    """Inputs the config asks for that its task doesn't offer (per the task catalogue)."""
    try:
        from tactip_mujoco_gym.envs import TASKS
    except Exception:                                 # noqa: BLE001
        return []
    offered = (TASKS.get(cfg.get("env")) or {}).get("inputs")
    if not offered:
        return []
    return [i for i in inputs_of(cfg) if i not in offered]


def runs(cfg, variant=None, tasks=None):
    """Every training run a config describes (tasks x variants x seeds), as dicts
    {name, task, variant, seed, cfg}; and [(task, variant, missing inputs)] for the
    variants a task can't provide (left out)."""
    out, skipped = [], []
    for task, tc in task_configs(cfg, tasks):
        for vname, c in variants(tc, variant):
            if task is not None:
                missing = _unavailable(c)
                if missing:
                    skipped.append((task, vname, missing))
                    continue
            tr = c.get("train") or {}
            first = int(tr.get("seed", 0))
            seeds = [None] if tr.get("seeds") is None else range(first, first + int(tr["seeds"]))
            for s in seeds:
                cc = copy.deepcopy(c)
                if s is not None:
                    cc["train"] = dict(cc.get("train") or {}, seed=s)
                out.append(dict(name=run_name(cc, vname, task=task, seed=s),
                                task=task or cc.get("env"), variant=vname or "default",
                                seed=first if s is None else s, cfg=cc))
    return out, skipped


def resolve(cfg, variant=None, task=None):
    """The single config for `variant` (and `task`, if the config has tasks)."""
    tcs = task_configs(cfg, [task] if task else None)
    if len(tcs) > 1:
        raise ValueError(f"the config has tasks {[t for t, _ in tcs]}: pick one with task=...")
    base = tcs[0][1]
    if variant is None and base.get("variants"):
        names = list(base["variants"]) if isinstance(base["variants"], dict) else None
        raise ValueError(f"the config has variants {names}: pick one with variant=...")
    return variants(base, variant)[0][1]


# ----------------------------------------------------------------------------- pieces
def _imports(cfg):
    for mod in cfg.get("imports") or []:
        importlib.import_module(mod)


def _path(cfg, value):
    """Relative file paths in a config are relative to the config file."""
    if isinstance(value, str) and (value.endswith(".xml") or value.endswith(".pt")):
        p = value if os.path.isabs(value) else os.path.join(cfg.get("_dir", "."), value)
        if os.path.exists(p):
            return os.path.normpath(p)
    return value


def inputs_of(cfg):
    modes = modes_dict(cfg.get("obs_modes"))
    if modes is None:
        return ["privileged"]
    return [k for k, on in modes.items() if on]


def run_name(cfg, variant=None, task=None, seed=None):
    """<name>[_<task>][_<variant>][_s<seed>] (task with a tasks config, seed with train.seeds)."""
    if task is not None:
        name = f"{cfg.get('name') or 'experiment'}_{task}"
    else:
        name = cfg.get("name") or f"{cfg['env']}_{'+'.join(inputs_of(cfg)) or 'proprio'}"
    if variant:
        name += f"_{variant}"
    if seed is not None:
        name += f"_s{seed}"
    return name


def experiment_name(cfg):
    """Folder name for an experiment's results: results/<this>/."""
    return cfg.get("name") or cfg.get("env") or "experiment"


def env_kwargs(cfg):
    """Keyword arguments for gym.make: env_kwargs + sensor_xml / sensors / ... + obs_modes /
    obs_format."""
    kw = dict(cfg.get("env_kwargs") or {})
    for key in ENV_KEYS:
        if cfg.get(key) is not None:
            kw[key] = copy.deepcopy(cfg[key])
    if "sensor_xml" in kw:
        kw["sensor_xml"] = _path(cfg, kw["sensor_xml"])
    if cfg.get("obs_modes") is not None:
        kw["obs_modes"] = modes_dict(cfg["obs_modes"])
    if cfg.get("obs_format"):
        kw["obs_format"] = cfg["obs_format"]
    return kw


def run_arguments(cfg, variant=None, name=None):
    """The RL_BASE.run(...) call for a resolved config, as (args, kwargs)."""
    if "env" not in cfg:
        raise ValueError("the config needs an 'env' (a task id), or tasks")
    name = name or run_name(cfg, variant)
    train = dict(cfg.get("train") or {})
    policy = dict(cfg.get("policy") or {})
    kw = env_kwargs(cfg)
    obs_modes = kw.pop("obs_modes", None)
    obs_format = kw.pop("obs_format", "flat")
    run_kw = dict(obs_modes=obs_modes, obs_format=obs_format, env_kwargs=kw,
                  algo=train.get("algo", "PPO"), seed=train.get("seed", 0),
                  n_envs=train.get("n_envs"), algo_kwargs=train.get("algo_kwargs"),
                  encoders=_path(cfg, policy.get("encoders")),
                  freeze_encoders=policy.get("freeze_encoders", True))
    for k in (
        "record_video",
        "video_every_percent",
        "video_seconds",
        "checkpoint_every",
        "log_every",
    ):
        if k in train:
            run_kw[k] = train[k]
    return (cfg["env"], name, name, int(train.get("steps", 1_000_000))), run_kw


# ----------------------------------------------------------------------------- actions
def _load(cfg):
    return load_config(cfg) if not isinstance(cfg, dict) or "_dir" not in cfg else cfg


def _trained(name, model_dir="models"):
    return os.path.exists(os.path.join(model_dir, name + ".zip"))


def make_env(cfg, variant=None, task=None, **extra):
    """gym.make for a config (and variant / task); extra keyword arguments override."""
    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401
    cfg = resolve(_load(cfg), variant, task)
    _imports(cfg)
    kw = env_kwargs(cfg)
    kw.update(extra)
    env_id = cfg["env"] if ":" in cfg["env"] else f"tactip_mujoco_gym:{cfg['env']}"
    return gym.make(env_id, **kw)


def train(cfg, variant=None, dry_run=False, steps=None, tasks=None, skip_trained=False,
          steps_scale=None):
    """Train every selected run (all of them by default) with RL_BASE.run.
    steps replaces each run's train.steps, steps_scale multiplies them (pilots);
    skip_trained leaves runs whose model already exists (resuming a long experiment).
    With several runs, one that fails is logged (results/<name>/errors.log) and the rest
    carry on."""
    cfg = _load(cfg)
    planned, skipped = runs(cfg, variant, tasks)
    for task, v, missing in skipped:
        print(f"--- skipped {task} / {v}: the task has no {'/'.join(missing)} input")
    out, failed = [], []
    calls = []
    for r in planned:
        args, kw = run_arguments(r["cfg"], r["variant"], name=r["name"])
        if steps is not None:
            args = (*args[:3], int(steps))
        elif steps_scale:
            args = (*args[:3], max(1, int(round(args[3] * steps_scale))))
        calls.append((r, args, kw))
        out.append((args, kw))
    if len(calls) > 1:
        total = sum(a[3] for _, a, _ in calls)
        left = sum(a[3] for _, a, _ in calls if not _trained(a[1]))
        print(f"{len(calls)} runs, {total / 1e6:,.1f}M training steps ({left / 1e6:,.1f}M not trained yet)")
    for i, (r, args, kw) in enumerate(calls, 1):
        done = _trained(args[1])
        tag = f"[{i}/{len(calls)}] " if len(calls) > 1 else ""
        print(f"=== {tag}{args[1]}: {args[0]}, {args[3]:,} steps{'  (trained)' if done else ''}"
              f"\n    run{_fmt(args, kw)}")
        if dry_run:
            continue
        if skip_trained and done:
            print("    already trained: skipped")
            continue
        try:
            _imports(r["cfg"])
            from tactip_mujoco_gym.RL_BASE import run
            run(*args, **kw)
        except KeyboardInterrupt:
            raise
        except Exception as err:                      # noqa: BLE001
            if len(calls) == 1:
                raise
            failed.append(args[1])
            msg = traceback.format_exc()
            print(f"    FAILED: {type(err).__name__}: {err}")
            from tactip_mujoco_gym.results import results_dir
            with open(os.path.join(results_dir(experiment_name(cfg)), "errors.log"), "a") as f:
                f.write(f"=== {args[1]}\n{msg}\n")
    if failed:
        print(f"\n{len(failed)} run(s) failed (see results/{experiment_name(cfg)}/errors.log): "
              + ", ".join(failed))
    return out


def evaluate(cfg, variant=None, episodes=None, tasks=None, save=True):
    """Test every selected run's trained model (runs not trained yet are skipped) on
    evaluate.episodes episodes. save: write each episode to results/<name>/episodes.csv
    (replacing that run's earlier rows), then summary.csv, the table and success.png.
    Returns {run name: {success_rate, mean_return, mean_length}}."""
    cfg = _load(cfg)
    from tactip_mujoco_gym import results as rs
    from tactip_mujoco_gym.RL_BASE import evaluate as _evaluate
    planned, _ = runs(cfg, variant, tasks)
    out_dir = rs.results_dir(experiment_name(cfg)) if save else None
    results = {}
    for r in planned:
        c, name = r["cfg"], r["name"]
        if not _trained(name):
            print(f"{name}: not trained yet, skipped")
            continue
        _imports(c)
        ev = dict(c.get("evaluate") or {})
        res = _evaluate(c["env"], name, n_episodes=int(episodes or ev.get("episodes", 50)),
                        deterministic=ev.get("deterministic", True), seed=ev.get("seed", 1000),
                        per_episode=True)
        eps = res.pop("episodes")
        results[name] = res
        if save:
            rows = [{"run": name, "task": r["task"], "variant": r["variant"], "seed": r["seed"],
                     "trial": k, "success": int(e["success"]), "return": round(e["ret"], 4),
                     "length": e["length"], "fail_reason": e["fail_reason"]}
                    for k, e in enumerate(eps)]
            rs.replace_episodes(os.path.join(out_dir, "episodes.csv"), name, rows)
    if save and results:
        rs.write_summary(out_dir)
    return results


def summary(cfg):
    """The table, summary.csv and success.png again from results/<name>/episodes.csv."""
    from tactip_mujoco_gym import results as rs
    return rs.write_summary(rs.results_dir(experiment_name(_load(cfg))))


def visualise(cfg, variant=None, task=None, seed=None):
    cfg = _load(cfg)
    from tactip_mujoco_gym.RL_BASE import visualise as _visualise
    c = resolve(cfg, variant, task)
    tr = c.get("train") or {}
    if tr.get("seeds") is not None and seed is None:
        seed = int(tr.get("seed", 0))
    _imports(c)
    _visualise(c["env"], run_name(c, variant, task=task,
                                  seed=seed if tr.get("seeds") is not None else None))


def _fmt(args, kw):
    shown = {k: v for k, v in kw.items() if v not in (None, {}, "flat") or k == "obs_modes"}
    return "(" + ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in shown.items()]) + ")"
