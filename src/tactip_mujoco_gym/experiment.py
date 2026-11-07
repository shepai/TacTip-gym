"""
Experiment configs: one YAML / JSON file says what is being tested - the task, the robot's
options, the sensor hardware, the readouts, which inputs the policy gets, how they are fused,
and how to train - so a comparison is a set of files (or variants in one file), not edits to
code.

    python -m tactip_mujoco_gym train configs/peg_sensing.yaml               # every variant
    python -m tactip_mujoco_gym train configs/peg_sensing.yaml --variant tactile
    python -m tactip_mujoco_gym train configs/peg_sensing.yaml --dry-run     # show, don't run
    python -m tactip_mujoco_gym evaluate configs/peg_sensing.yaml --variant tactile

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
                    checkpoint_every: 500000}
    evaluate       {episodes: 50, deterministic: true, seed: 1000}
    imports        modules to import first (e.g. the module that registers your readout)
    variants       {name: {<any keys above>}} merged over the base (dicts merge, the rest
                   replaces), one run each

In Python:

    from tactip_mujoco_gym.experiment import load_config, make_env, train
    cfg = load_config("configs/peg_sensing.yaml")
    env = make_env(cfg, variant="tactile")
"""
import copy
import importlib
import json
import os

ENV_KEYS = ("sensor_xml", "sensor_stiffness", "sensors", "overrides")
TOP_KEYS = {"name", "description", "env", "env_kwargs", *ENV_KEYS, "obs_modes", "obs_format",
            "policy", "train", "evaluate", "imports", "variants"}
TRAIN_KEYS = {"algo", "steps", "n_envs", "seed", "algo_kwargs", "record_video",
              "video_every_percent", "video_seconds", "checkpoint_every"}
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


def resolve(cfg, variant=None):
    """The single config for `variant` (or the base config if it has no variants)."""
    if variant is None and cfg.get("variants"):
        names = list(cfg["variants"]) if isinstance(cfg["variants"], dict) else None
        raise ValueError(f"the config has variants {names}: pick one with variant=...")
    return variants(cfg, variant)[0][1]


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


def run_name(cfg, variant=None):
    name = cfg.get("name") or f"{cfg['env']}_{'+'.join(inputs_of(cfg)) or 'proprio'}"
    return f"{name}_{variant}" if variant else name


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


def run_arguments(cfg, variant=None):
    """The RL_BASE.run(...) call for a resolved config, as (args, kwargs)."""
    if "env" not in cfg:
        raise ValueError("the config needs an 'env' (a task id)")
    name = run_name(cfg, variant)
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
    for k in ("record_video", "video_every_percent", "video_seconds", "checkpoint_every"):
        if k in train:
            run_kw[k] = train[k]
    return (cfg["env"], name, name, int(train.get("steps", 1_000_000))), run_kw


# ----------------------------------------------------------------------------- actions
def make_env(cfg, variant=None, **extra):
    """gym.make for a config (and variant); extra keyword arguments override."""
    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401
    cfg = resolve(load_config(cfg) if not isinstance(cfg, dict) or "_dir" not in cfg else cfg,
                  variant)
    _imports(cfg)
    kw = env_kwargs(cfg)
    kw.update(extra)
    env_id = cfg["env"] if ":" in cfg["env"] else f"tactip_mujoco_gym:{cfg['env']}"
    return gym.make(env_id, **kw)


def train(cfg, variant=None, dry_run=False, steps=None):
    """Train every selected variant (all of them by default) with RL_BASE.run."""
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "_dir" not in cfg else cfg
    out = []
    for vname, c in variants(cfg, variant):
        args, kw = run_arguments(c, vname)
        if steps is not None:
            args = (*args[:3], int(steps))
        out.append((args, kw))
        print(f"=== {args[1]}: {args[0]}, {args[3]:,} steps\n    run{_fmt(args, kw)}")
        if dry_run:
            continue
        _imports(c)
        from tactip_mujoco_gym.RL_BASE import run
        run(*args, **kw)
    return out


def evaluate(cfg, variant=None, episodes=None):
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "_dir" not in cfg else cfg
    from tactip_mujoco_gym.RL_BASE import evaluate as _evaluate
    results = {}
    for vname, c in variants(cfg, variant):
        _imports(c)
        ev = dict(c.get("evaluate") or {})
        name = run_name(c, vname)
        results[name] = _evaluate(c["env"], name, n_episodes=int(episodes or ev.get("episodes", 50)),
                                  deterministic=ev.get("deterministic", True),
                                  seed=ev.get("seed", 1000))
    return results


def visualise(cfg, variant=None):
    cfg = load_config(cfg) if not isinstance(cfg, dict) or "_dir" not in cfg else cfg
    from tactip_mujoco_gym.RL_BASE import visualise as _visualise
    c = resolve(cfg, variant)
    _imports(c)
    _visualise(c["env"], run_name(c, variant))


def _fmt(args, kw):
    shown = {k: v for k, v in kw.items() if v not in (None, {}, "flat") or k == "obs_modes"}
    return "(" + ", ".join([repr(a) for a in args] + [f"{k}={v!r}" for k, v in shown.items()]) + ")"
