"""
Command line:

    python -m tactip_mujoco_gym list                         tasks, readouts, sensor models
    python -m tactip_mujoco_gym describe TacArm-peg          build a task, list its inputs
    python -m tactip_mujoco_gym describe TacArm-peg --sensor-xml my_sensor.xml \\
                                         --sensors '{"force": "mypkg.sensors:Force"}'
    python -m tactip_mujoco_gym describe configs/peg_sensing.yaml --variant tactile
    python -m tactip_mujoco_gym train configs/peg_sensing.yaml [--variant NAME ...] [--dry-run]
    python -m tactip_mujoco_gym evaluate configs/peg_sensing.yaml [--variant NAME] [--episodes N]
    python -m tactip_mujoco_gym summary configs/peg_sensing.yaml      results table + chart again
    python -m tactip_mujoco_gym visualise configs/peg_sensing.yaml --variant NAME

    A config with several tasks (configs/sensing_trials.yaml):
    python -m tactip_mujoco_gym train configs/sensing_trials.yaml --dry-run          the plan
    python -m tactip_mujoco_gym train configs/sensing_trials.yaml --skip-trained     resumes
    python -m tactip_mujoco_gym train configs/sensing_trials.yaml --tasks TacArm-peg --steps-scale 0.1
    python -m tactip_mujoco_gym visualise configs/sensing_trials.yaml --task TacArm-peg --variant tactip

(also installed as the `tactip-gym` command)
"""
import argparse
import json
import sys


def _list(_args):
    from tactip_mujoco_gym import TASKS, list_sensor_models, list_sensors
    print("Tasks (gym.make id):")
    for k, t in TASKS.items():
        print(f"  {k:<18} {t['summary']}")
        print(f"  {'':<18} robot: {t['robot']}")
        print(f"  {'':<18} inputs: {', '.join(t['inputs']) or '-'}; actions: {t['actions']}")
        for opt, doc in t["options"].items():
            print(f"  {'':<18}   {opt}: {doc}")
    print("\nSensor readouts (sensors={key: name}):")
    for k, doc in list_sensors().items():
        print(f"  {k:<18} {doc}")
    print("\nSensor hardware (sensor_xml=name, or a path to a standalone MJCF file):")
    for k, doc in list_sensor_models().items():
        print(f"  {k:<18} {doc}")


def _describe(args):
    import gymnasium as gym
    import tactip_mujoco_gym  # noqa: F401
    if args.target.endswith((".yaml", ".yml", ".json")):
        from tactip_mujoco_gym.experiment import make_env
        env = make_env(args.target, variant=args.variant, task=args.task)
    else:
        kw = json.loads(args.env_kwargs)
        if args.sensor_xml:
            kw["sensor_xml"] = args.sensor_xml
        if args.sensors:
            kw["sensors"] = json.loads(args.sensors)
        if args.obs_modes:
            kw["obs_modes"] = args.obs_modes
        env = gym.make(args.target if ":" in args.target else f"tactip_mujoco_gym:{args.target}", **kw)
    raw = env.unwrapped
    print(f"{args.target}: action space {env.action_space.shape}")
    mounts = getattr(raw, "sensor_mounts", [])
    print(f"sensor mounts: {', '.join(f'{m.name} (prefix {m.prefix!r})' for m in mounts) or 'none'}")
    if getattr(raw, "sensor_xml", None) is not None:
        print(f"sensor hardware: {raw.sensor_xml}")
    print("inputs:")
    for k, (ok, what) in raw.describe_inputs().items():
        on = "ON " if raw.obs_modes.get(k) else "   "
        print(f"  {on}{k:<15} {'' if ok else 'UNAVAILABLE: '}{what}")
    obs, _ = env.reset(seed=0)
    sp = env.observation_space
    if hasattr(sp, "spaces"):
        print("observation (dict): " + ", ".join(f"{k} {v.shape} {v.dtype}" for k, v in sp.spaces.items()))
    else:
        print(f"observation (flat): {sp.shape}")
    env.close()


def _train(args):
    from tactip_mujoco_gym.experiment import train
    train(args.config, variant=args.variant or None, dry_run=args.dry_run, steps=args.steps,
          tasks=args.tasks or None, skip_trained=args.skip_trained, steps_scale=args.steps_scale)


def _evaluate(args):
    from tactip_mujoco_gym.experiment import evaluate
    evaluate(args.config, variant=args.variant or None, episodes=args.episodes,
             tasks=args.tasks or None)


def _summary(args):
    from tactip_mujoco_gym.experiment import summary
    summary(args.config)


def _visualise(args):
    from tactip_mujoco_gym.experiment import visualise
    visualise(args.config, variant=args.variant, task=args.task, seed=args.seed)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tactip_mujoco_gym",
                                 description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="tasks, sensor readouts and sensor models").set_defaults(fn=_list)

    d = sub.add_parser("describe", help="build a task (or a config's env) and list its inputs")
    d.add_argument("target", help="task id, or a config file")
    d.add_argument("--variant", default=None)
    d.add_argument("--task", default=None, help="a config's task (configs with tasks)")
    d.add_argument("--sensor-xml", default=None, help="sensor hardware (name or MJCF file)")
    d.add_argument("--sensors", default=None, help='readouts as JSON, e.g. \'{"tactile": {"clip": 10}}\'')
    d.add_argument("--obs-modes", default=None, help="e.g. tactile+vision")
    d.add_argument("--env-kwargs", default="{}", help="other env arguments as JSON")
    d.set_defaults(fn=_describe)

    t = sub.add_parser("train", help="train the runs a config describes")
    t.add_argument("config")
    t.add_argument("--variant", nargs="*", default=None, help="only these variants")
    t.add_argument("--tasks", nargs="*", default=None, help="only these tasks (configs with tasks)")
    t.add_argument("--steps", type=int, default=None, help="override train.steps")
    t.add_argument("--steps-scale", type=float, default=None, help="x every run's train.steps (pilots)")
    t.add_argument("--skip-trained", action="store_true", help="leave runs whose model exists (resume)")
    t.add_argument("--dry-run", action="store_true", help="print the runs without training")
    t.set_defaults(fn=_train)

    e = sub.add_parser("evaluate", help="test a config's trained models; save per-trial results")
    e.add_argument("config")
    e.add_argument("--variant", nargs="*", default=None)
    e.add_argument("--tasks", nargs="*", default=None)
    e.add_argument("--episodes", type=int, default=None)
    e.set_defaults(fn=_evaluate)

    m = sub.add_parser("summary", help="results table and chart from a config's saved trials")
    m.add_argument("config")
    m.set_defaults(fn=_summary)

    v = sub.add_parser("visualise", help="watch a config's trained model in the viewer")
    v.add_argument("config")
    v.add_argument("--variant", default=None)
    v.add_argument("--task", default=None)
    v.add_argument("--seed", type=int, default=None, help="which training run (train.seeds)")
    v.set_defaults(fn=_visualise)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
