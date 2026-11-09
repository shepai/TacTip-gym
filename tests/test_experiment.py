"""Experiment configs (no simulation)."""
import glob
import os

import pytest

pytest.importorskip("mujoco")
from conftest import ROOT  # noqa: E402
from tactip_mujoco_gym import experiment as ex  # noqa: E402
from tactip_mujoco_gym.__main__ import main  # noqa: E402

CONFIGS = sorted(glob.glob(os.path.join(ROOT, "configs", "*.yaml")))


@pytest.mark.parametrize("path", CONFIGS, ids=os.path.basename)
def test_configs_resolve(path):
    pytest.importorskip("yaml")
    cfg = ex.load_config(path)
    runs = ex.train(cfg, dry_run=True)
    assert runs
    names = [args[1] for args, _ in runs]
    assert len(set(names)) == len(names)
    envs = {cfg.get("env")} | {(t or {}).get("env", k) for k, t in (cfg.get("tasks") or {}).items()}
    for args, kw in runs:
        assert args[0] in envs
        assert kw["obs_modes"] is None or isinstance(kw["obs_modes"], dict)


def test_tasks_variants_and_seeds():
    cfg = ex.load_config({
        "name": "x", "train": {"steps": 100, "seeds": 2, "record_video": False},
        "variants": {"t": {"obs_modes": ["tactile"]},
                     "v": {"obs_modes": ["vision"], "obs_format": "dict"},
                     "o": {"obs_modes": ["tactile", "object"]}},
        "tasks": {"TacArm-peg": {"train": {"steps": 5}},
                  "TacGripper-pick": {"env_kwargs": {"sensor_stiffness": 3.0}}}})
    runs, skipped = ex.runs(cfg)
    names = [r["name"] for r in runs]
    assert skipped == [("TacArm-peg", "o", ["object"])]          # the arm has no object input
    assert len(runs) == 2 * 2 + 3 * 2 and len(set(names)) == len(names)
    assert "x_TacArm-peg_t_s0" in names and "x_TacGripper-pick_o_s1" in names
    calls = {args[1]: (args, kw) for args, kw in ex.train(cfg, dry_run=True)}
    args, kw = calls["x_TacArm-peg_v_s1"]
    assert args[0] == "TacArm-peg" and args[3] == 5 and kw["seed"] == 1
    assert kw["obs_format"] == "dict" and kw["record_video"] is False
    args, kw = calls["x_TacGripper-pick_t_s0"]
    assert args[3] == 100 and kw["env_kwargs"] == {"sensor_stiffness": 3.0}
    assert ex.train(cfg, dry_run=True, tasks=["TacArm-peg"], steps_scale=2.0)[0][0][3] == 10
    assert ex.resolve(cfg, "t", "TacArm-peg")["env"] == "TacArm-peg"
    with pytest.raises(ValueError, match="tasks"):
        ex.resolve(cfg, "t")
    with pytest.raises(ValueError, match="top level"):
        ex.load_config({"env": "TacArm-peg", "variants": {"a": {"tasks": {"TacArm-hill": {}}}}})


def test_results_summary(tmp_path):
    from tactip_mujoco_gym import results as rs
    lo, hi = rs.wilson(0, 20)
    assert lo == 0.0 and abs(hi - 0.161) < 1e-3
    rows = [{"run": f"r{s}", "task": "T", "variant": v, "seed": s, "trial": k,
             "success": int(k < n), "return": 1.0, "length": 10, "fail_reason": ""}
            for v, n in (("a", 10), ("b", 20)) for s in (0, 1) for k in range(20)]
    summ = {x["variant"]: x for x in rs.summarise(rows)}
    assert summ["a"]["success"] == 0.5 and summ["a"]["seeds"] == 2 and summ["b"]["success"] == 1.0
    path = tmp_path / "episodes.csv"
    rs.replace_episodes(str(path), "r0", rows[:20])
    rs.replace_episodes(str(path), "r0", rows[:20])                    # replaces, not appends
    assert len(rs.read_episodes(str(path))) == 20
    assert rs.write_summary(str(tmp_path)) and (tmp_path / "summary.csv").exists()


def test_variants_merge_and_typos():
    cfg = ex.load_config({"env": "TacArm-peg", "sensors": {"tactile": {"clip": 5}},
                          "variants": {"a": {"obs_modes": ["tactile"], "sensors": {"vision": {"width": 32}}}}})
    (name, c), = ex.variants(cfg)
    kw = ex.env_kwargs(c)
    assert kw["sensors"] == {"tactile": {"clip": 5}, "vision": {"width": 32}}
    assert kw["obs_modes"] == {"tactile": True} and ex.run_name(c, name) == "TacArm-peg_tactile_a"
    with pytest.raises(ValueError, match="unknown"):
        ex.load_config({"env": "TacArm-peg", "sensor": {}})
    with pytest.raises(ValueError, match="variants"):
        ex.resolve(cfg)


def test_cli_dry_run(capsys):
    pytest.importorskip("yaml")
    main(["train", os.path.join(ROOT, "configs", "peg_sensing.yaml"), "--dry-run", "--variant", "tactile"])
    out = capsys.readouterr().out
    assert "peg_tactile" in out and "'tactile': True" in out
    main(["list"])
    assert "TacArm-peg" in capsys.readouterr().out
    main(["train", os.path.join(ROOT, "configs", "sensing_trials.yaml"), "--dry-run",
          "--tasks", "TacArm-peg", "TacRig-texture", "--variant", "tactip", "vision"])
    out = capsys.readouterr().out
    assert "sensing_trials_TacArm-peg_tactip" in out and "sensing_trials_TacRig-texture_vision" in out
    assert "4 runs" in out
