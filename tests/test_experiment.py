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
    for args, kw in runs:
        assert args[0] in ex.load_config(path)["env"]
        assert kw["obs_modes"] is None or isinstance(kw["obs_modes"], dict)


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
