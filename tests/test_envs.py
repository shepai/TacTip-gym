"""The tasks themselves (skipped when the robot meshes / models are not in assets/)."""
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

import tactip_mujoco_gym  # noqa: E402,F401
from conftest import ROOT, needs_arm, robot_available  # noqa: E402

def _no_attach_notices(capfd, caught):
    """MuJoCo reports option conflicts both from C (straight to the console, in every worker
    process) and as Python warnings: neither should appear."""
    out, err = capfd.readouterr()
    assert "Attach conflict" not in out + err
    assert not [w for w in caught if "Attach conflict" in str(w.message)]


PRIVILEGED = {"TacArm-edge": (9, 13), "TacArm-peg": (8, 12), "TacArm-shape": (8, 17),
              "TacArm-hill": (8, 15)}


@needs_arm
@pytest.mark.parametrize("task", sorted(PRIVILEGED))
def test_arm_default_observation_unchanged(task):
    env = gym.make(task)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (sum(PRIVILEGED[task]),)
    obs, r, term, trunc, info = env.step(env.action_space.sample())
    assert np.isfinite(obs).all() and "is_success" in info
    env.close()


@needs_arm
def test_arm_every_input():
    env = gym.make("TacArm-peg", obs_modes=["privileged", "tactile", "tactile_image", "vision", "events"],
                   obs_format="dict")
    raw = env.unwrapped
    sp = env.observation_space.spaces             # (gymnasium.spaces.Dict sorts its keys)
    assert set(sp) == {"proprio", "privileged", "tactile", "tactile_image", "vision", "events"}
    assert sp["tactile"].shape == (6 * raw.n_markers,) and raw.n_markers == 71
    assert sp["vision"].shape == (64, 64, 3) and sp["tactile_image"].shape == (64, 64, 3)
    assert sp["events"].shape == (raw.n_taxels,)
    obs, _ = env.reset(seed=0)                    # the observation itself keeps the pipeline order
    assert list(obs) == ["proprio", "privileged", "tactile", "tactile_image", "vision", "events"]
    obs, *_ = env.step(env.action_space.sample())
    assert all(obs[k].shape == sp[k].shape for k in sp)
    env.close()


@needs_arm
def test_attached_tactip_matches_the_builtin_one(capfd):
    a = gym.make("TacArm-peg", obs_modes=["tactile"]).unwrapped
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        b = gym.make("TacArm-peg", obs_modes=["tactile"], sensor_xml="tactip").unwrapped
    _no_attach_notices(capfd, caught)
    assert b.sensor_mounts[0].prefix == "sensor_" and a.n_markers == b.n_markers
    # same marker rest shape in the sensor frame, and the same tip position for the arm
    np.testing.assert_allclose(a._nodes_rest, b._nodes_rest, atol=2e-4)
    np.testing.assert_allclose(a._tip_pos(), b._tip_pos(), atol=1e-3)
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    assert oa.shape == ob.shape
    a.close()
    b.close()


@needs_arm
def test_custom_sensor_on_the_arm():
    from examples.custom_sensor.pad_contact import PadContact
    from tactip_mujoco_gym.check_envs import peg_policy
    env = gym.make("TacArm-peg", sensor_xml=os.path.join(ROOT, "examples", "custom_sensor", "rigid_pad.xml"),
                   sensors={"contact": PadContact()}, obs_modes=["contact"], obs_format="dict",
                   overrides={"TIP_R": 0.02})
    raw = env.unwrapped
    assert not raw.describe_inputs()["tactile"][0]                    # no markers on a pad
    obs, _ = env.reset(seed=0)
    forces = []
    for _ in range(40):
        obs, *_ = env.step(peg_policy(raw))
        forces.append(np.linalg.norm(obs["contact"][:3]))
    assert max(forces) > 0.01
    env.close()


def test_quadruped(capfd):
    from tactip_mujoco_gym.envs.tactip_quadruped import TacQuadruped
    if not robot_available(TacQuadruped):
        pytest.skip("go2.xml / meshes missing")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make("TacQuad-walk", obs_modes=["tactile"], feet="tactip")
    _no_attach_notices(capfd, caught)
    raw = env.unwrapped
    assert [m.name for m in raw.sensor_mounts] == ["FL", "FR", "RL", "RR"]
    obs, _ = env.reset(seed=0)
    assert obs.shape[0] > 6 * raw.n_markers
    env.step(np.zeros(12))
    assert raw.get_foot_nodes().shape == (4, raw.n_markers // 4, 3)
    proxy = gym.make("TacQuad-walk", feet="proxy").unwrapped
    assert not proxy.describe_inputs()["tactile"][0]
    with pytest.raises(RuntimeError):
        proxy.set_obs_modes(["tactile"])


def test_humanoid(capfd):
    from tactip_mujoco_gym.envs.tactip_humanoid import TacHumanoidPick
    if not robot_available(TacHumanoidPick):
        pytest.skip("RB-Y1 model missing")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make("TacHumanoid-pick", obs_modes=["object", "tactile"])
    _no_attach_notices(capfd, caught)
    raw = env.unwrapped
    assert [m.name for m in raw.sensor_mounts] == ["left", "right"]
    env.reset(seed=0)
    env.step(np.zeros(env.action_space.shape))
    assert raw.get_hand_nodes().shape[0] == 2
