"""Surface following (arm), the Cartesian rig and the TacTip gripper. Skipped without MuJoCo,
and without the TacTip (assets/sensor/tactip.xml, or the arm meshes to extract it from)."""
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

import tactip_mujoco_gym  # noqa: E402,F401
from conftest import ASSETS, ROOT, have_arm_meshes, needs_arm  # noqa: E402

needs_tactip = pytest.mark.skipif(
    not (have_arm_meshes() or os.path.exists(os.path.join(ASSETS, "sensor", "tactip.xml"))),
    reason="no TacTip: assets/sensor/tactip.xml and the arm meshes are both missing")


def _no_attach_notices(capfd, caught):
    out, err = capfd.readouterr()
    assert "Attach conflict" not in out + err
    assert not [w for w in caught if "Attach conflict" in str(w.message)]


def _hfield(raw, hid):
    m = raw.model
    adr, nr, nc = m.hfield_adr[hid], m.hfield_nrow[hid], m.hfield_ncol[hid]
    return m.hfield_data[adr:adr + nr * nc].reshape(nr, nc)


# ----------------------------------------------------------------------------- arm
@needs_arm
@pytest.mark.parametrize("kw", [{}, {"wave": "chirp", "texture": "patches"},
                                {"wave": "uniform", "texture": 4, "follow_normal": False}],
                         ids=["default", "chirp_patches", "uniform_t4_depth_only"])
def test_surface_follow(kw):
    """Starts pressed onto the strip; the heightfield is the surface the task computes with;
    the scripted follower (check_envs.surface_policy) gets well along it."""
    from tactip_mujoco_gym.check_envs import surface_policy
    env = gym.make("TacArm-surface", obs_modes=["privileged", "tactile"], **kw)
    raw = env.unwrapped
    assert env.action_space.shape == ((2,) if kw.get("follow_normal", True) else (1,))
    obs, info = env.reset(seed=0)
    assert raw._touching(raw._surface_body), "the sensor should start pressed onto the strip"
    assert abs(info["depth_err"]) < 0.002
    stored = raw.STRIP_TOP - raw.HF_RANGE / 2 + _hfield(raw, raw._hf) * raw.HF_RANGE
    np.testing.assert_allclose(stored, raw._surf.H, atol=1e-6)
    if kw.get("texture") == "patches":
        assert len(info["textures"]) > 1
    while True:
        obs, r, term, trunc, info = env.step(surface_policy(raw))
        assert np.isfinite(obs).all()
        if term or trunc:
            break
    print(f"{kw}: progress {info['progress']:.2f}, success {info['is_success']}, "
          f"ended: {info['fail_reason'] or ('success' if info['is_success'] else 'time')}")
    assert info["progress"] > 0.4, f"the scripted follower stopped early: {info['fail_reason']}"
    env.close()


# ----------------------------------------------------------------------------- rig
@needs_tactip
def test_rig_texture(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make("TacRig-texture", obs_modes=["tactile", "force"], obs_format="dict")
    _no_attach_notices(capfd, caught)
    raw = env.unwrapped
    assert [m.name for m in raw.sensor_mounts] == ["tip"]
    assert env.action_space.shape == (3 + 1 + 6,)
    obs, info = env.reset(seed=0, options={"texture": 4, "start": (0.0, 0.0)})
    assert info["texture"] == 4 and info["plate_force"] == 0.0         # hovering above it
    from tactip_mujoco_gym import textures as tx
    np.testing.assert_allclose(_hfield(raw, raw._hf) * 7.0, tx.texture_heights(4, raw.TEXTURE_RES_MM),
                               atol=1e-4)
    assert 0.005 < raw._gap() < 0.015                                   # HOVER above the top
    obs = raw.press_to(1.9)
    assert 1.5 <= raw.plate_force < 5.0, raw.plate_force
    assert raw._gap() < 0.0, "pressing 1.9 N should push the tip below the surface"
    assert np.abs(obs["tactile"]).max() > 0.05                          # markers moved (mm)
    assert obs["force"][0] == pytest.approx(raw.plate_force / raw.F_REF)
    a = np.zeros(env.action_space.shape)
    a[3], a[4 + raw.texture_index] = 1.0, 1.0                            # answer: the right one
    obs, r, term, trunc, info = env.step(a)
    assert term and info["is_success"] and info["guess"] == raw.texture_index and r > 5
    env.close()


@needs_tactip
def test_rig_tilt_stl_and_protocol(tmp_path):
    from test_textures import write_stl
    from tactip_mujoco_gym.envs.tactile_rig import press_grid
    stl = str(tmp_path / "t2.stl")
    write_stl(stl, 2)
    env = gym.make("TacRig-v0", textures=[1, stl])
    raw = env.unwrapped
    _, info = env.reset(seed=0, options={"texture": stl, "tilt": 0.2})
    assert info["texture"] == stl and info["tilt"] == pytest.approx(0.2)
    assert raw._gap() > 0.0                                             # still above the block
    data = press_grid(env, points=2, span_mm=(4.0, 4.0), repeats=1, verbose=False)
    assert data["X"].shape[0] == 8 and set(data["y"]) == {0, 1}
    assert (data["force"] > 1.5).all()
    a, b = data["X"][data["y"] == 0], data["X"][data["y"] == 1]
    assert np.abs(a.mean(0) - b.mean(0)).max() > 0.01                  # the textures differ
    env.close()


@needs_tactip
def test_rig_with_another_sensor():
    """The rig takes any sensor hardware and readout, like every robot."""
    from examples.custom_sensor.pad_contact import PadContact
    env = gym.make("TacRig-v0", sensor_xml=os.path.join(ROOT, "examples", "custom_sensor", "rigid_pad.xml"),
                   sensors={"contact": PadContact()}, obs_modes=["contact", "force"], obs_format="dict")
    raw = env.unwrapped
    assert not raw.describe_inputs()["tactile"][0]
    env.reset(seed=0, options={"start": (0.0, 0.0)})
    obs = raw.press_to(1.0)
    assert raw.plate_force >= 1.0 and np.linalg.norm(obs["contact"][:3]) > 0.5
    env.close()


# ----------------------------------------------------------------------------- gripper
@needs_tactip
def test_gripper(capfd):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        env = gym.make("TacGripper-pick", obs_modes=["object", "tactile"])
    _no_attach_notices(capfd, caught)
    raw = env.unwrapped
    assert [m.name for m in raw.sensor_mounts] == ["left", "right"]
    assert env.action_space.shape == (5,)
    env.reset(seed=0)
    tips = raw._tips_local()
    assert np.linalg.norm(tips["left"] - tips["right"]) == pytest.approx(raw.START_WIDTH, abs=0.005)
    env.step(np.zeros(5))
    assert raw.get_hand_nodes().shape[0] == 2


@needs_tactip
@pytest.mark.parametrize("obj", ["box", "soft_cube"])
def test_gripper_scripted_pick(obj):
    """The humanoid's scripted picker drives the gripper too: it should at least lift."""
    from tactip_mujoco_gym.envs.tactip_humanoid import ScriptedPicker
    env = gym.make("TacGripper-pick", level=0.0)
    raw = env.unwrapped
    env.reset(seed=1, options={"object": obj})
    picker = ScriptedPicker(env)
    lifted = False
    for _ in range(raw.max_steps):
        _, _, term, trunc, info = env.step(picker())
        lifted |= raw._was_lifted
        if term or trunc:
            break
    tips = raw._tips_local()
    state = (f"phase {picker.phase}, grip force {np.round(raw._hand_force, 2)} N, tips "
             f"{np.round([(tips[s][2] - raw.TABLE_TOP) * 1000 for s in tips], 1)} mm above the "
             f"table, width cmd {raw._width_cmd * 1000:.0f} mm, object {raw._obj[3] * 1000:.0f} mm "
             f"wide, table force {raw._table_force:.1f} N")
    print(f"{obj}: lifted {lifted}, success {info['is_success']}, fail {info['fail_reason']!r}; {state}")
    assert lifted, f"the scripted picker didn't lift the {obj}: {state}"
    env.close()
