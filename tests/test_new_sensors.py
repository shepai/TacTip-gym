"""The PressTip, GelSight, depth-image TacTip and whisker sensors: the hardware alone, the
readouts each brings, pressed on the rig, on the gripper and the arm, and the sensor viewer.
Skipped without MuJoCo; none of them needs the TacTip."""
import os
import subprocess
import sys

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
gym = pytest.importorskip("gymnasium")

import tactip_mujoco_gym  # noqa: E402,F401
from conftest import ROOT, needs_arm  # noqa: E402
from tactip_mujoco_gym.sensor_tools import compile_spec, sensor_info  # noqa: E402
from tactip_mujoco_gym.sensors import (ContactDepth, ContactMap, PressTip, Whiskers,  # noqa: E402
                                       hardware, list_sensor_models, make_sensor,
                                       resolve_sensor_model, sensor_model_readouts)
from tactip_mujoco_gym.sensors.depth import exit_distance, pixel_rays, shade_gelsight  # noqa: E402

NEW = ["presstip", "gelsight", "tactip_depth", "whiskers"]


# ----------------------------------------------------------------------------- hardware
@pytest.mark.parametrize("name", NEW)
def test_hardware_alone(name, tmp_path):
    """Registered; compiles; tip on +z; no joint rings faster than MAX_OMEGA; sits still
    when simulated on its own; and the MJCF it writes loads back the same."""
    assert name in list_sensor_models()
    m = compile_spec(resolve_sensor_model(name))
    info = sensor_info(resolve_sensor_model(name))
    np.testing.assert_allclose(info["axis"], [0, 0, 1], atol=1e-9)
    k = m.jnt_stiffness
    arm = m.dof_armature[m.jnt_dofadr]
    assert (k > 0).all()
    assert (np.sqrt(k / arm) <= hardware.MAX_OMEGA * 1.001).all()
    d = mujoco.MjData(m)
    mujoco.mj_step(m, d, 3000)
    assert np.isfinite(d.qpos).all()
    assert np.abs(d.qvel).max() < 1e-3, "the sensor should settle under gravity"
    assert np.abs(d.qpos).max() < 5e-4, "and sag very little"
    path = hardware.write(name, str(tmp_path / f"{name}.xml"))
    m2 = compile_spec(resolve_sensor_model(path))
    assert (m2.nbody, m2.njnt, m2.ngeom, m2.nsite, m2.ncam) == (m.nbody, m.njnt, m.ngeom, m.nsite, m.ncam)


def test_hardware_shapes():
    assert len(hardware.presstip_pads()) == 15
    xml = hardware.whisker_array_xml(rows=2, cols=3, segments=3)
    m = compile_spec(mujoco.MjSpec.from_string(xml))
    assert m.njnt == 2 * 3 * 3 * 2
    tips = hardware.whisker_tips(rows=2, cols=3, segments=3)
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    got = np.array([d.site_xpos[m.site(f"w{i}_tip").id] for i in range(6)])
    np.testing.assert_allclose(got, tips, atol=1e-6)
    assert m.site("sensor_tip").pos[2] == pytest.approx(tips[:, 2].max(), abs=1e-6)


# ----------------------------------------------------------------------------- readout maths
def test_depth_reference_maths():
    G = mujoco.mjtGeom
    rays = pixel_rays(120, 32, 32).reshape(-1, 3)
    R_cam = np.diag([1.0, -1.0, -1.0])                   # camera looking along +z
    t = exit_distance(G.mjGEOM_SPHERE, np.array([0.035, 0, 0]), np.zeros(3), rays @ R_cam.T)
    np.testing.assert_allclose(t, 0.035)                 # from the dome's centre: its radius
    D = np.array([[0, 0, 1.0], [1, 0, 0]])
    t = exit_distance(G.mjGEOM_BOX, np.array([0.01, 0.01, 0.002]), np.array([0, 0, -0.01]), D)
    assert t[0] == pytest.approx(0.012) and np.isnan(t[1])   # the far face of a slab; a miss
    flat = shade_gelsight(np.zeros((8, 8)), 1e-4)
    np.testing.assert_allclose(flat, flat[0, 0])         # flat gel: one colour
    yy, xx = np.mgrid[0:24, 0:24]
    dent = np.clip(1e-3 - ((xx - 12) ** 2 + (yy - 12) ** 2) * 1e-5, 0, None)
    assert np.ptp(shade_gelsight(dent, 3e-4).reshape(-1, 3), axis=0).min() > 0.1


def test_presets_and_tweaks():
    assert sensor_model_readouts("whiskers") == {"tactile": "whiskers"}
    assert sensor_model_readouts("tactip") == {} and sensor_model_readouts("some/file.xml") == {}
    env = gym.make("TacRig-v0", sensor_xml="gelsight", sensors={"tactile_image": {"mode": "depth"}})
    raw = env.unwrapped
    img, grid = raw.sensors["tactile_image"], raw.sensors["tactile"]
    assert type(img) is ContactDepth and img.mode == "depth" and img.max_depth == 0.0015
    assert type(grid) is ContactMap and grid.available
    assert raw.sensors["vision"].spec()["type"] == "camera"      # defaults it doesn't bring stay
    again = make_sensor(img.spec())
    assert again.spec() == img.spec()
    env.close()


# ----------------------------------------------------------------------------- on the rig
def _rig(name, keys):
    env = gym.make("TacRig-v0", sensor_xml=name, obs_modes=keys, obs_format="dict")
    raw = env.unwrapped
    obs, _ = env.reset(seed=0, options={"texture": 1, "start": (0.0, 0.0)})
    return env, raw, obs


def _slide(env, raw, steps=40):
    """Slide along x keeping the height; returns the observations."""
    a = np.zeros(env.action_space.shape)
    a[0] = 0.5
    return [env.step(a)[0] for _ in range(steps)]


@pytest.mark.parametrize("name", ["gelsight", "tactip_depth"])
def test_contact_depth_on_rig(name):
    env, raw, obs = _rig(name, ["tactile_image", "tactile"])
    assert obs["tactile_image"].dtype == np.uint8
    assert obs["tactile_image"].shape == ((48, 64, 3) if name == "gelsight" else (64, 64, 1))
    if name == "tactip_depth":
        assert obs["tactile_image"].max() == 0, "hovering: nothing touches"
    assert obs["tactile"].max() < 1e-6
    obs = raw.press_to(1.9)
    assert raw.plate_force >= 1.5
    print(f"{name}: contact depth up to {obs['tactile'].max():.2f} mm on the grid")
    assert obs["tactile"].max() > 0.05                         # mm
    if name == "tactip_depth":
        assert obs["tactile_image"].max() > 20
    else:
        flat = obs["tactile_image"][0, 0].astype(int)            # a corner: no contact
        assert np.abs(obs["tactile_image"].astype(int) - flat).max() > 10
    env.close()


def test_presstip_on_rig():
    env, raw, obs = _rig("presstip", ["tactile"])
    s = raw.sensors["tactile"]
    assert type(s) is PressTip and obs["tactile"].shape == (15 + 1 + 3,)
    assert np.abs(obs["tactile"][:15]).max() == 0.0
    np.testing.assert_allclose(np.linalg.norm(obs["tactile"][16:]), 1.0, atol=0.05)   # 1 g
    obs = raw.press_to(1.9)
    pads = obs["tactile"][:15] * s.f_ref
    print(f"presstip pads (N): {np.round(pads, 2)}, plate {raw.plate_force:.2f} N")
    assert pads.sum() > 0.5 * raw.plate_force
    assert pads.argmax() == 0 or pads[0] > 0.1, "a press in the middle loads the middle pad"
    piezo = [o["tactile"][15] for o in _slide(env, raw)]
    assert max(piezo) > 0.0
    env.close()


def test_whiskers_on_rig():
    env, raw, obs = _rig("whiskers", ["tactile"])
    s = raw.sensors["tactile"]
    assert type(s) is Whiskers and s.counts == [12]
    o = obs["tactile"].reshape(12, 4)
    assert o[:, 2].sum() == 0 and np.abs(o[:, :2]).max() < 0.05, "hovering: untouched, straight"
    obs = raw.press_to(1.9, max_travel=0.015)                 # (whiskers barely push back)
    o = obs["tactile"].reshape(12, 4)
    print(f"whiskers touching: {int(o[:, 2].sum())}, bend up to {np.abs(o[:, :2]).max() * s.angle_ref:.3f} rad")
    assert o[:, 2].sum() >= 1 and np.abs(o[:, :2]).max() > 0.1
    vib = [o["tactile"].reshape(12, 4)[:, 3].max() for o in _slide(env, raw)]
    assert max(vib) > 0.0
    env.close()


# ----------------------------------------------------------------------------- other robots
@pytest.mark.parametrize("name", NEW)
def test_on_gripper(name):
    keys = list(sensor_model_readouts(name))
    env = gym.make("TacGripper-pick", sensor_xml=name, obs_modes=keys, obs_format="dict")
    raw = env.unwrapped
    assert [m.name for m in raw.sensor_mounts] == ["left", "right"]
    obs, _ = env.reset(seed=0)
    for _ in range(5):
        obs, *_ = env.step(np.zeros(env.action_space.shape))
    for k in keys:
        assert np.isfinite(np.asarray(obs[k], dtype=float)).all()
    if name == "gelsight":
        assert obs["tactile_image"].shape[-1] == 6                # two fingers, RGB each
    env.close()


@needs_arm
@pytest.mark.parametrize("name", NEW)
def test_on_arm(name):
    from tactip_mujoco_gym.check_envs import surface_policy
    keys = list(sensor_model_readouts(name))
    env = gym.make("TacArm-surface", sensor_xml=name, obs_modes=keys, obs_format="dict")
    raw = env.unwrapped
    obs, _ = env.reset(seed=0)
    for _ in range(20):
        obs, *_ = env.step(surface_policy(raw))
    for k in keys:
        assert np.isfinite(np.asarray(obs[k], dtype=float)).all()
    env.close()


# ----------------------------------------------------------------------------- the viewer
@pytest.mark.parametrize("name", ["presstip", "gelsight", "whiskers"])
def test_view_sensors_script(name, tmp_path):
    pytest.importorskip("matplotlib")
    out = tmp_path / f"{name}.png"
    script = os.path.join(ROOT, "examples", "demos", "view_sensors.py")
    run = subprocess.run([sys.executable, script, "TacRig-v0", "--sensor-xml", name, "--steps", "40",
                          "--save", str(out), "--one-window"], capture_output=True, text=True,
                         timeout=600, env={**os.environ, "MPLBACKEND": "Agg"})
    assert run.returncode == 0, run.stdout + run.stderr
    assert out.exists() and out.stat().st_size > 10_000
