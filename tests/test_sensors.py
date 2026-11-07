"""Readouts and sensor hardware on tiny inline models (no assets needed)."""
import os
import types

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("gymnasium")

from conftest import ROOT  # noqa: E402
from tactip_mujoco_gym.sensor_tools import attach_sensor, sensor_info  # noqa: E402
from tactip_mujoco_gym.sensors import (Camera, Mount, Sensor, SensorUnavailable, TacTipCamera,  # noqa: E402
                                       TacTipMarkers, TaxelEvents, list_sensors, make_sensor,
                                       register_sensor, sensor_class)

PAD = os.path.join(ROOT, "examples", "custom_sensor", "rigid_pad.xml")

FINGER = """
<mujoco>
  <worldbody>
    <light pos="0 0 1"/>
    <geom name="floor" type="plane" size="1 1 0.1"/>
    <body name="mount" pos="0 0 0.3">
      <site name="ee_site"/>
      <camera name="sensor_cam" pos="0 0 -0.01" zaxis="0 0 1"/>
      <camera name="handcam" pos="0 0.2 0" xyaxes="1 0 0 0 0 1"/>
      <body name="pin0" pos="0.01 0 -0.02"><joint name="p0" type="slide" axis="0 0 1"/>
        <geom type="sphere" size="0.002" mass="0.001" group="2" contype="0" conaffinity="0"/>
        <site name="s_c0"/></body>
      <body name="pin1" pos="-0.01 0 -0.02"><joint name="p1" type="slide" axis="1 0 0"/>
        <geom type="sphere" size="0.002" mass="0.001" group="2" contype="0" conaffinity="0"/>
        <site name="s_c1"/></body>
    </body>
  </worldbody>
</mujoco>"""


def finger_env(xml=FINGER):
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    mount = m.body("mount").id
    return types.SimpleNamespace(model=m, data=d, MARKER_PREFIX="s_c", SENSOR_SITE="ee_site",
                                 MARKER_SCALE=1e-3, MARKER_CLIP=20.0, SENSOR_CAM="sensor_cam",
                                 VISION_CAMERA="handcam",
                                 sensor_mounts=[Mount("tip", "", mount, ("body", mount))])


def test_markers_report_displacement_in_mm():
    env = finger_env()
    s = TacTipMarkers()
    s.bind(env)
    assert s.n_markers == 2 and s.counts == [2]
    s.calibrate(env, env.data)
    s.reset(env)
    env.data.qpos[env.model.joint("p0").qposadr[0]] = 0.003          # pin 0 up 3 mm
    mujoco.mj_forward(env.model, env.data)
    s.env_step(env)
    obs = s.observe(env)
    assert obs.shape == (12,)
    disp, vel = obs[:6].reshape(2, 3), obs[6:].reshape(2, 3)
    np.testing.assert_allclose(disp[0], [0, 0, 3.0], atol=1e-4)
    np.testing.assert_allclose(vel[0], [0, 0, 3.0], atol=1e-4)
    np.testing.assert_allclose(disp[1], 0, atol=1e-6)
    s.env_step(env)                                                    # no motion since
    np.testing.assert_allclose(s.observe(env)[6:], 0, atol=1e-6)
    assert np.allclose(s.displacement(env)[0, 0], [0, 0, 0.003], atol=1e-7)


def test_markers_unavailable_without_sites():
    env = finger_env(FINGER.replace('name="s_c', 'name="m_c'))
    with pytest.raises(SensorUnavailable, match="s_c"):
        TacTipMarkers().bind(env)
    s = TacTipMarkers(marker_prefix="m_c")
    s.bind(env)
    assert s.n_markers == 2


def test_camera_readouts_render():
    env = finger_env()
    try:
        mujoco.Renderer(env.model, 8, 8).close()
    except Exception as err:                       # no OpenGL here
        pytest.skip(f"no OpenGL: {err}")
    cam = Camera(width=40, height=30)
    cam.bind(env)
    img = cam.observe(env)
    assert img.shape == (30, 40, 3) and img.dtype == np.uint8
    assert cam.flatten(img).max() <= 1.0
    depth = Camera(width=20, height=20, mode="depth")
    depth.bind(env)
    assert depth.observe(env).shape == (20, 20, 1)
    tc = TacTipCamera(width=16, height=16)
    tc.bind(env)
    assert tc.observe(env).shape == (16, 16, 3)
    for r in (cam, depth, tc):
        r.close()
    env.VISION_CAMERA = "nope"
    with pytest.raises(SensorUnavailable, match="nope"):
        Camera().bind(env)


def test_taxel_events_fire_on_contact():
    xml = """
<mujoco>
  <worldbody>
    <body name="plate"><geom type="box" size="0.1 0.1 0.01"/>
      <site name="tx_0" type="box" pos="0 0 0.01" size="0.1 0.1 0.004"/></body>
    <body name="box" pos="0 0 0.05"><freejoint/><geom type="box" size="0.02 0.02 0.02" mass="0.2"/></body>
  </worldbody>
  <sensor><touch name="tx_0" site="tx_0"/></sensor>
</mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    env = types.SimpleNamespace(model=m, data=d, TAXEL_PREFIX="tx_", EVENT_THRESHOLD=0.02)
    s = TaxelEvents()
    s.bind(env)
    assert m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR       # off while unused
    s.active = True
    s.on_select(env, True)
    assert not m.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_SENSOR
    for _ in range(300):
        mujoco.mj_step(m, d)
        s.physics_step(env)
    assert s.events().shape[0] > 0
    assert s.forces(env)[0] > 1.0                                       # the box rests on it


def test_rigid_pad_tip_from_site():
    info = sensor_info(PAD)
    np.testing.assert_allclose(info["tip"], [0, 0, 0.024], atol=1e-9)
    np.testing.assert_allclose(info["axis"], [0, 0, 1], atol=1e-9)
    assert info["n_markers"] == 0 and abs(info["radius"] - 0.02) < 1e-6


def test_attach_markerless_sensor_puts_the_tip_where_asked():
    spec = mujoco.MjSpec.from_string("""
<mujoco><worldbody><body name="link" pos="0 0 1"><geom type="capsule" size="0.02 0.05"/></body>
</worldbody></mujoco>""")
    attach_sensor(spec, spec.body("link"), PAD, prefix="pad_", tip_at=[0.1, 0, -0.2], axis_to=[0, 0, -1])
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    tip = m.site("pad_sensor_tip").id
    np.testing.assert_allclose(d.site_xpos[tip], [0.1, 0, 0.8], atol=1e-6)
    np.testing.assert_allclose(d.site_xmat[tip].reshape(3, 3)[:, 2], [0, 0, -1], atol=1e-6)


class Doubler(Sensor):
    """Test readout."""
    key = "double"

    def __init__(self, k=2.0):
        self.k = k

    def observe(self, env):
        return np.array([self.k], np.float32)


def test_registry_forms():
    register_sensor("doubler_test", Doubler, overwrite=True)
    assert {"tactip_markers", "tactip_camera", "camera", "taxel_events"} <= set(list_sensors())
    assert isinstance(make_sensor("doubler_test"), Doubler)
    assert make_sensor({"type": "doubler_test", "k": 3}).k == 3
    assert make_sensor({"k": 5}, default="doubler_test").k == 5
    assert sensor_class("tactip_mujoco_gym.sensors.camera:Camera") is Camera
    a = Doubler(k=4)
    b = make_sensor(a)
    assert b is not a and b.k == 4
    spec = b.spec()
    assert spec["k"] == 4 and spec["type"].endswith(":Doubler")         # importable path
    with pytest.raises(KeyError):
        make_sensor("no_such_sensor")
