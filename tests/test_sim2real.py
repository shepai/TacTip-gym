"""sim2real: the calibration pipeline on a synthetic TacTip whose truth is known (no MuJoCo or
hardware needed), the real-rig driver against a mock board and camera, and the simulated rig
when MuJoCo and the TacTip model are available."""
import os

import numpy as np
import pytest

from tactip_mujoco_gym.sim2real import (MarkerDetector, MarkerTracker, RealRig, RealToSim,
                                        RigCalibration, calibration_probes, fit_calibration,
                                        hinge_fit, match_rest, run_probes)

cv2 = pytest.importorskip("cv2")


# ============================================================================ a synthetic TacTip
def dome_layout(rings=7, pitch=2.2):
    """Marker x, y (mm) down the axis: a centre pin and hexagonal rings, like a TacTip."""
    pts = [(0.0, 0.0)]
    for r in range(1, rings + 1):
        for k in range(6 * r):
            a = 2 * np.pi * k / (6 * r)
            pts.append((r * pitch * np.cos(a), r * pitch * np.sin(a)))
    return np.array(pts)


class Truth:
    """A camera (lens + mount rotation + mirror) and a skin (how markers move in a press)."""

    def __init__(self, theta=np.radians(37), mirror=True, px_per_mm=14.0, centre=(320, 240),
                 fisheye=0.012, frame_R=None, stiffness=1.0, sigma=4.0, seed=0):
        self.theta, self.mirror, self.k = theta, mirror, px_per_mm
        self.c = np.array(centre, float)
        self.fisheye = fisheye
        self.frame_R = np.diag([1.0, -1.0, -1.0]) if frame_R is None else frame_R  # sensor pointing down
        self.stiffness, self.sigma = stiffness, sigma
        self.rng = np.random.default_rng(seed)

    def camera(self, xy):
        """mm on the sensor -> px (barrel distortion grows with radius)."""
        xy = np.asarray(xy, float)
        r = np.linalg.norm(xy, axis=-1, keepdims=True)
        u = xy / (1 + self.fisheye * r ** 2 / 10) * self.k
        c, s = np.cos(self.theta), np.sin(self.theta)
        u = u @ np.array([[c, -s], [s, c]]).T
        if self.mirror:
            u = u * np.array([-1.0, 1.0])
        return u + self.c

    def world_disp(self, layout, probe, stiffness=None):
        """Marker displacement (mm, world x y z) on a flat block: pressed in by `depth`, then
        slid by (shear x, shear y) with the skin sticking. probe: (x, y, depth[, sx, sy]); on
        a flat block x, y don't matter."""
        k = self.stiffness if stiffness is None else stiffness
        d = probe[2]
        sx, sy = probe[3:5] if len(probe) == 5 else (0.0, 0.0)
        if d <= 0:
            return np.zeros((len(layout), 3))
        r = np.linalg.norm(layout, axis=1, keepdims=True) + 1e-9
        g = np.exp(-(r ** 2) / (2 * self.sigma ** 2))
        lat = 0.35 * d / k * g * layout / r * np.minimum(r, self.sigma) / self.sigma
        lat = lat - g * np.array([sx, sy])                   # stuck to the block
        z = 0.8 * d / k * g[:, 0]
        return np.concatenate([lat, z[:, None]], axis=1)


def synthetic_captures(truth, layout, probes, visible=0.8, stiffnesses=(0.5, 1.0, 2.0)):
    rmax = np.linalg.norm(layout, axis=1).max()
    seen = np.flatnonzero(np.linalg.norm(layout, axis=1) <= visible * rmax)
    order = truth.rng.permutation(seen)                       # the camera numbers pins its own way
    rest_px = truth.camera(layout[order]) + truth.rng.normal(0, 0.15, (len(order), 2))
    pins, ok = [], []
    for p in probes:
        D = truth.world_disp(layout, p)
        # the camera also sees the z motion a little (pins come closer -> move outwards)
        moved = layout + D[:, :2] * (1 + 0.05 * D[:, 2:3])
        pins.append(truth.camera(moved[order]) + truth.rng.normal(0, 0.15, (len(order), 2)))
        ok.append(truth.rng.random(len(order)) > 0.02)
    real = dict(probes=probes, pins=np.array(pins), ok=np.array(ok), rest=rest_px,
                steps_per_mm=np.array([25.0, 25.0, 25.0]), axis_sign=np.array([1.0, 1.0, 1.0]),
                force_raw=np.array([4000 * p[2] / truth.stiffness for p in probes]) + 50)
    sims = []
    for k in stiffnesses:
        W = np.array([truth.world_disp(layout, p, k) for p in probes])
        markers = W @ truth.frame_R                            # world -> readout frame (R^T d)
        sims.append(dict(probes=probes, markers=markers, lateral=W[..., :2],
                         force=np.array([2.0 * p[2] / k for p in probes]), layout=layout,
                         frame_R=truth.frame_R, sensor_stiffness=k))
    return real, sims, order


@pytest.fixture(scope="module")
def fitted():
    layout = dome_layout()
    truth = Truth()
    probes = calibration_probes()
    real, sims, order = synthetic_captures(truth, layout, probes)
    calib = fit_calibration(real, sims, verbose=False)
    return calib, truth, layout, real, sims, order, probes


def test_layout_finds_the_right_pins(fitted):
    calib, truth, layout, real, sims, order, probes = fitted
    assert calib.layout_map.mirror == truth.mirror
    right = order[calib.real_idx] == calib.sim_idx
    assert right.mean() > 0.97 and len(calib.real_idx) > 0.9 * len(order)


def test_stiffness_and_maps(fitted):
    calib, truth, *_ = fitted
    assert calib.sim["sensor_stiffness"] == truth.stiffness
    h = calib.metrics["heldout"]
    assert h["per_marker"]["matched"]["r2"] > 0.95
    assert h["per_marker"]["matched"]["r2"] >= h["geometric"]["matched"]["r2"]
    assert calib.metrics["force_r2"] > 0.99


def test_save_load_and_new_session(fitted, tmp_path):
    calib, truth, layout, real, sims, order, probes = fitted
    path = calib.save(str(tmp_path / "c.json"))
    c2 = RigCalibration.load(path)
    # a new session: the camera shifted 1.5 px and the pins come out in another order
    perm = np.random.default_rng(3).permutation(len(c2.real_rest))
    shift = np.array([1.5, -1.0])
    session_rest = real["rest"][perm] + shift
    to_sim = RealToSim(c2, session_rest)
    assert to_sim.n_seen == len(c2.real_rest)
    sim = [s for s in sims if s["sensor_stiffness"] == truth.stiffness][0]
    k = len(probes) - 1                                      # deepest press, slid 1 mm
    pred = to_sim.markers(real["pins"][k][perm] + shift)
    err = np.sqrt(((pred - sim["markers"][k]) ** 2).mean())
    assert err < 0.25 * np.abs(sim["markers"][k]).max()
    obs = to_sim.observation(real["pins"][k][perm] + shift)
    assert obs.shape == (2 * 3 * len(layout),) and obs.dtype == np.float32


# ============================================================================ pieces
def render_pins(xy, shape=(480, 640), radius=4):
    img = np.zeros(shape, np.uint8)
    for x, y in xy:
        cv2.circle(img, (int(round(x * 8)), int(round(y * 8))), radius * 8, 255, -1, shift=3)
    return cv2.GaussianBlur(img, (0, 0), 1.0)


def test_detector_and_tracker():
    truth = Truth(mirror=False, fisheye=0.0)
    xy = truth.camera(dome_layout(rings=6))
    det = MarkerDetector()
    P = det(render_pins(xy))
    d = np.linalg.norm(xy[:, None] - P[None], axis=-1).min(axis=1)
    assert len(P) == len(xy) and d.max() < 0.3
    tr = MarkerTracker(P)
    moved = P + np.array([2.0, -1.0])
    pos, ok = tr.update(det(render_pins(moved)))
    assert ok.all() and np.abs(pos - moved).max() < 0.3
    assert (match_rest(P, P[::-1]) == np.arange(len(P))[::-1]).all()


def test_hinge_fit():
    z = np.linspace(3, -1.5, 91)
    s = 0.05 * np.random.default_rng(0).normal(size=z.size) + 3 * np.maximum(0, 0.4 - z) ** 1.5
    z0, p, b, a = hinge_fit(z, s)
    assert abs(z0 - 0.4) < 0.08


# ============================================================================ the real rig, mocked
class MockBoard:
    """listener_MP.py on a Pico, as Controller drives it: moves in whole steps, + moves stop
    at the limit switches (home: every axis at 0), readButtons() order [z, a, y, x]."""

    def __init__(self, start=(-3000, -2000, -500)):
        self.abs = np.array(start, int)              # where the carriage really is, steps

    def move(self, x, y, z, a):
        assert all(isinstance(v, int) for v in (x, y, z, a)), "the board takes whole steps"
        assert max(abs(x), abs(y), abs(z)) <= 200
        self.abs = np.minimum(self.abs + (x, y, z), 0)

    def sendCommand(self, msg):
        assert msg == "BUTTONS"
        x, y, z = (self.abs >= 0).astype(int)
        return [z, 0, y, x]

    def getPressure(self):
        return [0, 0, 0, 0]


class MockTacTip:
    """A camera whose pins move when the carriage is below a flat surface (and slide with it
    while touching)."""

    def __init__(self, board, spm=25.0, surface=-2000):
        self.b, self.spm, self.surface = board, spm, surface
        self.truth = Truth(mirror=False, fisheye=0.0)
        self.layout = dome_layout(rings=5)
        self.anchor = None

    def read(self):
        x, y, z = self.b.abs / self.spm
        d = max(0.0, (self.surface - self.b.abs[2]) / self.spm)
        if d <= 0:
            self.anchor = None
        elif self.anchor is None:
            self.anchor = np.array([x, y])
        sx, sy = (np.array([x, y]) - self.anchor) if self.anchor is not None else (0.0, 0.0)
        D = self.truth.world_disp(self.layout, (0.0, 0.0, d, sx, sy))
        return True, render_pins(self.truth.camera(self.layout + D[:, :2]))

    def grab(self):
        return True


def test_real_rig_setup_then_reset():
    board = MockBoard()
    rig = RealRig(board, MockTacTip(board), steps_per_mm=(25, 25, 25), settle_s=0, flush=0)
    # real-setup: home, the preset towards the block, rest image high up, jog down, contact
    rig.home(verbose=False)
    assert (board.abs == 0).all()
    rig.move_steps(-1900, -6400, 0)
    rig.set_rest(n_frames=3, verbose=False)
    rig.move_to(z=-2000 / 25 + 3)                         # jogged to 3 mm above the block
    rig.find_contact(verbose=False)
    rig.home_offset = rig.contact_offset()
    assert abs(rig.home_offset[2] - (-2000)) <= 4 and rig.home_offset[1] == -6400

    # a later session: the carriage is somewhere else and has missed steps
    rig.z_contact = None
    board.abs = np.array([-2500, -3100, -900])
    rig.reset(verbose=False)                              # home, then the start pose
    expect = rig.home_offset + np.array([0, 0, 10 * 25])
    assert (board.abs == expect).all()                    # 10 mm above the block centre
    rig.find_contact(verbose=False)                       # down to 3 mm, then the search
    assert abs(rig.z_contact) < 0.15                      # where the setup expected it

    probes = calibration_probes(depths_mm=(1.0,), shears_mm=(0.5,))
    data = run_probes(rig, probes, verbose=False)
    assert data["pins"].shape == (5, len(rig.rest), 2) and data["ok"].all()
    dx = (data["pins"][1] - data["pins"][2])[:, 0]        # slid +x vs -x
    assert abs(dx).max() > 3
    rig.reset(verbose=False)
    assert (board.abs == rig.home_offset + np.array([0, 0, 10 * 25 + int(round(rig.z_contact * 25))])).all()


def test_contact_search_refuses_to_start_touching():
    board = MockBoard()
    rig = RealRig(board, MockTacTip(board), steps_per_mm=(25, 25, 25), settle_s=0, flush=0,
                  home_offset=(-1900, -6400, -2000))
    rig.home(verbose=False)
    rig.set_rest(n_frames=2, verbose=False)
    rig.move_to(x=0, y=0, z=-1.5)                         # already pressed in
    with pytest.raises(RuntimeError, match="already touching"):
        super(RealRig, rig).find_contact(verbose=False)   # search from here, no approach


# ============================================================================ the simulated rig
def _have_tactip_model():
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    root = os.path.join(os.path.dirname(__file__), "..", "src", "tactip_mujoco_gym", "assets")
    return (os.path.exists(os.path.join(root, "sensor", "tactip.xml"))
            or os.path.exists(os.path.join(root, "robotarm", "assets", "z1_Link00.stl")))


@pytest.mark.skipif(not _have_tactip_model(), reason="needs MuJoCo and the TacTip model")
def test_sim_rig_contact_and_presses():
    from tactip_mujoco_gym.sim2real import SimRig
    rig = SimRig()
    rig.reset(verbose=False)
    assert abs(rig.gap_mm() - 10.0) < 0.5                 # the start pose: 10 mm above
    zc = rig.find_contact(verbose=False)
    rig.move_to(z=0.0)
    rig.settle()
    assert abs(rig.gap_mm()) < 0.5                        # z = 0 is the surface
    data = run_probes(rig, calibration_probes(depths_mm=(0.5, 1.5), shears_mm=(0.5,)), verbose=False)
    assert data["markers"].shape == (10, rig.n, 3)
    assert data["force"][5] > data["force"][0] > 0          # 1.5 mm vs 0.5 mm, no slide
    # sliding +x and -x move the markers in contact opposite ways along x
    assert data["lateral"][1, :, 0].sum() * data["lateral"][2, :, 0].sum() < 0
    assert rig.layout.shape == (rig.n, 2) and np.isfinite(zc)
    rig.close()


# ============================================================================ pose calibration
def _true_map(sim_mm):
    """A rig whose x runs backwards against the simulator's, 25 steps / mm."""
    s = np.atleast_2d(sim_mm)
    return s @ np.diag([-25.0, 25.0, 25.0]).T + np.array([-1900.0, -6400.0, -1750.0])


def test_pose_map_axis_fit_and_rig_setup(tmp_path):
    from tactip_mujoco_gym.sim2real import PoseMap
    pm = PoseMap()
    for p in ([0, 0, 0], [5, 0, 0], [0, 5, 0], [0, 0, -5], [3, -4, -8]):
        pm.add(p, _true_map(p)[0] + np.random.default_rng(len(pm.points)).integers(-1, 2, 3))
    pm.fit("axis")
    assert np.allclose(np.diag(pm.A), [-25, 25, 25], rtol=0.02)
    assert pm.residual_mm.max() < 0.1
    setup = pm.rig_setup()
    assert setup["axis_sign"] == [-1, 1, 1]
    assert np.allclose(setup["home_offset"], [-1900, -6400, -2000], atol=3)   # contact: 10 mm down
    pm2 = PoseMap.load(pm.save(str(tmp_path / "p.json")))
    assert np.allclose(pm2.to_steps([1, 2, 3]), pm.to_steps([1, 2, 3]))
    assert np.allclose(pm2.to_sim(pm2.to_steps([1, 2, 3])), [1, 2, 3])


def test_pose_map_needs_variation_or_known_scale():
    from tactip_mujoco_gym.sim2real import PoseMap
    pm = PoseMap()
    pm.add([0, 0, 0], _true_map([0, 0, 0])[0])
    pm.add([5, 0, 0], _true_map([5, 0, 0])[0])
    with pytest.raises(ValueError, match="y"):
        pm.fit("axis")
    pm.known_spm, pm.known_sign = np.array([25.0] * 3), np.array([-1, 1, 1])
    pm.fit("axis")
    assert np.allclose(pm.to_steps([0, 2, -3]), _true_map([0, 2, -3])[0])


def test_pose_map_affine_recovers_a_skewed_rig():
    from tactip_mujoco_gym.sim2real import PoseMap
    th = np.radians(3)                                     # gantry 3 degrees off square
    A = 25 * np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    pm = PoseMap()
    for p in np.random.default_rng(0).uniform(-8, 8, (8, 3)):
        pm.add(p, A @ p + [-1900, -6400, -1750])
    pm.fit("affine")
    assert np.allclose(pm.A, A, atol=0.2) and pm.residual_mm.max() < 0.05


def test_stepper_worker_chunks_clamps_and_homes():
    from tactip_mujoco_gym.sim2real.teleop import StepperWorker
    board = MockBoard()
    rig = RealRig(board, MockTacTip(board), steps_per_mm=(25, 25, 25), settle_s=0, flush=0)
    rig.home(verbose=False)
    w = StepperWorker(rig, chunk=50)
    w.set_target([-1234, -567, -89])
    w.wait(10)
    assert (board.abs == [-1234, -567, -89]).all() and (rig._steps == board.abs).all()
    w.nudge(2, +500)                                        # can't go above home
    w.wait(10)
    assert board.abs[2] == 0
    board.abs = board.abs + [-40, 0, 0]                     # the motors slipped 40 steps
    w.request_home()
    w.wait(10)
    assert (board.abs == 0).all() and (rig._steps == 0).all()
    w.stop()
    assert w.error is None


def test_connect_controller_uses_installed_package(monkeypatch, tmp_path):
    """Rig-controller as a package: its own listener unless one is given."""
    import sys
    import types
    from tactip_mujoco_gym.sim2real import connect_controller
    seen = {}

    class Controller:
        def __init__(self, port, variant="circuitpython", file="packaged/listener_MP.py"):
            seen.update(port=port, variant=variant, file=file)
    monkeypatch.setitem(sys.modules, "rig_controller", types.SimpleNamespace(Controller=Controller))

    connect_controller("/dev/ttyACM0")
    assert seen == dict(port="/dev/ttyACM0", variant="micropython", file="packaged/listener_MP.py")
    mine = tmp_path / "listener_MP.py"
    mine.write_text("# board code")
    connect_controller("/dev/ttyACM0", str(mine))
    assert seen["file"] == str(mine)
    with pytest.raises(FileNotFoundError, match="leave --listener out"):
        connect_controller("/dev/ttyACM0", ".../Rig-controller/Code/Examples/Board Examples/listener_MP.py")


def test_too_few_pins_is_an_error(tmp_path, monkeypatch):
    """A blank or wrong camera must not pass for a TacTip (it once 'found' 2 pins)."""
    monkeypatch.chdir(tmp_path)

    class Blank:
        def read(self):
            return True, np.zeros((480, 640), np.uint8)

        def grab(self):
            return True
    rig = RealRig(MockBoard(), Blank(), steps_per_mm=(25, 25, 25), settle_s=0, flush=0)
    with pytest.raises(RuntimeError, match="pins found"):
        rig.set_rest(n_frames=2, verbose=False)
    assert (tmp_path / "rest_failed.png").exists()


def test_real_setup_opens_the_simulator(monkeypatch):
    import tactip_mujoco_gym.sim2real.__main__ as cli
    got = {}
    monkeypatch.setattr(cli, "cmd_pose_calibrate", lambda a: got.update(vars(a)))
    cli.main(["real-setup", "--port", "/dev/ttyACM0", "--camera", "2", "--measure-steps",
              "--out-dir", "calib"])
    assert got["out"] == os.path.join("calib", "pose_calibration.json")
    assert got["setup_out"] == os.path.join("calib", "rig_setup.json") and got["camera"] == 2
