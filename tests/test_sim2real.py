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
    """listener_MP.py on a Pico, as Controller drives it: moves in whole steps."""

    def __init__(self):
        self.steps = np.zeros(3, int)
        self.calls = 0

    def move(self, x, y, z, a):
        assert all(isinstance(v, int) for v in (x, y, z, a)), "the board takes whole steps"
        assert max(abs(x), abs(y), abs(z)) <= 200
        self.steps += (x, y, z)
        self.calls += 1

    def getPressure(self):
        return [0, 0, 0, 0]


class MockTacTip:
    """A camera whose pins move when the board's z is below a flat surface (and slide with it
    while touching)."""

    def __init__(self, board, spm=25.0, surface_mm=-3.0):
        self.b, self.spm, self.surface = board, spm, surface_mm
        self.truth = Truth(mirror=False, fisheye=0.0)
        self.layout = dome_layout(rings=5)
        self.anchor = None

    def read(self):
        x, y, z = self.b.steps / self.spm
        d = max(0.0, self.surface - z)
        if d <= 0:
            self.anchor = None
        elif self.anchor is None:
            self.anchor = np.array([x, y])
        sx, sy = (np.array([x, y]) - self.anchor) if self.anchor is not None else (0.0, 0.0)
        D = self.truth.world_disp(self.layout, (0.0, 0.0, d, sx, sy))
        return True, render_pins(self.truth.camera(self.layout + D[:, :2]))

    def grab(self):
        return True


def test_real_rig_mocked():
    board = MockBoard()
    rig = RealRig(board, MockTacTip(board), steps_per_mm=(25, 25, 25), settle_s=0, flush=0)
    rig.set_rest(n_frames=3, verbose=False)
    zc = rig.find_contact(step_mm=0.05, verbose=False)
    assert abs(zc - (-3.0)) < 0.15                       # where the mock surface is
    assert abs(rig.pos[2] - 1.0) <= 0.5 / 25 + 1e-9       # backed off to clear_mm (whole steps)
    probes = calibration_probes(depths_mm=(1.0,), shears_mm=(0.5,))
    data = run_probes(rig, probes, verbose=False)
    assert data["pins"].shape == (5, len(rig.rest), 2) and data["ok"].all()
    # slid +x: the pins in contact moved one way, slid -x: the other
    dx = (data["pins"][1] - data["pins"][2] - 0)[:, 0]
    assert abs(dx).max() > 3
    assert np.all(board.steps == np.rint(np.array(rig._raw()) * 25))


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
