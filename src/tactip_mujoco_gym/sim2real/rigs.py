"""
The real rig and the simulated one behind one interface, in millimetres, so the same press
protocol runs on both.

    sim  = SimRig()                                        # TacRig-v0 over a flat block
    real = RealRig(connect_controller("COM19", listener), open_camera(1),
                   **load_rig_setup("calib/rig_setup.json"))

    for rig in (sim, real):
        rig.reset()                        # the start pose: 10 mm above the block centre
        rig.find_contact()                 # z = 0 where the sensor first touches the surface
        data = run_probes(rig, grid_probes(points=5, span_mm=6, depths_mm=(0.5, 1, 1.5, 2)))
        rig.reset()

Both rigs start from the same pose: the simulated one's reset (TacRig-v0 puts the tip HOVER,
10 mm, above the block's top at its centre), and on the real rig, homing against the limit
switches (z first, then x and y) followed by a fixed move from home to that same pose. The
fixed move (home_offset, board steps from home to the block centre at contact height) is
measured once by `python -m tactip_mujoco_gym.sim2real real-setup` and saved with the steps
per mm in rig_setup.json. Homing again before every session (and texture) also clears any
steps the motors missed.

Coordinates (both rigs): x, y in mm from the block centre along the simulator's rig axes, z in
mm above the point of first contact found by find_contact() (negative = pressed in). Moves
are absolute. Positions are the commanded ones, as a stepper rig knows them.

The real rig is github.com/shepai/Rig-controller, MicroPython variant: a Pico running
Code/Examples/Board Examples/listener_MP.py, driven from the PC by Code/Controller
(Controller(port, file=listener_path)). Moves go to the board as whole steps in short chunks
(each MOVE is one exec on the board, and the board must answer within mpremote's timeout).
"""
import datetime
import json
import os
import sys
import tempfile
import time

import numpy as np

from .markers import MarkerDetector, MarkerTracker


# ============================================================================ helpers
def contact_signal(disp, top=0.1):
    """How much the sensor is deformed: mean of the largest `top` fraction of the marker
    displacement magnitudes (the contact patch, not diluted by the untouched pins)."""
    mag = np.linalg.norm(np.asarray(disp, float).reshape(len(disp), -1), axis=1)
    if not len(mag):
        return 0.0
    k = max(1, int(np.ceil(top * len(mag))))
    return float(np.sort(mag)[-k:].mean())


def hinge_fit(z, s, powers=(1.0, 1.5, 2.0), resolution=None):
    """Where contact starts, from a descent: signal s(z) = b + a * max(0, z0 - z)^p, fitted
    by least squares over z0 (and p in `powers`). Returns (z0, p, b, a). Threshold-free, so
    a noisy camera (px) and the simulation (mm) find the same physical point."""
    z = np.asarray(z, float)
    s = np.asarray(s, float)
    if len(z) < 4:
        raise ValueError("need at least 4 points of the descent")
    lo, hi = z.min(), z.max()
    step = resolution or max((hi - lo) / 400, 1e-4)
    best = (np.inf, hi, 1.0, float(s.mean()), 0.0)
    for p in powers:
        for z0 in np.arange(lo, hi + step, step):
            h = np.maximum(0.0, z0 - z) ** p
            A = np.stack([np.ones_like(h), h], axis=1)
            coef, *_ = np.linalg.lstsq(A, s, rcond=None)
            if coef[1] <= 0:
                continue
            sse = float(((A @ coef - s) ** 2).sum())
            if sse < best[0]:
                best = (sse, float(z0), float(p), float(coef[0]), float(coef[1]))
    return best[1], best[2], best[3], best[4]


def flat_block_stl(path=None):
    """An STL box (a flat block) for the simulated rig: scaled to the dataset block's size
    (29.3 x 30 x 7 mm), its top is level with the textured blocks' highest points."""
    path = path or os.path.join(tempfile.gettempdir(), "tactip_gym_flat_block.stl")
    if not os.path.exists(path):
        v = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]
        faces = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]
        lines = ["solid flat"]
        for a, b, c, d in faces:
            for tri in ((a, b, c), (a, c, d)):
                lines.append("facet normal 0 0 0\nouter loop")
                lines += [f"vertex {v[i][0]} {v[i][1]} {v[i][2]}" for i in tri]
                lines.append("endloop\nendfacet")
        lines.append("endsolid flat")
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
    return path


# ============================================================================ common
class Rig:
    """Shared logic: absolute moves in mm with z measured from contact, contact finding."""
    kind = "rig"
    signal_units = ""
    default_stop = 1.0
    start_height_mm = 10.0             # the start pose: this far above the block centre's top

    def __init__(self):
        self.z_contact = None          # raw z (mm, the backend's own zero) of first contact
        self.contact_trace = None

    # ---- to implement
    def _raw(self):
        """Commanded position (x, y, z) mm in the backend's own frame."""
        raise NotImplementedError

    def _move_raw(self, x, y, z):
        raise NotImplementedError

    def read(self):
        raise NotImplementedError

    def signal(self):
        """contact_signal() of the current reading."""
        raise NotImplementedError

    def settle(self):
        pass

    def reset(self):
        """Back to the start pose: over the block centre, start_height_mm above it."""
        raise NotImplementedError

    # ---- shared
    @property
    def pos(self):
        x, y, z = self._raw()
        return np.array([x, y, z - (self.z_contact or 0.0)])

    def move_to(self, x=None, y=None, z=None):
        """Absolute move (mm); None keeps an axis where it is. z is from contact once
        find_contact() has run."""
        cx, cy, cz = self._raw()
        zc = self.z_contact or 0.0
        self._move_raw(cx if x is None else float(x), cy if y is None else float(y),
                       cz if z is None else float(z) + zc)

    def find_contact(self, step_mm=0.05, max_travel_mm=6.0, stop=None, past_mm=0.6,
                     clear_mm=1.0, verbose=True):
        """Lower the sensor from where it is in step_mm steps, reading the sensor after each,
        until its contact signal passes `stop` (then a further past_mm), and fit where the
        deformation began (hinge_fit). Sets z = 0 there and backs off to z = clear_mm.
        Start a few mm above the surface. Returns the contact height in raw mm."""
        stop = self.default_stop if stop is None else stop
        x, y, z0 = self._raw()
        self.settle()
        zs, sig = [z0], [self.signal()]
        if sig[0] >= stop:
            raise RuntimeError(f"[{self.kind}] the sensor is already touching where the contact "
                               f"search starts (signal {sig[0]:.3g} {self.signal_units}): "
                               "start higher")
        n_max = int(np.ceil(max_travel_mm / step_mm))
        past_at = None
        for k in range(1, n_max + 1):
            z = z0 - k * step_mm
            self._move_raw(x, y, z)
            self.settle()
            zs.append(z)
            sig.append(self.signal())
            if past_at is None and sig[-1] >= stop:
                past_at = z - past_mm
            if past_at is not None and z <= past_at:
                break
        if past_at is None:
            self._move_raw(x, y, z0)
            raise RuntimeError(f"no contact within {max_travel_mm} mm (signal reached "
                               f"{max(sig):.3g} {self.signal_units}, stop at {stop})")
        zc, p, b, a = hinge_fit(zs, sig, resolution=step_mm / 10)
        self.z_contact = zc
        self.contact_trace = dict(z=np.array(zs), signal=np.array(sig), z0=zc, power=p)
        if verbose:
            print(f"[{self.kind}] contact at {zc:.3f} mm (raw), signal ~ (z0 - z)^{p:g}; "
                  f"descended {z0 - zs[-1]:.2f} mm")
        self._move_raw(x, y, zc + clear_mm)
        self.settle()
        return zc


def grid_probes(points=5, span_mm=6.0, depths_mm=(0.5, 1.0, 1.5, 2.0), centre_mm=(0.0, 0.0),
                repeats=1):
    """Press positions: a points x points grid span_mm wide, each pressed to every depth.
    Rows (x mm, y mm, depth mm)."""
    xs = centre_mm[0] + np.linspace(-span_mm / 2, span_mm / 2, points) if points > 1 else [centre_mm[0]]
    ys = centre_mm[1] + np.linspace(-span_mm / 2, span_mm / 2, points) if points > 1 else [centre_mm[1]]
    rows = [(x, y, d) for _ in range(repeats) for x in xs for y in ys for d in depths_mm]
    return np.array(rows, float)


def calibration_probes(depths_mm=(0.5, 1.0, 1.5, 2.0), shears_mm=(0.5, 1.0), points=1,
                       span_mm=0.0, repeats=1):
    """The calibration protocol on a flat block. Rows (x, y, depth, shear x, shear y) mm:
    press to each depth, then (except for the plain press) slide by each shear along +-x
    and +-y before reading.

    A flat block dents the skin the same way wherever it is pressed, and the dent is round:
    it shows how far the markers move, not which way the camera is turned. The shears do:
    while the skin sticks to the block (keep them small, ~1 mm), the markers in contact stay
    with the block, so they move by -shear relative to the sensor on both rigs."""
    base = grid_probes(points, span_mm, depths_mm, repeats=repeats)
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    rows = []
    for x, y, d in base:
        rows.append((x, y, d, 0.0, 0.0))
        rows += [(x, y, d, s * u, s * v) for s in shears_mm for u, v in dirs]
    return np.array(rows, float)


def run_probes(rig, probes, clear_mm=1.5, approach_mm=0.5, shear_step_mm=0.25, verbose=True):
    """Press at every probe: rows (x, y, depth) or (x, y, depth, shear x, shear y) mm. Up to
    clear_mm, across, down in approach_mm steps to -depth, slide by the shear in
    shear_step_mm steps, settle, read, back up. Returns {"probes", <every key read()
    returns, stacked>}; the same protocol and timing on either rig."""
    if rig.z_contact is None:
        raise RuntimeError("find_contact() first: depths are measured from contact")
    probes = np.asarray(probes, float)
    probes = probes.reshape(-1, probes.shape[-1] if probes.ndim > 1 else 3)
    if probes.shape[1] not in (3, 5):
        raise ValueError("probe rows are (x, y, depth) or (x, y, depth, shear x, shear y)")
    recs = []
    t0 = time.time()
    for k, row in enumerate(probes):
        x, y, d = row[:3]
        sx, sy = row[3:5] if len(row) == 5 else (0.0, 0.0)
        rig.move_to(z=clear_mm)
        rig.move_to(x=x, y=y)
        for z in np.arange(clear_mm - approach_mm, -d, -approach_mm):
            rig.move_to(z=z)
        rig.move_to(z=-d)
        n = int(np.ceil(np.hypot(sx, sy) / shear_step_mm))
        for f in np.arange(1, n + 1) / max(n, 1):
            rig.move_to(x=x + f * sx, y=y + f * sy)
        rig.settle()
        recs.append(rig.read())
        rig.move_to(z=clear_mm)
        if verbose and (k % 10 == 0 or k == len(probes) - 1):
            el = time.time() - t0
            sh = f", shear ({sx:+.2f}, {sy:+.2f})" if (sx or sy) else ""
            print(f"[{rig.kind}] press {k + 1}/{len(probes)} at ({x:+.2f}, {y:+.2f}) mm, "
                  f"depth {d:.2f} mm{sh}  ({el / (k + 1):.1f} s each)")
    rig.move_to(x=0.0, y=0.0, z=clear_mm)
    out = {"probes": probes}
    for key in recs[0]:
        vals = [r[key] for r in recs]
        if all(v is None for v in vals):
            continue
        out[key] = np.stack([np.asarray(v if v is not None else np.nan) for v in vals])
    return out


# ============================================================================ simulation
class SimRig(Rig):
    """TacRig-v0 (envs/tactile_rig.py) driven by position. Moves go through env.step() (the
    rig's velocity actions, solved so the commanded axes land exactly on target), so the
    sensor's dynamics are the ones every experiment sees.

    read() -> {"markers": (n, 3) mm   marker displacement from rest, in the readout's frame
                                     (exactly the first half of the "tactile" input),
               "lateral": (n, 2) mm   the same displacements along the rig's x and y,
               "force":   N           the plate's force sensor}
    """
    kind = "sim"
    signal_units = "mm"
    default_stop = 0.15

    def __init__(self, env=None, textures=None, sensor_stiffness=1.0, sensor_xml=None,
                 overrides=None, settle_steps=10, readout="tactile"):
        super().__init__()
        if env is None:
            import gymnasium as gym
            import tactip_mujoco_gym  # noqa: F401  (registers the tasks)
            textures = list(textures) if textures is not None else [flat_block_stl()]
            env = gym.make("TacRig-v0", textures=textures, sensor_stiffness=sensor_stiffness,
                           sensor_xml=sensor_xml, overrides=overrides)
        self.env = env
        self.e = e = env.unwrapped
        self.sensor_stiffness = float(getattr(e, "sensor_stiffness", sensor_stiffness))
        self.settle_steps = int(settle_steps)
        self.readout_key = readout
        e.set_obs_format("dict")
        e.set_obs_modes([readout])
        self.readout = e.sensors[readout]
        if not getattr(self.readout, "available", True):
            raise RuntimeError(f"readout {readout!r} unavailable: {self.readout.reason}")
        self.n = int(self.readout.n_markers)
        self.start_height_mm = float(getattr(e, "HOVER", 0.010)) * 1000.0
        self.set_texture(e.textures[0])
        self._frame_R = self._frame_rotation()
        self.layout = self._layout()

    # ---- environment
    def set_texture(self, texture):
        """Put `texture` (one of env.textures) in the holder and reset to the start pose. The
        contact height stays valid: every block is 7 mm tall."""
        self.e.reset(options={"texture": texture, "start": (0.0, 0.0), "tilt": 0.0})
        self.texture = texture

    def reset(self, texture=None, verbose=True):
        """The env's own reset: the tip at the block centre, HOVER (10 mm) above its top."""
        self.set_texture(self.texture if texture is None else texture)
        if verbose:
            print(f"[sim] reset: over the block centre, {self.gap_mm():.1f} mm above it")

    def _frame_rotation(self):
        from tactip_mujoco_gym.sensors import frame_pose
        frames = getattr(self.readout, "_frames", None)
        if not frames:
            return np.eye(3)
        return frame_pose(self.e.data, frames[0])[1].copy()

    def _layout(self):
        """Rest marker positions seen down the sensor axis: rig x-y (mm) from the tip."""
        from tactip_mujoco_gym.sensors import frame_pose
        rest = np.asarray(self.readout.rest, float)               # readout frame, m
        tip = np.asarray(self.e.data.site_xpos[self.e._tip_site], float)
        frames = getattr(self.readout, "_frames", None)
        c = np.asarray(frame_pose(self.e.data, frames[0])[0], float) if frames else tip
        world = rest @ self._frame_R.T + c                        # nodes() inverted
        return (world - tip)[:, :2] * 1000.0

    def _raw(self):
        return tuple(float(v) * 1000.0 for v in self.e._cmd)

    def _action(self, a3):
        a = np.zeros(self.e.action_space.shape, dtype=np.float32)
        a[:3] = a3
        if getattr(self.e, "_decide_index", None) is not None:
            a[self.e._decide_index] = -1.0                        # (TacRig-texture: no answer)
        return a

    def limits_mm(self):
        """The rig's travel (raw mm): (lo, hi) per axis."""
        return self.e._lo * 1000.0, self.e._hi * 1000.0

    def step_toward(self, target_mm):
        """One env step towards target (raw mm, clipped to the travel): the velocity action
        that lands the commanded axes exactly on it, at the rig's speed. Returns True once
        there and stopped. For live control: one call per frame."""
        e = self.e
        target = np.clip(np.asarray(target_mm, float) / 1000.0, e._lo, e._hi)
        err = target - e._cmd
        if np.all(np.abs(err) < 1e-7) and np.all(np.abs(e._vel_cmd) < 1e-6):
            e.step(self._action(np.zeros(3)))           # hold (the physics keeps running)
            return True
        k = float(e.ACTION_SMOOTHING)
        per_step = np.asarray(e.AXIS_SPEED, float) * e.dt
        v_des = np.clip(err / per_step, -1.0, 1.0)
        a = np.clip((v_des - k * e._vel_cmd) / (1.0 - k), -1.0, 1.0)
        e.step(self._action(a))
        return False

    def _move_raw(self, x, y, z, max_steps=5000):
        for _ in range(max_steps):
            if self.step_toward([x, y, z]):
                break
        else:
            raise RuntimeError(f"the simulated rig didn't reach {[x, y, z]} mm")

    def settle(self):
        for _ in range(self.settle_steps):
            self.e.step(self._action(np.zeros(3)))

    def gap_mm(self):
        """The tip's height above the surface under it (mm; the simulation's ground truth)."""
        return self.e._gap() * 1000.0

    def approach(self, gap_mm=1.5):
        """Go quickly to gap_mm above the surface under the tip (the simulation knows where
        the surface is; on the real rig this is the jog to just above the block)."""
        x, y, z = self._raw()
        self._move_raw(x, y, z - (self.gap_mm() - gap_mm))
        self.settle()

    def find_contact(self, gap_mm=3.0, **kw):
        """From wherever the sensor is (the start pose, normally): straight down to gap_mm
        above the surface, then the slow search (as the real rig does)."""
        self.approach(gap_mm)
        return super().find_contact(**kw)

    # ---- sensing
    @property
    def marker_scale(self):
        """Metres per unit of the "tactile" input (env.MARKER_SCALE, 1e-3: mm)."""
        return float(getattr(self.readout, "_scale", getattr(self.e, "MARKER_SCALE", 1e-3)))

    @property
    def marker_clip(self):
        return float(getattr(self.readout, "_clip", getattr(self.e, "MARKER_CLIP", 20.0)))

    def displacement(self):
        """(n, 3) mm in the readout's frame (the readout's own clipping applies)."""
        obs = np.asarray(self.readout.observe(self.e), float)
        return obs[:3 * self.n].reshape(self.n, 3) * self.marker_scale * 1000.0

    def read(self):
        d = self.displacement()
        return {"markers": d, "lateral": (d @ self._frame_R.T)[:, :2],
                "force": float(self.e.plate_force)}

    def observation(self):
        """The full "tactile" input, as a policy would get it."""
        return np.asarray(self.readout.observe(self.e), np.float32)

    def signal(self):
        return contact_signal(self.displacement())

    def close(self):
        self.env.close()


# ============================================================================ real rig
def _controller_class(rig_code_dir=None):
    """Rig-controller's Controller: the installed package (pip install -e Rig-controller,
    `rig_controller`), else the older Code/Controller folder (rig_code_dir)."""
    try:
        from rig_controller import Controller
        return Controller
    except ImportError:
        pass
    if rig_code_dir and rig_code_dir not in sys.path:
        sys.path.insert(1, rig_code_dir)
    try:
        import Controller as old
        return old.Controller
    except ImportError as err:
        raise ImportError("can't import Rig-controller: pip install it (pip install -e "
                          "path/to/Rig-controller), or pass rig_code_dir= its Code/ folder") from err


def connect_controller(port, listener_path=None, rig_code_dir=None):
    """Rig-controller's Controller on `port`, MicroPython variant. listener_path: a board
    listener to run instead of the package's own (None: the packaged one, which the
    installed package uses by default)."""
    import inspect
    cls = _controller_class(rig_code_dir)
    params = inspect.signature(cls).parameters
    kw = {}
    if "variant" in params:
        kw["variant"] = "micropython"
    if listener_path:
        if not os.path.exists(listener_path):
            raise FileNotFoundError(f"no board listener at {listener_path!r} (leave --listener "
                                    "out to use the one packaged with Rig-controller)")
        kw["file"] = listener_path
    elif "file" in params and params["file"].default is inspect.Parameter.empty:
        raise ValueError("this Rig-controller version needs the board listener's path: "
                         "--listener path/to/listener_MP.py")
    return cls(port, **kw)


CAMERA_HINT = ("find the TacTip's index with `python -m tactip_mujoco_gym.sim2real cameras` "
               "(on Linux each USB camera usually takes two /dev/video numbers, and only the "
               "even one gives pictures)")


def open_camera(index=0, width=None, height=None):
    """An OpenCV camera (the TacTip's), optionally at a set resolution. Fails unless it
    actually delivers frames (OpenCV can 'open' a device that never does)."""
    import cv2
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"camera {index} didn't open: {CAMERA_HINT}")
    if width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    good = 0
    for _ in range(10):
        ok, f = cap.read()
        good += bool(ok and f is not None and f.size)
    if good == 0:
        cap.release()
        raise RuntimeError(f"camera {index} gives no pictures: {CAMERA_HINT}")
    return cap


def list_cameras(max_index=10, save_dir=None):
    """The camera indices that deliver pictures: [(index, width, height)]; with save_dir, a
    snapshot of each (camera_<index>.png) so you can see which one is the TacTip."""
    import cv2
    found = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(10):
                ok, f = cap.read()
                if ok and f is not None and f.size:
                    frame = f
            if frame is None:
                continue
            found.append((i, frame.shape[1], frame.shape[0]))
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                cv2.imwrite(os.path.join(save_dir, f"camera_{i}.png"), frame)
        finally:
            cap.release()
    return found


class RealRig(Rig):
    """The Rig-controller rig (MicroPython board) with a TacTip camera, in the simulator's
    coordinates.

    controller    Controller(port, file=listener_MP.py) from Rig-controller (or anything with
                  move(x, y, z, a) in steps and, for the plate, getPressure())
    camera        cv2.VideoCapture of the TacTip (or anything with read() -> (ok, frame))
    steps_per_mm  (x, y, z): measure them with measure_steps_per_mm()
    axis_sign     (x, y, z): +1 if a positive step count moves the sensor along the
                  simulator's + axis relative to the block (z: + is up), else -1. The
                  board's z is negative-down, so z is +1.
    plate         read the base's force plate (listener_MP.py with Rig(plate_mode=1 or 2))
    max_chunk     most steps per axis in one MOVE (each MOVE has to finish within mpremote's
                  timeout, and a timed-out MOVE is resent by Controller.sendCommand)
    settle_s      wait after a move before reading (s)
    flush         camera frames thrown away before a reading (drivers buffer old frames)
    track         look at the pins after every move, so the tracker follows them through
                  a press (run_probes moves in small steps for this) and none are lost
    home_offset   board steps (x, y, z) from home (all limit switches pressed) to the block
                  centre at contact height: measured by real-setup, in rig_setup.json.
                  Without it the rig can home but not reach the start pose by itself.
    home_buttons  readButtons() index of the x, y, z switches (the board's pins: [z, a, y, x])

    read() -> {"pins": (n, 2) px tracked pin positions (rest order), "ok": (n,) found this
               frame, "force_raw": plate total (or None)}
    """
    kind = "real"
    signal_units = "px"
    default_stop = 4.0

    def __init__(self, controller, camera, steps_per_mm, axis_sign=(1, 1, 1), detector=None,
                 plate=False, max_chunk=200, settle_s=0.3, flush=4, track=True,
                 home_offset=None, home_buttons=(3, 2, 0), home_chunk=100, **_):
        super().__init__()
        self.c = controller
        self.cam = camera
        self.spm = np.asarray(steps_per_mm, float).reshape(3)
        self.sign = np.asarray(axis_sign, float).reshape(3)
        if np.any(self.spm <= 0):
            raise ValueError("steps_per_mm must be positive (direction goes in axis_sign)")
        self.detector = detector or MarkerDetector()
        self.plate = bool(plate)
        self.max_chunk = int(max_chunk)
        self.settle_s = float(settle_s)
        self.flush = int(flush)
        self.track = bool(track)
        self._steps = np.zeros(3, dtype=np.int64)    # steps from the block centre at contact
        self.home_offset = None if home_offset is None else np.asarray(home_offset, np.int64)
        self.home_buttons = tuple(home_buttons)
        self.home_chunk = int(home_chunk)
        self.homed = False
        self.tracker = None
        self.force_zero = 0.0

    # ---- motion
    def _raw(self):
        return tuple(self.sign * self._steps / self.spm)

    def _move_raw(self, x, y, z):
        target = np.rint(np.array([x, y, z]) * self.spm * self.sign).astype(np.int64)
        self.move_steps(*(target - self._steps))
        if self.track and self.tracker is not None:
            self.pins()

    def move_steps(self, dx, dy, dz):
        """Relative move in board steps, sent in chunks of at most max_chunk per axis."""
        rem = np.array([dx, dy, dz], dtype=np.int64)
        while np.any(rem != 0):
            n = max(1, int(np.ceil(np.abs(rem).max() / self.max_chunk)))
            part = (rem / n).astype(np.int64)
            part[part == 0] = np.sign(rem[part == 0])
            self.c.move(int(part[0]), int(part[1]), int(part[2]), 0)
            self._steps += part
            rem -= part

    def buttons(self):
        """Limit switch states, the board's readButtons() order ([z, a, y, x] on the Pico)."""
        for name in ("getButtons", "buttons", "readButtons"):      # newer Controller methods
            f = getattr(self.c, name, None)
            if callable(f):
                return [int(v) for v in f()]
        return [int(v) for v in self.c.sendCommand("BUTTONS")]

    def home(self, max_steps=60000, verbose=True):
        """Drive to the limit switches, as Controller.reset() does (+ steps until pressed),
        but z first: the sensor lifts clear before x and y move, so it can't drag across the
        block. Afterwards positions count from the block centre (via home_offset), or from
        home if there is none yet (real-setup)."""
        ix, iy, iz = self.home_buttons
        limit = int(max_steps / self.home_chunk) + 1     # in steps: works before steps/mm are known
        for axes in (("z",), ("x", "y")):
            for _ in range(limit):
                b = self.buttons()
                todo = [a for a in axes if not b[{"x": ix, "y": iy, "z": iz}[a]]]
                if not todo:
                    break
                d = [self.home_chunk if a in todo else 0 for a in "xyz"]
                self.c.move(d[0], d[1], d[2], 0)
            else:
                raise RuntimeError(f"{'/'.join(axes)} didn't reach its limit switch within "
                                   f"{max_steps} steps: check the switches "
                                   f"(home_buttons={self.home_buttons})")
        self._steps = (-self.home_offset.copy() if self.home_offset is not None
                       else np.zeros(3, dtype=np.int64))
        self.homed = True
        if self.tracker is not None:
            self.tracker.reset()                     # lifted clear: the pins are at rest
        if verbose:
            print("[real] homed (all limit switches pressed)")

    def go_to_start(self, verbose=True):
        """From home to the start pose: across at the top, then down to start_height_mm
        above the block centre (above contact once find_contact() has run)."""
        if not self.homed or self.home_offset is None:
            raise RuntimeError("home() first, with home_offset from real-setup (rig_setup.json)")
        zc = self.z_contact or 0.0
        x, y, z = self._raw()
        self._move_raw(0.0, 0.0, z)
        self._move_raw(0.0, 0.0, zc + self.start_height_mm)
        self.settle()
        if verbose:
            print(f"[real] at the start pose: over the block centre, {self.start_height_mm:g} mm up")

    def reset(self, verbose=True):
        """Home, then the start pose: the same pose the simulated rig resets to."""
        self.home(verbose=verbose)
        self.go_to_start(verbose=verbose)

    def find_contact(self, gap_mm=3.0, **kw):
        """With a home_offset, contact is expected at z = 0: go straight down to gap_mm above
        it, then search slowly (as the simulated rig does). Without one (real-setup), search
        down from where the sensor is."""
        if self.home_offset is not None and self.homed:
            x, y, _ = self._raw()
            self._move_raw(x, y, gap_mm)
            self.settle()
        return super().find_contact(**kw)

    def contact_offset(self):
        """Board steps from home to here (x, y) at the contact height: the home_offset, once
        homed without one and find_contact() has run."""
        if self.z_contact is None:
            raise RuntimeError("find_contact() first")
        zc = int(np.rint(self.z_contact * self.spm[2] * self.sign[2]))
        return np.array([self._steps[0], self._steps[1], zc], dtype=np.int64)

    def zero_xy(self):
        """Make the current x, y the block centre (x = y = 0)."""
        self._steps[:2] = 0

    def settle(self):
        if self.settle_s > 0:
            time.sleep(self.settle_s)

    # ---- camera
    def frame(self):
        for _ in range(self.flush):
            self.cam.grab() if hasattr(self.cam, "grab") else self.cam.read()
        ok, f = self.cam.read()
        if not ok or f is None:
            raise RuntimeError("the TacTip camera returned no frame")
        return f

    def set_rest(self, n_frames=20, min_pins=20, verbose=True):
        """Record the untouched pins (sensor in free space) and start tracking them. Fails if
        fewer than min_pins are found (the wrong camera, or detector settings that don't
        suit the image): the frame is saved to rest_failed.png to look at."""
        frames = [self.frame() for _ in range(n_frames)]
        try:
            rest = self.detector.rest(frames)
        except RuntimeError:
            rest = np.zeros((0, 2))
        if len(rest) < min_pins:
            try:
                import cv2
                cv2.imwrite("rest_failed.png", frames[-1])
            except Exception:
                pass
            raise RuntimeError(
                f"only {len(rest)} TacTip pins found (expected dozens; saved the image to "
                "rest_failed.png). If it isn't the TacTip's view: " + CAMERA_HINT + ". If it is: "
                "adjust --detector, e.g. '{\"invert\": true}' for dark pins, or block/offset")
        self.tracker = MarkerTracker(rest)
        if self.plate:
            self.force_zero = float(np.mean([self._plate_raw() for _ in range(5)]))
        if verbose:
            print(f"[real] {len(rest)} pins at rest (pitch "
                  f"{self.tracker.max_jump * 2:.1f} px)")
        return rest

    @property
    def rest(self):
        return None if self.tracker is None else self.tracker.rest

    def pins(self):
        if self.tracker is None:
            raise RuntimeError("set_rest() first, with the sensor in free space")
        return self.tracker.update(self.detector(self.frame()))

    def _plate_raw(self):
        return float(np.sum(self.c.getPressure()))

    def read(self):
        pos, ok = self.pins()
        force = self._plate_raw() - self.force_zero if self.plate else None
        return {"pins": pos, "ok": ok, "force_raw": force}

    def signal(self):
        pos, ok = self.pins()
        return contact_signal((pos - self.tracker.rest)[ok])

    def close(self):
        if hasattr(self.cam, "release"):
            self.cam.release()


# ============================================================================ set-up help
def measure_steps_per_mm(rig, axis, steps=-800, ask=None):
    """Move one axis of a RealRig by `steps` (negative: away from the limit switches) and ask
    how far the sensor moved relative to the block (mm, calipers / a rule; negative if it
    went against the simulator's + axis, z + up). Moves back afterwards.
    Returns (steps per mm, sign)."""
    ask = ask or input
    i = "xyz".index(axis)
    move = [0, 0, 0]
    move[i] = int(steps)
    rig.move_steps(*move)
    mm = float(ask(f"{axis}: moved {steps} steps. How far did the sensor move relative to the "
                   f"block, in mm (negative if against the simulator's +{axis}"
                   f"{', i.e. down' if axis == 'z' else ''})? "))
    move[i] = -int(steps)
    rig.move_steps(*move)
    if mm == 0:
        raise ValueError("a distance of 0: move more steps")
    return abs(steps / mm), int(np.sign(mm) * np.sign(steps))


SETUP_KEYS = ("steps_per_mm", "axis_sign", "home_offset", "home_buttons")


def save_rig_setup_dict(path, d):
    """Write rig_setup.json from a dict of RealRig keyword arguments (PoseMap.rig_setup())."""
    d = dict(d, created=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return path


def save_rig_setup(path, rig, preset=None):
    """The real rig's constants (RealRig keyword arguments) as JSON."""
    d = dict(steps_per_mm=rig.spm.tolist(), axis_sign=rig.sign.astype(int).tolist(),
             home_offset=None if rig.home_offset is None else rig.home_offset.tolist(),
             home_buttons=list(rig.home_buttons), start_height_mm=rig.start_height_mm,
             preset=None if preset is None else list(preset),
             created=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(path, "w") as f:
        json.dump(d, f, indent=1)
    return path


def load_rig_setup(path):
    """RealRig keyword arguments from rig_setup.json (real-setup writes it)."""
    with open(path) as f:
        d = json.load(f)
    return {k: d[k] for k in SETUP_KEYS if d.get(k) is not None}


def jog(rig, ask=None, show=None):
    """Move the real rig by hand: 'x 2', 'y -0.5', 'z -1' (mm, relative), 's z -40' (steps),
    'show' (display the camera if show= a callable), 'done'. real-setup uses it once, to put
    the sensor over the block centre a few mm above it; after that, reset() gets there."""
    ask = ask or input
    print("jog: x|y|z <mm>, s x|y|z <steps>, show, done")
    while True:
        cmd = ask("jog> ").strip().split()
        if not cmd:
            continue
        if cmd[0] == "done":
            return
        if cmd[0] == "show" and show is not None:
            show()
            continue
        try:
            if cmd[0] == "s":
                d = [0, 0, 0]
                d["xyz".index(cmd[1])] = int(cmd[2])
                rig.move_steps(*d)
            else:
                x, y, z = rig._raw()
                v = float(cmd[1])
                rig._move_raw(x + v * (cmd[0] == "x"), y + v * (cmd[0] == "y"), z + v * (cmd[0] == "z"))
        except (ValueError, IndexError):
            print("?  e.g. 'z -0.5', 's x 100', 'done'")
