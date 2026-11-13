"""
Drive the simulated rig and the real one with the keyboard, side by side.

pose_calibrate()  Both rigs reset (the simulated one to its start pose, the real one homes
                  and moves a preset towards the block). Two windows:
                    - the MuJoCo viewer: keys there move the simulated rig;
                    - "real rig" (the TacTip camera, or a status panel without one): keys
                      there move the real rig (Tab: the simulated one, or both together).
                  Put both in the same place relative to their blocks, press Enter, and
                  repeat at a few poses (different x, y and z: e.g. over the centre high and
                  low, and a few mm to each side). Every pose refits the map (PoseMap) and
                  saves it, with the rig_setup.json it implies.
follow()          Load the map; you drive the simulated rig, the real one follows in the
                  background.

Keys
  both windows    up / down      +x / -x          (the simulator's axes; on the real rig, the
                  right / left   +y / -y           board's until a map is fitted, then the
                  PgUp / PgDn    z up / down       simulator's)
                  Enter          record this pose (both rigs must have stopped)
  "real rig"      w s a d r f    as the arrows and PgUp / PgDn
  window only     1 .. 5         step size
                  Tab            what the keys move: real -> sim -> both
                  g              real rig to where the map puts the simulated pose
                  n              simulated rig to where the map puts the real pose
                  u / Backspace  forget the last pose
                  h              home the real rig     z  reset the simulated rig
                  q / Esc        save and quit
The MuJoCo viewer keeps its own letter, digit, Tab and Backspace shortcuts (rendering
toggles, reset), so those are only read in the "real rig" window.

On macOS the MuJoCo viewer needs `mjpython` instead of `python`.
"""
import os
import queue
import threading
import time

import numpy as np

from .pose_map import PoseMap

# MuJoCo viewer (GLFW) key codes
GLFW = {265: "up", 264: "down", 262: "right", 263: "left", 266: "pgup", 267: "pgdn",
        257: "enter", 335: "enter"}
# OpenCV waitKeyEx codes: Windows, Linux (GTK), macOS
CV = {2490368: "up", 2621440: "down", 2555904: "right", 2424832: "left", 2162688: "pgup",
      2228224: "pgdn",
      65362: "up", 65364: "down", 65363: "right", 65361: "left", 65365: "pgup", 65366: "pgdn",
      63232: "up", 63233: "down", 63235: "right", 63234: "left", 63276: "pgup", 63277: "pgdn",
      13: "enter", 10: "enter", 8: "undo", 127: "undo", 9: "tab", 27: "quit"}
CV_LETTERS = {"w": "up", "s": "down", "d": "right", "a": "left", "r": "pgup", "f": "pgdn",
              "g": "go", "n": "sim_to_real", "u": "undo", "h": "home", "z": "sim_reset",
              "q": "quit", "1": "size1", "2": "size2", "3": "size3", "4": "size4", "5": "size5"}
MOVES = {"up": (0, 1), "down": (0, -1), "right": (1, 1), "left": (1, -1), "pgup": (2, 1),
         "pgdn": (2, -1)}

SIM_STEP_MM = (0.05, 0.2, 1.0, 5.0, 20.0)
REAL_STEPS = (1, 5, 25, 125, 500)
WINDOW = "real rig"


def viewer_key(code):
    return GLFW.get(int(code))


def cv_key(code):
    if code is None or code < 0:
        return None
    if code in CV:
        return CV[code]
    c = chr(code & 0xFF).lower() if (code & 0xFF) < 128 else ""
    return CV_LETTERS.get(c)


# ============================================================================ real rig worker
class NoCamera:
    """Stands in for the TacTip camera: a blank frame (the window still takes keys)."""

    def read(self):
        return True, np.zeros((360, 480, 3), np.uint8)

    def grab(self):
        return True

    def release(self):
        pass


class StepperWorker:
    """Moves the real rig towards a target (board steps from home) in the background, in
    short chunks, so a new target takes over within one chunk and the keyboard and the
    simulation never wait for the motors. Targets are clamped to <= 0 (home is the top of
    every axis) and to `bounds` ((lo, hi) steps per axis) if given."""

    def __init__(self, rig, chunk=50, bounds=None):
        self.rig = rig
        self.chunk = int(chunk)
        self.bounds = None if bounds is None else (np.asarray(bounds[0]), np.asarray(bounds[1]))
        self._lock = threading.Lock()
        self._target = None
        self._home = False
        self._stop = False
        self.busy = False
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def pos(self):
        return self.rig._steps.copy()

    def clamp(self, steps):
        s = np.minimum(np.rint(np.asarray(steps, float)), 0).astype(np.int64)
        if self.bounds is not None:
            s = np.clip(s, self.bounds[0], self.bounds[1]).astype(np.int64)
        return s

    @property
    def target(self):
        with self._lock:
            return None if self._target is None else self._target.copy()

    def set_target(self, steps):
        with self._lock:
            self._target = self.clamp(steps)

    def nudge(self, axis, steps):
        with self._lock:
            base = self._target.copy() if self._target is not None else self.pos
            base[axis] += int(steps)
            self._target = self.clamp(base)

    def request_home(self):
        with self._lock:
            self._home, self._target = True, None

    def idle(self):
        with self._lock:
            t, h = self._target, self._home
        return not h and not self.busy and (t is None or np.array_equal(t, self.pos))

    def wait(self, timeout=120.0):
        t0 = time.time()
        while not self.idle():
            if self.error:
                raise self.error
            if time.time() - t0 > timeout:
                raise TimeoutError("the real rig didn't reach its target")
            time.sleep(0.02)

    def stop(self):
        self._stop = True
        self.thread.join(timeout=5)

    def _run(self):
        while not self._stop:
            try:
                with self._lock:
                    home, tgt = self._home, self._target
                if home:
                    self.busy = True
                    self.rig.home(verbose=False)
                    with self._lock:
                        self._home = False
                    self.busy = False
                    continue
                if tgt is None or np.array_equal(tgt, self.rig._steps):
                    self.busy = False
                    time.sleep(0.01)
                    continue
                self.busy = True
                d = np.clip(tgt - self.rig._steps, -self.chunk, self.chunk).astype(np.int64)
                self.rig.c.move(int(d[0]), int(d[1]), int(d[2]), 0)
                self.rig._steps += d
            except Exception as err:                       # surfaced in the main loop
                self.error = err
                self.busy = False
                return


# ============================================================================ the windows
def _panel(frame, lines, width=640):
    import cv2
    f = frame.copy()
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    scale = width / f.shape[1]
    f = cv2.resize(f, (width, int(f.shape[0] * scale)))
    pad = np.zeros((18 * len(lines) + 10, width, 3), np.uint8)
    for i, (text, colour) in enumerate(lines):
        cv2.putText(pad, text, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1,
                    cv2.LINE_AA)
    return np.vstack([f, pad])


class _Session:
    """What both tools share: the simulated rig in a MuJoCo viewer, the real rig behind a
    StepperWorker, a keyboard queue fed by both windows."""

    def __init__(self, sim, real, worker, pose_map=None):
        self.sim, self.real, self.worker, self.pm = sim, real, worker, pose_map
        self.q = queue.Queue()
        self.mode = "real"
        self.size = 2
        self.msg = ""
        self.quit = False
        self.sim_target = np.array(sim._raw())

    # ---- keys
    def on_viewer_key(self, code):
        name = viewer_key(code)
        if name:
            self.q.put(("viewer", name))

    def poll_window(self):
        import cv2
        name = cv_key(cv2.waitKeyEx(1))
        if name:
            self.q.put(("window", name))

    def say(self, text):
        self.msg = text
        print(f"[teleop] {text}")

    def fitted(self):
        return self.pm is not None and self.pm.A is not None

    def real_dir(self, axis):
        """Real-rig key direction: the simulator's axis once a map says which way it is."""
        if self.fitted():
            return 1 if self.pm.A[axis, axis] >= 0 else -1
        return 1

    def move(self, who, axis, sign):
        if who in ("sim", "both"):
            lo, hi = self.sim.limits_mm()
            self.sim_target[axis] = np.clip(self.sim_target[axis] + sign * SIM_STEP_MM[self.size],
                                            lo[axis], hi[axis])
            if who == "both":
                if self.fitted():
                    self.worker.set_target(self.pm.to_steps(self.sim_target))
                else:
                    self.say("both: no map yet (record poses first); moved the sim only")
        if who == "real":
            self.worker.nudge(axis, sign * self.real_dir(axis) * REAL_STEPS[self.size])

    def handle_common(self, src, name):
        """Moves and sizes; returns True if handled."""
        if name in MOVES:
            axis, sign = MOVES[name]
            who = "sim" if src == "viewer" and self.mode == "real" else self.mode
            self.move(who, axis, sign)
            return True
        if name.startswith("size"):
            self.size = int(name[-1]) - 1
            self.say(f"step: sim {SIM_STEP_MM[self.size]:g} mm, real {REAL_STEPS[self.size]} steps")
            return True
        if name == "tab":
            self.mode = {"real": "sim", "sim": "both", "both": "real"}[self.mode]
            self.say(f"window keys move: {self.mode}")
            return True
        if name == "home":
            self.worker.request_home()
            self.say("homing the real rig")
            return True
        if name == "sim_reset":
            self.sim.reset(verbose=False)
            self.sim_target = np.array(self.sim._raw())
            self.say("simulated rig reset")
            return True
        if name == "quit":
            self.quit = True
            return True
        if name == "go":
            if self.fitted():
                self.worker.set_target(self.pm.to_steps(self.sim_target))
                self.say("real rig -> the simulated pose")
            else:
                self.say("g: no map yet")
            return True
        if name == "sim_to_real":
            if self.fitted():
                lo, hi = self.sim.limits_mm()
                self.sim_target = np.clip(self.pm.to_sim(self.worker.pos), lo, hi)
                self.say("simulated rig -> the real pose")
            else:
                self.say("n: no map yet")
            return True
        return False

    # ---- frame
    def tick(self, viewer):
        if self.worker.error:
            raise RuntimeError(f"the real rig stopped: {self.worker.error!r}") from self.worker.error
        with viewer.lock():
            self.sim.step_toward(self.sim_target)
        self.sim.e.sync_renderer(viewer)
        viewer.sync()

    def status(self, extra=()):
        sim = np.array(self.sim._raw())
        real = self.worker.pos
        tgt = self.worker.target
        moving = "" if self.worker.idle() else "  (moving)"
        lines = [(f"keys move: {self.mode}  |  step: sim {SIM_STEP_MM[self.size]:g} mm, "
                  f"real {REAL_STEPS[self.size]} steps", (255, 255, 255)),
                 (f"sim  x {sim[0]:+7.2f}  y {sim[1]:+7.2f}  z {sim[2]:+7.2f} mm", (120, 220, 255)),
                 (f"real x {real[0]:+6d}  y {real[1]:+6d}  z {real[2]:+6d} steps from home{moving}",
                  (120, 255, 120))]
        if tgt is not None and not np.array_equal(tgt, real):
            lines.append((f"     -> {tgt[0]:+6d}     {tgt[1]:+6d}     {tgt[2]:+6d}", (120, 255, 120)))
        if self.fitted():
            m = self.pm.to_sim(real)
            lines.append((f"real pose in sim mm: x {m[0]:+.2f}  y {m[1]:+.2f}  z {m[2]:+.2f}",
                          (200, 200, 200)))
        lines += list(extra)
        if self.msg:
            lines.append((self.msg, (0, 220, 255)))
        return lines

    def show(self, extra=()):
        import cv2
        try:
            ok, frame = self.real.cam.read()
        except Exception:
            ok, frame = False, None
        if not ok or frame is None:
            frame = np.zeros((360, 480, 3), np.uint8)
        cv2.imshow(WINDOW, _panel(frame, self.status(extra)))


def _sleep_rest(t0, dt):
    time.sleep(max(0.0, dt - (time.time() - t0)))


# ============================================================================ pose calibration
def pose_calibrate(sim, real, out_path, kind="axis", preset=(-1900, -6400), resume=False,
                   setup_out=None, home_on_exit=True, chunk=50, known_spm=None, known_sign=None):
    """Interactive: drive both rigs to matching poses, Enter to record each. Saves
    `out_path` (PoseMap) after every pose and, once it fits, the rig_setup.json it implies
    (setup_out; None: next to out_path). known_spm / known_sign (e.g. from an earlier
    rig_setup.json) let an axis you don't vary take a known scale. See the module docstring
    for the keys."""
    import cv2
    import mujoco.viewer

    real.home_offset = None                      # positions count from home here
    pm = PoseMap.load(out_path) if resume and os.path.exists(out_path) else \
        PoseMap(known_spm=known_spm, known_sign=known_sign)
    if pm.points and pm.can_fit(kind):
        pm.fit(kind)
    setup_out = setup_out or os.path.join(os.path.dirname(os.path.abspath(out_path)), "rig_setup.json")

    print(__doc__[__doc__.index("Keys"):])
    print("[teleop] homing the real rig (z first), then the preset towards the block ...")
    real.home(verbose=False)
    real.move_steps(int(preset[0]), int(preset[1]), 0)
    sim.reset(verbose=False)
    worker = StepperWorker(real, chunk=chunk)
    s = _Session(sim, real, worker, pm)
    s.say("both rigs reset. Match the poses, Enter to record. "
          f"{len(pm.points)} poses so far")

    def save():
        """The poses, and (when they fit) the rig_setup.json they imply: always together."""
        pm.save(out_path)
        if pm.A is not None:
            from .rigs import save_rig_setup_dict
            save_rig_setup_dict(setup_out, pm.rig_setup(home_buttons=real.home_buttons))

    def record():
        if not worker.idle() or np.linalg.norm(np.array(sim._raw()) - s.sim_target) > 1e-3:
            s.say("wait until both rigs have stopped, then Enter")
            return
        pm.add(sim._raw(), worker.pos)
        try:
            pm.fit(kind)
            fit = True
        except ValueError as err:
            pm.A = None
            fit = False
            s.say(f"pose {len(pm.points)} recorded; {err}")
        save()
        if fit:
            print(pm.summary())
            r = pm.residual_mm[-1] if pm.residual_mm is not None else float("nan")
            s.say(f"pose {len(pm.points)} recorded (fits to {r:.2f} mm); saved {os.path.basename(out_path)}")

    def undo():
        p = pm.undo()
        if p is None:
            s.say("no poses to forget")
            return
        try:
            pm.fit(kind)
        except ValueError:
            pm.A = None
        save()
        s.say(f"forgot pose {len(pm.points) + 1}; {len(pm.points)} left")

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        with mujoco.viewer.launch_passive(sim.e.model, sim.e.data, key_callback=s.on_viewer_key) as viewer:
            while viewer.is_running() and not s.quit:
                t0 = time.time()
                s.poll_window()
                while not s.q.empty():
                    src, name = s.q.get()
                    if s.handle_common(src, name):
                        continue
                    if name == "enter":
                        record()
                    elif name == "undo":
                        undo()
                s.tick(viewer)
                s.show([(f"poses: {len(pm.points)}" + (f"  ({pm.kind} map)" if s.fitted() else ""),
                         (255, 200, 120))])
                _sleep_rest(t0, sim.e.dt)
    finally:
        cv2.destroyWindow(WINDOW)
        save()
        if home_on_exit and worker.error is None:
            worker.request_home()
            try:
                worker.wait()
            except Exception as err:
                print(f"[teleop] couldn't home on exit: {err!r}")
        worker.stop()
    print(f"[teleop] saved {out_path}" + (f" and {setup_out}" if pm.A is not None else ""))
    print(pm.summary())
    return pm


# ============================================================================ follow
def follow(sim, real, pose_map, margin_mm=20.0, home_on_exit=True, chunk=50, window=True):
    """You drive the simulated rig (MuJoCo viewer, or the "real rig" window); the real rig
    follows through the pose map, in the background. The real rig is kept within margin_mm
    (x, y) of the poses the map was recorded at, and above home."""
    import mujoco.viewer
    if window:
        import cv2

    if pose_map.A is None:
        pose_map.fit(pose_map.kind or "axis")
    S, T = pose_map.arrays()
    spm, _ = pose_map.steps_per_mm()
    pad = np.array([margin_mm * spm[0], margin_mm * spm[1], 0.0])
    # z: whatever the simulated rig can reach (its own limits), mapped
    lo_mm, hi_mm = sim.limits_mm()
    zz = pose_map.to_steps(np.array([[0, 0, lo_mm[2]], [0, 0, hi_mm[2]]]))[:, 2]
    lo = np.r_[T[:, 0].min() - pad[0], T[:, 1].min() - pad[1], zz.min()]
    hi = np.r_[T[:, 0].max() + pad[0], T[:, 1].max() + pad[1], zz.max()]

    real.home_offset = None                      # positions count from home here
    print("[follow] homing the real rig (z first) ...")
    real.home(verbose=False)
    sim.reset(verbose=False)
    worker = StepperWorker(real, chunk=chunk, bounds=(lo, hi))
    s = _Session(sim, real, worker, pose_map)
    s.mode = "sim"
    s.say("drive the simulated rig: arrows, PgUp / PgDn (1-5 step size in this window); "
          "the real rig follows")
    if window:
        cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    try:
        with mujoco.viewer.launch_passive(sim.e.model, sim.e.data, key_callback=s.on_viewer_key) as viewer:
            while viewer.is_running() and not s.quit:
                t0 = time.time()
                if window:
                    s.poll_window()
                while not s.q.empty():
                    src, name = s.q.get()
                    if name in MOVES:
                        s.move("sim", *MOVES[name])
                    elif name in ("tab", "go", "sim_to_real"):
                        pass                                  # always sim -> real here
                    else:
                        s.handle_common(src, name)
                s.tick(viewer)
                worker.set_target(pose_map.to_steps(sim._raw()))   # where the sim is now
                if window:
                    s.show()
                _sleep_rest(t0, sim.e.dt)
    finally:
        if window:
            cv2.destroyWindow(WINDOW)
        if home_on_exit and worker.error is None:
            worker.request_home()
            try:
                worker.wait()
            except Exception as err:
                print(f"[follow] couldn't home on exit: {err!r}")
        worker.stop()
