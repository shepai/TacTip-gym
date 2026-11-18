"""
Calibrate the real rig (Rig-controller, MicroPython) and TacTip against the simulated TacRig,
then run the same experiment on either.

Matching the rigs' poses by hand (keyboard, both rigs live), and the follow demo:

    # both rigs reset; drive each to the same pose (MuJoCo window: sim, "real rig" window:
    # real), Enter to record; a few poses. Writes pose_calibration.json + rig_setup.json
    python -m tactip_mujoco_gym.sim2real pose-calibrate --port COM19 --camera 1 \\
        --out calib/pose_calibration.json
    # drive the simulation, the real rig follows
    python -m tactip_mujoco_gym.sim2real follow --pose-calibration calib/pose_calibration.json \\
        --port COM19 --camera 1
    (keys: tactip_mujoco_gym/sim2real/teleop.py; on macOS run these with mjpython)

Rig-controller is used as an installed package (pip install -e path/to/Rig-controller) with
its own board listener. --listener runs a different listener file; --rig-code is only for an
old checkout that isn't installed.

Both rigs start every run from the same pose, 10 mm above the block centre: the simulated
rig's reset, and on the real rig homing against the limit switches (z first, then x, y)
followed by a fixed move measured once by real-setup. Every run also ends there.

    # 0. once (and again if the rig or the holder moves): measure steps/mm and the move from
    #    home to the block, with the FLAT block in the holder (interactive)
    python -m tactip_mujoco_gym.sim2real real-setup --port COM19 --camera 1 \\
        --measure-steps --out-dir calib

    # 1. the simulated presses, at a few skin stiffnesses (no hardware)
    python -m tactip_mujoco_gym.sim2real sim-capture --stiffness 0.5 1 2 4 --out-dir calib

    # 2. the same presses on the real rig: homes, goes to the start pose, presses, homes
    python -m tactip_mujoco_gym.sim2real real-capture --port COM19 --camera 1 \\
        --out-dir calib

    # 3. fit: layout, skin stiffness, per-marker maps, force; writes rig_calibration.json
    python -m tactip_mujoco_gym.sim2real fit --out-dir calib

    # 4. an experiment on both rigs, data in the simulated "tactile" format
    python -m tactip_mujoco_gym.sim2real experiment --rig sim  --calibration calib/rig_calibration.json --out sim.npz
    python -m tactip_mujoco_gym.sim2real experiment --rig real --calibration calib/rig_calibration.json \\
        --port COM19 --camera 1 --out real.npz
    python -m tactip_mujoco_gym.sim2real compare sim.npz real.npz

Steps 1 and 2 must use the same protocol options (--depths, --shears, --points, --span,
--repeats); the defaults match: presses to 0.5-2 mm, each also slid 0.5 and 1 mm along +-x
and +-y (the slides tell which way the camera is turned), twice. Both rigs press a flat
block for calibrating (the simulated one is an STL box the size of a dataset block; print
one, 29.3 x 30 x 7 mm, for the real holder).

The learned image mapping (sim2real/mapping.py), from sessions recorded with rig_dataset
(Rig-controller repo): an autoencoder for the simulated TacTip, one for the real TacTip, and
maps between their latent spaces. Then "real_tactip" in the gym is the simulated TacTip
mapped to what the real one would see:

    python -m tactip_mujoco_gym.sim2real train-mapping data/ --out models/sim2real
    python -m tactip_mujoco_gym.sim2real train-mapping data/ --backend linear --out models/lin
    python -m tactip_mujoco_gym.sim2real mapping-preview models/sim2real data/session_new
    gym.make("TacRig-v0", sensors={"real_tactip": {"model": "models/sim2real"}},
             obs_modes=["real_tactip"])
"""
import argparse
import glob
import json
import os

import numpy as np

from .calibration import RealToSim, RigCalibration, fit_calibration
from .markers import MarkerDetector
from .rigs import (RealRig, SimRig, calibration_probes, connect_controller, flat_block_stl,
                   grid_probes, jog, load_rig_setup, measure_steps_per_mm, open_camera,
                   run_probes, save_rig_setup)


# ============================================================================ helpers
def _calib_protocol_args(ap):
    ap.add_argument("--depths", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0],
                    help="press depths, mm")
    ap.add_argument("--shears", type=float, nargs="+", default=[0.5, 1.0],
                    help="sideways slides after pressing, mm (each along +-x and +-y)")
    ap.add_argument("--points", type=int, default=1, help="grid points per side (1: the centre)")
    ap.add_argument("--span", type=float, default=0.0, help="grid width, mm")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--clear", type=float, default=1.5, help="height between presses, mm")


def _calib_probes(a):
    return calibration_probes(depths_mm=a.depths, shears_mm=a.shears, points=a.points,
                              span_mm=a.span, repeats=a.repeats)


def _grid_protocol_args(ap):
    ap.add_argument("--points", type=int, default=5, help="grid points per side")
    ap.add_argument("--span", type=float, default=5.0, help="grid width, mm")
    ap.add_argument("--depths", type=float, nargs="+", default=[1.0], help="press depths, mm")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--clear", type=float, default=1.5, help="height between presses, mm")


def _grid_probes(a):
    return grid_probes(points=a.points, span_mm=a.span, depths_mm=a.depths, repeats=a.repeats)


def _real_args(ap):
    ap.add_argument("--port", required=True, help="the Pico's serial port, e.g. COM19 or /dev/ttyACM0")
    ap.add_argument("--listener", default=None,
                    help="a board listener to use instead of Rig-controller's packaged one")
    ap.add_argument("--rig-code", default=None,
                    help="only if Rig-controller isn't pip-installed: its Code/ folder")
    ap.add_argument("--camera", type=int, default=0, help="the TacTip's camera index")
    ap.add_argument("--width", type=int, default=None)
    ap.add_argument("--height", type=int, default=None)
    ap.add_argument("--detector", default=None, help="MarkerDetector settings as JSON")
    ap.add_argument("--plate", action="store_true", help="read the base's force plate")
    ap.add_argument("--settle", type=float, default=0.3, help="wait after moves, s")


def _say(msg):
    input(f"\n>>> {msg}\n    [Enter when ready] ")


def _save_preview(rig, path):
    """The current frame with the detected rest pins drawn and numbered."""
    try:
        import cv2
    except ImportError:
        return None
    f = rig.frame().copy()
    if f.ndim == 2:
        f = cv2.cvtColor(f, cv2.COLOR_GRAY2BGR)
    for i, (x, y) in enumerate(rig.rest):
        cv2.circle(f, (int(round(x)), int(round(y))), 4, (0, 0, 255), 1)
        cv2.putText(f, str(i), (int(x) + 4, int(y) - 4), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)
    cv2.imwrite(path, f)
    return path


def _detector(a, out_dir):
    """--detector if given, else the settings tuned in an earlier run (detector.json),
    else the defaults (tuned automatically if they find too few pins)."""
    if getattr(a, "detector", None):
        return MarkerDetector(**json.loads(a.detector))
    path = os.path.join(out_dir, "detector.json")
    if os.path.exists(path):
        with open(path) as f:
            return MarkerDetector(**json.load(f))
    return MarkerDetector()


def _setup_path(a, near=None):
    if getattr(a, "setup", None):
        return a.setup
    folder = near or getattr(a, "out_dir", "calib")
    return os.path.join(folder, "rig_setup.json")


def _make_real(a, detector, setup_path):
    if not os.path.exists(setup_path):
        raise SystemExit(f"no {setup_path}: run real-setup first (it measures the move from "
                         "home to the block)")
    setup = load_rig_setup(setup_path)
    if "home_offset" not in setup:
        raise SystemExit(f"{setup_path} has no home_offset: run real-setup again")
    c = connect_controller(a.port, a.listener, a.rig_code)
    cam = open_camera(a.camera, a.width, a.height)
    return RealRig(c, cam, detector=detector, plate=a.plate, settle_s=a.settle, **setup)


def _real_start(rig, out_dir):
    """Home, the start pose, a rest image there (the sensor is in free air), contact.
    Returns the rest pins."""
    _say("Put the FLAT block in the holder. The rig will home (z up first, then x and y) "
         "and go to the start pose, 10 mm above the block centre.")
    rig.reset()
    rest = rig.set_rest()
    if rig.detector_tuned:
        path = os.path.join(out_dir, "detector.json")
        with open(path, "w") as f:
            json.dump(rig.detector.to_dict(), f, indent=1)
        print(f"    saved the tuned detector settings to {path} (used from now on)")
    p = _save_preview(rig, os.path.join(out_dir, "rest_pins.png"))
    if p:
        print(f"    detected pins drawn on {p}: check every pin is found once "
              "(else adjust --detector)")
    print("    lowering until the TacTip feels the block ...")
    rig.find_contact()
    return rest


# ============================================================================ commands
def cmd_sim_capture(a):
    os.makedirs(a.out_dir, exist_ok=True)
    probes = _calib_probes(a)
    for k in a.stiffness:
        print(f"\n=== simulated rig, sensor_stiffness {k} ===")
        rig = SimRig(textures=[flat_block_stl()], sensor_stiffness=k, sensor_xml=a.sensor_xml)
        rig.reset()                                  # the start pose, as on the real rig
        rig.find_contact()
        data = run_probes(rig, probes, clear_mm=a.clear)
        rig.reset()
        out = os.path.join(a.out_dir, f"sim_k{k:g}.npz")
        np.savez_compressed(
            out, probes=data["probes"], markers=data["markers"], lateral=data["lateral"],
            force=data["force"], layout=rig.layout, frame_R=rig._frame_R,
            sensor_stiffness=k, marker_scale=rig.marker_scale, marker_clip=rig.marker_clip,
            velocity=bool(getattr(rig.readout, "velocity", True)),
            env=json.dumps({"sensor_xml": a.sensor_xml} if a.sensor_xml else {}),
            z_contact=rig.z_contact,
            contact_z=rig.contact_trace["z"], contact_signal=rig.contact_trace["signal"],
            n_markers=rig.n)
        print(f"saved {out}: {len(probes)} presses, {rig.n} markers, "
              f"force {data['force'].min():.2f}..{data['force'].max():.2f} N")
        rig.close()


def cmd_cameras(a):
    from .rigs import list_cameras
    found = list_cameras(a.max_index, a.out_dir)
    if not found:
        print("no camera gives pictures (is the TacTip plugged in? on Linux, are you in the "
              "'video' group?)")
    for i, w, h in found:
        print(f"camera {i}: {w} x {h}" + (f"  -> {a.out_dir}/camera_{i}.png" if a.out_dir else ""))
    if found:
        print("use the TacTip's index as --camera")


def cmd_real_setup(a):
    if not a.no_sim:
        # matched against the simulator, side by side (pose-calibrate)
        print("[real-setup] matching the real rig to the simulated one: the MuJoCo viewer and "
              "the 'real rig' window open. (--no-sim: the old text-only set-up.)")
        if a.measure_steps or a.steps_per_mm:
            print("[real-setup] steps/mm come from the poses you match; --measure-steps / "
                  "--steps-per-mm aren't needed")
        a.out = os.path.join(a.out_dir, "pose_calibration.json")
        a.setup_out = a.setup or os.path.join(a.out_dir, "rig_setup.json")
        a.setup = None
        return cmd_pose_calibrate(a)
    os.makedirs(a.out_dir, exist_ok=True)
    detector = _detector(a, a.out_dir)
    if a.steps_per_mm is None and not a.measure_steps:
        raise SystemExit("give --steps-per-mm X Y Z, or --measure-steps")
    c = connect_controller(a.port, a.listener, a.rig_code)
    cam = open_camera(a.camera, a.width, a.height)
    spm = a.steps_per_mm or [1.0, 1.0, 1.0]
    rig = RealRig(c, cam, spm, a.axis_sign, detector=detector, plate=a.plate, settle_s=a.settle)

    _say("Put the FLAT block in the holder and make sure nothing is in the way: the rig will "
         "home (z up to its switch first, then x and y).")
    rig.home()
    print(f"    moving the preset {a.preset[0]}, {a.preset[1]} steps (x, y) towards the block ...")
    rig.move_steps(a.preset[0], a.preset[1], 0)
    rig.set_rest()                                       # high up: free air

    if a.measure_steps:
        _say("Measuring steps per mm: each axis moves "
             f"{abs(a.measure_n)} steps and back. Have a rule or calipers ready.")
        spm, sign = [], []
        for ax in "xyz":
            k, g = measure_steps_per_mm(rig, ax, -abs(a.measure_n))
            spm.append(k)
            sign.append(g)
            print(f"    {ax}: {k:.3f} steps/mm, sign {g:+d}")
        rig.spm = np.asarray(spm, float)
        rig.sign = np.asarray(sign, float)

    _say("Now jog the sensor over the centre of the block and down to 2-4 mm above it, NOT "
         "touching (x/y/z in mm, e.g. 'z -20'; 'done' when there). This is the only time.")
    jog(rig)
    print("    lowering until the TacTip feels the block ...")
    rig.find_contact()                                   # no offset yet: from here, down
    rig.home_offset = rig.contact_offset()
    out = _setup_path(a)
    save_rig_setup(out, rig, preset=a.preset)
    off_mm = rig.sign * rig.home_offset / rig.spm
    print(f"saved {out}: block centre at contact is {rig.home_offset.tolist()} steps from home "
          f"({off_mm[0]:.1f}, {off_mm[1]:.1f}, {off_mm[2]:.1f} mm)")

    # check: from home again, the start pose and contact where expected
    rig.z_contact = None
    rig.reset()
    _say("Check: the sensor should be over the block centre, 10 mm above it.")
    zc = rig.find_contact()
    print(f"    contact found {zc:+.2f} mm from where the setup expects it (should be ~0)")
    rig.reset()
    rig.close()


def cmd_real_capture(a):
    os.makedirs(a.out_dir, exist_ok=True)
    detector = _detector(a, a.out_dir)
    rig = _make_real(a, detector, _setup_path(a))
    rest = _real_start(rig, a.out_dir)
    probes = _calib_probes(a)
    print(f"\npressing {len(probes)} times ...")
    data = run_probes(rig, probes, clear_mm=a.clear)
    out = os.path.join(a.out_dir, "real.npz")
    extra = {"force_raw": data["force_raw"]} if "force_raw" in data else {}
    np.savez_compressed(out, probes=data["probes"], pins=data["pins"], ok=data["ok"], rest=rest,
                        steps_per_mm=rig.spm, axis_sign=rig.sign,
                        detector=json.dumps(detector.to_dict()), z_contact=rig.z_contact,
                        contact_z=rig.contact_trace["z"], contact_signal=rig.contact_trace["signal"],
                        **extra)
    print(f"saved {out}: {len(probes)} presses, {len(rest)} pins "
          f"(found in {data['ok'].mean():.0%} of readings)")
    rig.reset()
    rig.close()


def _load_npz(path):
    z = np.load(path, allow_pickle=False)
    d = {k: z[k] for k in z.files}
    for k in ("detector", "env"):
        if k in d:
            d[k] = json.loads(str(d[k]))
    for k in ("sensor_stiffness", "marker_scale", "marker_clip", "z_contact"):
        if k in d:
            d[k] = float(d[k])
    if "velocity" in d:
        d["velocity"] = bool(d["velocity"])
    return d


def _teleop_rigs(a, known):
    from .teleop import NoCamera
    textures = [flat_block_stl()] if a.texture in (None, "flat") else \
        [int(a.texture) if str(a.texture).isdigit() else a.texture]
    sim = SimRig(textures=textures, sensor_stiffness=a.stiffness)
    c = connect_controller(a.port, a.listener, a.rig_code)
    cam = open_camera(a.camera, a.width, a.height) if a.camera >= 0 else NoCamera()
    real = RealRig(c, cam, known.get("steps_per_mm", (1.0, 1.0, 1.0)),
                   known.get("axis_sign", (1, 1, 1)), settle_s=0.0, track=False,
                   home_buttons=known.get("home_buttons", (3, 2, 0)))
    return sim, real


def _scale(a, out):
    """Measured steps/mm (measure-steps), if there are any: --scale, or next to `out`."""
    path = getattr(a, "scale", None) or os.path.join(os.path.dirname(os.path.abspath(out)),
                                                     "steps_per_mm.json")
    if not os.path.exists(path):
        if getattr(a, "scale", None):
            raise SystemExit(f"no {path}")
        return None, None, None
    with open(path) as f:
        d = json.load(f)
    return d["steps_per_mm"], d.get("axis_sign"), path


def _model(a, spm):
    return a.model if a.model != "auto" else ("offset" if spm is not None else "axis")


def cmd_pose_calibrate(a):
    from .teleop import pose_calibrate
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    known = load_rig_setup(a.setup) if a.setup and os.path.exists(a.setup) else {}
    spm, sign, path = _scale(a, a.out)
    if spm is not None:
        print(f"[pose-calibrate] measured steps/mm from {path}: {spm} (offset model: one good pose "
              "is enough, more check it)")
    kind = _model(a, spm)
    sim, real = _teleop_rigs(a, known)
    pose_calibrate(sim, real, a.out, kind=kind, preset=a.preset, resume=a.resume,
                   setup_out=a.setup_out, home_on_exit=not a.stay,
                   known_spm=spm if spm is not None else known.get("steps_per_mm"),
                   known_sign=sign if spm is not None else known.get("axis_sign"))
    sim.close()
    real.close()


def cmd_measure_steps(a):
    """Each axis starts pressed against its push switch (home): mark there, it moves away
    by a set number of steps, type how far it went, it goes back to the switch."""
    from .teleop import NoCamera
    c = connect_controller(a.port, a.listener, a.rig_code)
    rig = RealRig(c, NoCamera(), (1.0, 1.0, 1.0), track=False, settle_s=0.0)
    _say("Measuring steps per mm, from the push switches. For each axis the rig first homes, so "
         "that axis is pressed against its switch: mark that spot, it moves away from the "
         "switch, then you type how far it went. Clear the bed first (z moves down).")
    spm = []
    for ax in "xyz":
        rig.home(verbose=False)                      # every axis back against its switch
        n = abs(a.steps if ax != "z" else a.z_steps)
        what = {"x": "the plate (x)", "y": "the main body (y)", "z": "the sensor carriage (z)"}[ax]
        _say(f"{ax}: {what} is now pressed against its push switch. Zero your calipers / mark "
             f"the rule here. It will then move {n} steps away from the switch.")
        k, _ = measure_steps_per_mm(rig, ax, -n, signed=False)
        spm.append(k)
        print(f"    {ax}: {k:.2f} steps/mm")
    out = a.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    with open(out, "w") as f:
        json.dump(dict(steps_per_mm=spm, axis_sign=None, steps_moved=a.steps,
                       z_steps_moved=a.z_steps, measured_from="the push switches (home)"), f, indent=1)
    print(f"saved {out}. Directions come from the poses. Next: real-setup --resume (or a fresh "
          "real-setup); it uses these automatically.")
    rig.home(verbose=False)


def cmd_pose_edit(a):
    from .pose_map import PoseMap
    from .rigs import save_rig_setup_dict
    pm = PoseMap.load(a.file)
    if a.drop:
        n = pm.drop(a.drop)
        print(f"dropped {n} pose(s): {sorted(set(a.drop))}")
    spm, sign, path = _scale(a, a.file)
    if spm is not None:
        pm.known_spm, pm.known_sign = np.asarray(spm, float), (None if sign is None else np.asarray(sign, float))
    kind = _model(a, spm) if a.model != "keep" else (pm.kind or "axis")
    try:
        pm.fit(kind)
    except ValueError as err:
        pm.A = None
        print(f"not enough poses for a {kind} map: {err}")
    pm.save(a.file)
    if pm.A is not None:
        setup = os.path.join(os.path.dirname(os.path.abspath(a.file)), "rig_setup.json")
        save_rig_setup_dict(setup, pm.rig_setup())
        print(f"saved {a.file} and {setup}")
    print(pm.summary())


def cmd_follow(a):
    from .pose_map import PoseMap
    from .teleop import follow
    pm = PoseMap.load(a.pose_calibration)
    sim, real = _teleop_rigs(a, {})
    follow(sim, real, pm, margin_mm=a.margin, home_on_exit=not a.stay, window=not a.no_window)
    sim.close()
    real.close()


def cmd_fit(a):
    real_path = a.real or os.path.join(a.out_dir, "real.npz")
    sim_paths = a.sim or sorted(glob.glob(os.path.join(a.out_dir, "sim_k*.npz")))
    if not sim_paths:
        raise SystemExit("no simulated captures (sim-capture first)")
    real = _load_npz(real_path)
    sims = [_load_npz(p) for p in sim_paths]
    calib = fit_calibration(real, sims, degree=a.degree)
    out = a.out or os.path.join(a.out_dir, "rig_calibration.json")
    calib.save(out)
    print(f"\nsaved {out}")


def tactile_vector(markers_mm, scale_mm, clip, velocity=True):
    """A press's "tactile" input: displacement and (static press) zero change."""
    d = np.asarray(markers_mm, float).ravel() / scale_mm
    parts = [d, np.zeros_like(d)] if velocity else [d]
    return np.clip(np.concatenate(parts), -clip, clip).astype(np.float32)


def cmd_experiment(a):
    calib = RigCalibration.load(a.calibration)
    textures = [int(t) if str(t).isdigit() else t for t in a.textures]
    probes = _grid_probes(a)
    scale_mm = calib.sim["marker_scale"] * 1000.0
    clip, vel = calib.sim["marker_clip"], calib.sim["velocity"]
    X, y, pos, F = [], [], [], []

    if a.rig == "sim":
        rig = SimRig(textures=[flat_block_stl()] + textures, **calib.sim_env_kwargs())
        rig.reset()
        rig.find_contact()                           # on the flat block, as on the real rig
        for i, t in enumerate(textures):
            rig.reset(texture=t)                     # new block, back to the start pose
            data = run_probes(rig, probes, clear_mm=a.clear)
            X += [tactile_vector(m, scale_mm, clip, vel) for m in data["markers"]]
            F += list(data["force"])
            y += [i] * len(probes)
            pos += list(probes)
        rig.reset()
        rig.close()
    else:
        detector = MarkerDetector(**calib.detector) if calib.detector else MarkerDetector()
        rig = _make_real(a, detector, _setup_path(a, near=os.path.dirname(os.path.abspath(a.calibration))))
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        rest = _real_start(rig, os.path.dirname(os.path.abspath(a.out)))
        to_sim = RealToSim(calib, rest)
        print(f"    {to_sim.n_seen} of {len(calib.real_rest)} calibrated pins found this session")
        for i, t in enumerate(textures):
            rig.home()                               # out of the way, and clears missed steps
            _say(f"Put texture {t} in the holder (same orientation as the simulator's).")
            rig.go_to_start()
            data = run_probes(rig, probes, clear_mm=a.clear)
            for p, ok in zip(data["pins"], data["ok"]):
                X.append(tactile_vector(to_sim.markers(p, ok), scale_mm, clip, vel))
            if "force_raw" in data and calib.force_map is not None:
                F += list(calib.force_newtons(data["force_raw"]))
            else:
                F += [np.nan] * len(probes)
            y += [i] * len(probes)
            pos += list(probes)
        rig.reset()
        rig.close()

    np.savez_compressed(a.out, X=np.stack(X), y=np.array(y), pos=np.array(pos),
                        force=np.array(F, float), textures=np.array([str(t) for t in textures]),
                        rig=a.rig, n_disp=3 * int(calib.sim["n_markers"]))
    print(f"saved {a.out}: {len(y)} presses x {np.stack(X).shape[1]} inputs ({a.rig} rig)")


def cmd_compare(a):
    s, r = np.load(a.sim), np.load(a.real)
    if s["X"].shape != r["X"].shape or not np.allclose(s["pos"], r["pos"]):
        raise SystemExit("the two runs differ in presses / inputs: same options on both")
    n = int(s["n_disp"])                      # the displacement part (not the change)
    S, R = s["X"][:, :n], r["X"][:, :n]

    def report(name, S, R):
        sse = float(((S - R) ** 2).sum())
        energy = float((S ** 2).sum())
        sst = float(((S - S.mean(axis=0)) ** 2).sum())
        line = (f"{name}: rmse {np.sqrt(sse / S.size):.4f}, error / signal "
                f"{np.sqrt(sse / energy) if energy > 0 else float('nan'):.1%}")
        if sst > 1e-9 * max(energy, 1e-12):     # R2 only means something if presses differ
            line += f", R2 across presses {1 - sse / sst:.3f}"
        print(line)

    report("marker displacement, real (mapped) vs sim", S, R)
    for t in np.unique(s["y"]):
        k = s["y"] == t
        report(f"  texture {s['textures'][t]}", S[k], R[k])
    if np.isfinite(r["force"]).any():
        ok = np.isfinite(r["force"])
        print(f"plate force: sim {s['force'][ok].mean():.2f} N, real {r['force'][ok].mean():.2f} N on average")


# ============================================================================ main
def _real_crop(a):
    if a.real_full_frame:
        return "full"
    return tuple(a.real_crop) if a.real_crop else None


def cmd_train_mapping(a):
    from .mapping import check_image_size, train_mapping
    from .paired import PairedData, crop_check
    if a.backend == "torch":
        check_image_size(a.real_size[0], a.real_size[1], "--real-size")
        if a.input == "image":
            check_image_size(a.sim_size, a.sim_size, "--sim-size")
    data = PairedData.load(a.sessions, sim_input=a.input, sim_size=a.sim_size,
                           real_size=tuple(a.real_size), real_crop=_real_crop(a), gray=a.gray,
                           real_bgr=not a.real_rgb, phases=a.phases, textures=a.textures,
                           every=a.every, max_samples=a.max_samples)
    os.makedirs(a.out, exist_ok=True)
    print(f"[mapping] the real crop: {crop_check(data, os.path.join(a.out, 'crop_check.png'))} "
          "(the magenta box should hold the pins; change it with --real-crop X Y W H)")
    settings = dict(latent_sim=a.latent_sim, latent_real=a.latent_real, epochs_ae=a.epochs[0],
                    epochs_map=a.epochs[1], epochs_joint=a.epochs[2], batch=a.batch, lr=a.lr,
                    patience=a.patience, reverse=not a.no_reverse, seed=a.seed)
    m = train_mapping(data, out=a.out, backend=a.backend, val=a.val, split=a.split,
                      device=a.device, **settings)
    print(f"[mapping] {m}")
    print("use it:\n"
          "    from tactip_mujoco_gym.sim2real import load_mapping\n"
          f"    m = load_mapping({a.out!r}); real = m.sim_to_real(sim_image)\n"
          "  or in the gym:\n"
          f"    gym.make('TacRig-v0', sensors={{'real_tactip': {{'model': {a.out!r}}}}}, "
          "obs_modes=['real_tactip'])")
    return m


def cmd_mapping_preview(a):
    from .mapping import describe_metrics, load_mapping
    from .paired import PairedData
    m = load_mapping(a.model, device=a.device)
    print(f"[mapping] {m}; trained on {len(m.info.get('sessions', []))} session(s), "
          f"{m.info.get('train_samples', '?')} samples")
    if m.info.get("metrics"):
        print("[mapping] at training: " + describe_metrics(
            m.info["metrics"], "on " + m.info.get("metrics_on", "validation")))
    data = PairedData.load(a.sessions, prep=m.prep, phases=a.phases, textures=a.textures,
                           every=a.every, max_samples=a.max_samples)
    print("[mapping] " + describe_metrics(m.evaluate(data), "on these sessions"))
    m.preview(data, n=a.n, path=a.out)
    print(f"[mapping] {a.out}: rows are samples; columns sim | real | sim->real | real AE"
          + (" | real->sim" if m.info.get("reverse", True) else ""))


def _data_args(p):
    p.add_argument("--phases", nargs="+", default=None, choices=("approach", "contact", "retract"),
                   help="only these samples (default: all, so the untouched sensor is learnt too)")
    p.add_argument("--textures", type=int, nargs="+", default=None, help="only these textures")
    p.add_argument("--every", type=int, default=1, help="every k-th sample along a trajectory")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--device", default=None, help="torch device (default: the GPU if there is one)")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tactip_mujoco_gym.sim2real",
                                 description="Calibrate the real rig + TacTip to the simulated TacRig.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("sim-capture", help="the calibration presses on the simulated rig")
    p.add_argument("--stiffness", type=float, nargs="+", default=[1.0],
                   help="sensor_stiffness values to simulate (fit picks the closest)")
    p.add_argument("--sensor-xml", default=None)
    p.add_argument("--out-dir", default="calib")
    _calib_protocol_args(p)
    p.set_defaults(fn=cmd_sim_capture)

    p = sub.add_parser("cameras", help="list the cameras that give pictures (find the TacTip)")
    p.add_argument("--max-index", type=int, default=10)
    p.add_argument("--out-dir", default="calib", help="snapshots go here")
    p.set_defaults(fn=cmd_cameras)

    p = sub.add_parser("real-setup", help="once: match the real rig to the simulated one "
                                          "(opens the simulator; same as pose-calibrate)")
    _real_args(p)
    p.add_argument("--no-sim", action="store_true",
                   help="the old text-only set-up: jog by eye to the block, no simulator")
    p.add_argument("--model", choices=("auto", "offset", "axis", "affine"), default="auto",
                   help="auto: offset with measured steps/mm (measure-steps), else axis")
    p.add_argument("--scale", default=None, help="measured steps/mm (default: "
                                                  "<out-dir>/steps_per_mm.json if there is one)")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--texture", default="flat")
    p.add_argument("--stiffness", type=float, default=1.0)
    p.add_argument("--stay", action="store_true")
    p.add_argument("--steps-per-mm", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    p.add_argument("--axis-sign", type=int, nargs=3, default=[1, 1, 1], metavar=("X", "Y", "Z"))
    p.add_argument("--measure-steps", action="store_true", help="measure steps/mm and signs")
    p.add_argument("--measure-n", type=int, default=800, help="steps moved when measuring")
    p.add_argument("--preset", type=int, nargs=2, default=[-1900, -6400], metavar=("X", "Y"),
                   help="steps from home to about the block centre (Controller.calibrate_texture's)")
    p.add_argument("--out-dir", default="calib")
    p.add_argument("--setup", default=None, help="default: <out-dir>/rig_setup.json")
    p.set_defaults(fn=cmd_real_setup)

    p = sub.add_parser("real-capture", help="the calibration presses on the real rig")
    _real_args(p)
    p.add_argument("--out-dir", default="calib")
    p.add_argument("--setup", default=None, help="default: <out-dir>/rig_setup.json")
    _calib_protocol_args(p)
    p.set_defaults(fn=cmd_real_capture)

    p = sub.add_parser("fit", help="fit the calibration from the captures")
    p.add_argument("--out-dir", default="calib")
    p.add_argument("--real", default=None, help="default: <out-dir>/real.npz")
    p.add_argument("--sim", nargs="+", default=None, help="default: <out-dir>/sim_k*.npz")
    p.add_argument("--degree", type=int, default=3, help="lens map polynomial degree")
    p.add_argument("--out", default=None)
    p.set_defaults(fn=cmd_fit)

    p = sub.add_parser("experiment", help="press a grid on texture blocks, on either rig")
    p.add_argument("--rig", choices=("sim", "real"), required=True)
    p.add_argument("--calibration", required=True)
    p.add_argument("--textures", nargs="+", default=["1", "2", "3", "4", "5", "6"])
    p.add_argument("--out", required=True)
    p.add_argument("--port")
    p.add_argument("--listener", default=None, help="default: Rig-controller's packaged one")
    p.add_argument("--rig-code", default=None, help="only if Rig-controller isn't pip-installed")
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--height", type=int, default=None)
    p.add_argument("--plate", action="store_true")
    p.add_argument("--settle", type=float, default=0.3)
    p.add_argument("--setup", default=None, help="default: rig_setup.json next to --calibration")
    _grid_protocol_args(p)
    p.set_defaults(fn=cmd_experiment)

    def teleop_args(p):
        p.add_argument("--port", required=True)
        p.add_argument("--listener", default=None, help="default: Rig-controller's packaged one")
        p.add_argument("--rig-code", default=None, help="only if Rig-controller isn't pip-installed")
        p.add_argument("--camera", type=int, default=-1, help="TacTip camera index (-1: none)")
        p.add_argument("--width", type=int, default=None)
        p.add_argument("--height", type=int, default=None)
        p.add_argument("--texture", default="flat", help="simulated block: flat, 1..20 or an STL")
        p.add_argument("--stiffness", type=float, default=1.0, help="sensor_stiffness")
        p.add_argument("--stay", action="store_true", help="don't home the real rig on exit")

    p = sub.add_parser("pose-calibrate", help="match both rigs' poses with the keyboard")
    teleop_args(p)
    p.add_argument("--out", default="calib/pose_calibration.json")
    p.add_argument("--model", choices=("auto", "offset", "axis", "affine"), default="auto",
                   help="auto: offset with measured steps/mm (measure-steps), else axis")
    p.add_argument("--scale", default=None, help="measured steps/mm (default: steps_per_mm.json "
                                                  "next to --out, if there is one)")
    p.add_argument("--preset", type=int, nargs=2, default=[-1900, -6400], metavar=("X", "Y"),
                   help="steps from home towards the block after homing")
    p.add_argument("--resume", action="store_true", help="add to the poses already in --out")
    p.add_argument("--setup", default=None, help="a rig_setup.json with known steps/mm (optional)")
    p.add_argument("--setup-out", default=None, help="default: rig_setup.json next to --out")
    p.set_defaults(fn=cmd_pose_calibrate)

    p = sub.add_parser("measure-steps", help="measure each axis's steps per mm with a rule")
    p.add_argument("--port", required=True)
    p.add_argument("--listener", default=None)
    p.add_argument("--rig-code", default=None)
    p.add_argument("--steps", type=int, default=800, help="steps moved away from the switch on x and y")
    p.add_argument("--z-steps", type=int, default=400, help="steps moved down on z (from the top)")
    p.add_argument("--out", default="calib/steps_per_mm.json")
    p.set_defaults(fn=cmd_measure_steps)

    p = sub.add_parser("pose-edit", help="drop poses from a pose calibration and refit")
    p.add_argument("--file", default="calib/pose_calibration.json")
    p.add_argument("--drop", type=int, nargs="+", default=None, metavar="N",
                   help="pose numbers to forget (1 = the first, as the summary lists them)")
    p.add_argument("--model", choices=("keep", "auto", "offset", "axis", "affine"), default="keep")
    p.add_argument("--scale", default=None)
    p.set_defaults(fn=cmd_pose_edit)

    p = sub.add_parser("follow", help="drive the simulated rig; the real rig follows")
    teleop_args(p)
    p.add_argument("--pose-calibration", default="calib/pose_calibration.json")
    p.add_argument("--margin", type=float, default=20.0,
                   help="how far (mm, x / y) past the recorded poses the real rig may go")
    p.add_argument("--no-window", action="store_true", help="only the MuJoCo viewer")
    p.set_defaults(fn=cmd_follow)

    p = sub.add_parser("compare", help="real (mapped) vs simulated data of the same experiment")
    p.add_argument("sim")
    p.add_argument("real")
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("train-mapping", help="learn the sim <-> real TacTip image mapping "
                                             "(two autoencoders + latent maps) from rig_dataset sessions")
    p.add_argument("sessions", nargs="+", help="session folders, or folders holding them (data/)")
    p.add_argument("--out", default="models/sim2real", help="folder the mapping is saved in")
    p.add_argument("--backend", choices=("torch", "linear"), default="torch",
                   help="torch: conv autoencoders + MLP maps; linear: PCA + ridge (a quick baseline)")
    p.add_argument("--input", choices=("image", "markers"), default="image",
                   help="what the simulated side is: the sim TacTip image, or the marker displacements")
    p.add_argument("--sim-size", type=int, default=64, help="sim image side the model sees (px)")
    p.add_argument("--real-size", type=int, nargs=2, default=[64, 64], metavar=("W", "H"),
                   help="real image size the model sees and makes")
    p.add_argument("--real-crop", type=int, nargs=4, default=None, metavar=("X", "Y", "W", "H"),
                   help="part of the camera frame to keep (default: the centre square)")
    p.add_argument("--real-full-frame", action="store_true", help="keep the whole camera frame")
    p.add_argument("--real-rgb", action="store_true",
                   help="the recorded real frames are RGB (rig_dataset's are BGR)")
    p.add_argument("--gray", action="store_true", help="one grey channel for both sides")
    p.add_argument("--latent-sim", type=int, default=32)
    p.add_argument("--latent-real", type=int, default=64)
    p.add_argument("--epochs", type=int, nargs=3, default=[30, 60, 20], metavar=("AE", "MAP", "JOINT"),
                   help="epochs for each autoencoder, the latent maps, then all together")
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10,
                   help="end a stage after this many epochs without a better validation loss")
    p.add_argument("--no-reverse", action="store_true", help="don't learn real -> sim")
    p.add_argument("--val", type=float, default=0.15, help="fraction held out for validation")
    p.add_argument("--split", choices=("motion", "trajectory", "texture", "session"),
                   default="motion", help="what is held out together (default: whole motions)")
    p.add_argument("--seed", type=int, default=0)
    _data_args(p)
    p.set_defaults(fn=cmd_train_mapping)

    p = sub.add_parser("mapping-preview", help="score a trained mapping on sessions and draw examples")
    p.add_argument("model", help="the mapping's folder (train-mapping --out)")
    p.add_argument("sessions", nargs="+")
    p.add_argument("--out", default="mapping_preview.png")
    p.add_argument("--n", type=int, default=8, help="samples drawn")
    _data_args(p)
    p.set_defaults(fn=cmd_mapping_preview)

    a = ap.parse_args(argv)
    if a.cmd == "experiment" and a.rig == "real" and not a.port:
        ap.error("--rig real needs --port")
    a.fn(a)


if __name__ == "__main__":
    main()
