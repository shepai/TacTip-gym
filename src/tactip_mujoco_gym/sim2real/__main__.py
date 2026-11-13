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


def cmd_pose_calibrate(a):
    from .teleop import pose_calibrate
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    known = load_rig_setup(a.setup) if a.setup and os.path.exists(a.setup) else {}
    sim, real = _teleop_rigs(a, known)
    pose_calibrate(sim, real, a.out, kind=a.model, preset=a.preset, resume=a.resume,
                   setup_out=a.setup_out, home_on_exit=not a.stay,
                   known_spm=known.get("steps_per_mm"), known_sign=known.get("axis_sign"))
    sim.close()
    real.close()


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
    p.add_argument("--model", choices=("axis", "affine"), default="axis")
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
    p.add_argument("--model", choices=("axis", "affine"), default="axis")
    p.add_argument("--preset", type=int, nargs=2, default=[-1900, -6400], metavar=("X", "Y"),
                   help="steps from home towards the block after homing")
    p.add_argument("--resume", action="store_true", help="add to the poses already in --out")
    p.add_argument("--setup", default=None, help="a rig_setup.json with known steps/mm (optional)")
    p.add_argument("--setup-out", default=None, help="default: rig_setup.json next to --out")
    p.set_defaults(fn=cmd_pose_calibrate)

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

    a = ap.parse_args(argv)
    if a.cmd == "experiment" and a.rig == "real" and not a.port:
        ap.error("--rig real needs --port")
    a.fn(a)


if __name__ == "__main__":
    main()
