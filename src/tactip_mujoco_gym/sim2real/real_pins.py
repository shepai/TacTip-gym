"""
The real TacTip's pins as coordinates, from recorded rig_dataset sessions: x, y from the
colour image (the white pins), z from the RealSense depth image where it has a reading.

    python -m tactip_mujoco_gym.sim2real depth-check data/session_20261117_104611
    python -m tactip_mujoco_gym.sim2real extract-pins data/session_20261117_104611

depth-check looks at some frames and says whether the depth image is usable at the pins:
how many pins have a depth reading, how far away they are, and whether pressing changes
their depth by more than the depth noise. A RealSense D4xx only measures from about 7 cm,
so close inside a TacTip it may give holes.

extract-pins finds the pins in every frame and follows each through its trajectory
(MarkerTracker), numbered the same way in every trajectory of the session. It writes
<session>/pins/<trajectory file>.npz with
    real_pins_xy   (N, P, 2) pin centres, px in the colour image
    real_pins_ok   (N, P)    found in that frame
    real_pins_z    (N, P)    depth at each pin, mm (nan where no reading)
    real_pins      (N, P, 3) movement from the trajectory's untouched frames: x, y in px,
                             z in mm (nan where unknown)
and pins/pins.json (the detector settings, the pins' rest positions, depth coverage).
PairedData and the marker map (marker_map.py) read these with the session.

The depth at a pin is read at the same place in the depth image, scaled if its size
differs. That is right when colour and depth are registered: recorded with
rig_dataset collect --d401-align, or a camera whose colour comes from the depth imager (such
as the D405). Otherwise it is approximate, and depth-check says so.
"""
import json
import os
import warnings

import numpy as np

from .markers import MarkerTracker, auto_tune, match_rest, spacing
from .resim import read_meta, traj_arrays


def _depth_scale_mm(meta):
    d = meta.get("d401") or {}
    return 1000.0 * float(d.get("depth_scale_m") or 0.001)


def _registered(meta):
    cfg = meta.get("config") or {}
    name = str((meta.get("d401") or {}).get("name", ""))
    if cfg.get("d401_align"):
        return True, "colour warped onto depth when recording"
    if "D405" in name.upper():
        return True, f"{name}: colour from the depth imager"
    return False, (f"{name or 'the camera'}: colour and depth not registered when recording, so "
                   "depth at a pin is approximate (record with --d401-align to fix)")


def depth_at(depth, xy, colour_hw, scale_mm, r=1):
    """Depth (mm) at pixel positions xy of the colour image: the median of the valid
    readings in a (2r+1)^2 window at the same place in the depth image; nan if none."""
    H, W = depth.shape[:2]
    sy, sx = H / colour_hw[0], W / colour_hw[1]
    out = np.full(len(xy), np.nan, np.float32)
    for i, (x, y) in enumerate(np.asarray(xy, float)):
        if not np.isfinite(x):
            continue
        u, v = int(round(x * sx)), int(round(y * sy))
        win = depth[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1]
        win = win[win > 0]
        if len(win):
            out[i] = float(np.median(win)) * scale_mm
    return out


def _untouched(arr):
    from .paired import _untouched as u
    ph = arr.get("phase", np.zeros(len(arr["real_tactip"]), np.int8))
    d = arr["waypoint"][:, 2] if "waypoint" in arr else np.full(len(ph), np.nan)
    return u(ph, d), ph, d


def _detector(sdir, meta, rows):
    """Auto-tuned detector and the session's rest pins, from the first trajectory with
    real images. Only pins within the centre circle of the frame count (the sensor)."""
    for row in rows:
        a = traj_arrays(sdir, row, ("real_tactip", "phase", "waypoint"), meta)
        if "real_tactip" not in a:
            continue
        still, _, _ = _untouched(a)
        frames = a["real_tactip"][still]
        H, W = frames.shape[1:3]
        det, n = auto_tune(frames[0], roi=(W / 2, H / 2, 0.5 * min(H, W)))   # the round sensor
        if det is None or n < 6:
            raise RuntimeError(f"{sdir}: couldn't find the pins in the real images (best: {n} "
                               "evenly spaced). Check the camera image shows them")
        return det, det.rest(frames), row
    raise ValueError(f"{sdir}: no trajectory has real_tactip images")


def depth_check(sdir, n_traj=12, log=print, picture=None):
    """Is the depth image usable at the pins? Returns a dict, logs a summary and can
    write a picture (one frame's pins on the colour image next to its depth)."""
    meta = read_meta(sdir)
    rows = meta.get("trajectories", [])
    scale = _depth_scale_mm(meta)
    det, rest, _ = _detector(sdir, meta, rows)
    pick = [rows[i] for i in np.unique(np.linspace(0, len(rows) - 1, min(n_traj, len(rows))).astype(int))]
    valid_frame, valid_pins, dist, noise, change, pic = [], [], [], [], [], None
    for row in pick:
        a = traj_arrays(sdir, row, ("real_tactip", "depth", "phase", "waypoint"), meta)
        if "real_tactip" not in a or "depth" not in a:
            continue
        still, ph, d = _untouched(a)
        hw = a["real_tactip"].shape[1:3]
        deepest = int(np.nanargmax(np.where(np.isfinite(d), d, -np.inf))) if np.isfinite(d).any() else len(d) // 2
        for i in still:
            valid_frame.append(float((a["depth"][i] > 0).mean()))
        P0 = det(a["real_tactip"][still[0]])
        if len(P0) < 3:
            continue
        z0 = depth_at(a["depth"][still[0]], P0, hw, scale)
        valid_pins.append(float(np.isfinite(z0).mean()))
        dist += list(z0[np.isfinite(z0)])
        if len(still) > 1:
            z1 = depth_at(a["depth"][still[1]], P0, hw, scale)
            ok = np.isfinite(z0) & np.isfinite(z1)
            if ok.any():
                noise.append(float(np.median(np.abs(z1[ok] - z0[ok]))))
        Pd = det(a["real_tactip"][deepest])
        if len(Pd):
            j = match_rest(P0, Pd)
            m = j >= 0
            zd = np.full(len(P0), np.nan, np.float32)
            zd[m] = depth_at(a["depth"][deepest], Pd[j[m]], hw, scale)
            ok = np.isfinite(z0) & np.isfinite(zd)
            if ok.any():
                change.append(float(np.percentile(np.abs(zd[ok] - z0[ok]), 90)))
        if pic is None and picture:
            pic = _picture(a["real_tactip"][deepest], a["depth"][deepest], Pd, meta)
    reg, why = _registered(meta)
    out = dict(trajectories=len(pick), pins=int(len(rest)),
               depth_valid_frame=float(np.mean(valid_frame)) if valid_frame else float("nan"),
               depth_valid_pins=float(np.mean(valid_pins)) if valid_pins else float("nan"),
               distance_mm=float(np.median(dist)) if dist else float("nan"),
               noise_mm=float(np.median(noise)) if noise else float("nan"),
               contact_change_mm=float(np.median(change)) if change else float("nan"),
               registered=reg, registration=why)
    usable = (out["depth_valid_pins"] >= 0.6 and np.isfinite(out["contact_change_mm"]) and
              out["contact_change_mm"] > 2 * max(out["noise_mm"], 0.05))
    out["usable"] = bool(usable)
    log(f"[pins] {sdir}: {out['pins']} pins found at rest")
    log(f"[pins] depth readings: {out['depth_valid_frame']:.0%} of each depth frame, "
        f"{out['depth_valid_pins']:.0%} of the pins")
    log(f"[pins] pins are {out['distance_mm']:.1f} mm from the camera (median); untouched "
        f"frame-to-frame depth noise {out['noise_mm']:.2f} mm; pressing the deepest moves the "
        f"pins (90th percentile) {out['contact_change_mm']:.2f} mm in depth")
    log(f"[pins] registration: {why}")
    log("[pins] -> " + ("depth is usable: extract-pins will give x, y and z" if usable else
                        "depth isn't usable at the pins (too few readings, or pressing doesn't "
                        "change it above the noise): extract-pins will give x and y, z where it can"))
    if pic is not None and picture:
        from .paired import save_png
        save_png(picture, pic)
        log(f"[pins] drawn: {picture}")
    return out


def _picture(frame, depth, pins, meta, h=300):
    from .paired import grid
    rgb = np.asarray(frame)[..., ::-1].copy() if np.asarray(frame).ndim == 3 else np.repeat(np.asarray(frame)[..., None], 3, 2)
    r = max(1, int(round(rgb.shape[0] / 200)))
    for x, y in np.asarray(pins, float):
        u, v = int(round(x)), int(round(y))
        rgb[max(v - r, 0):v + r + 1, max(u - r, 0):u + r + 1] = (255, 0, 255)
    d = depth.astype(np.float32)
    valid = d > 0
    pic = np.zeros(d.shape + (3,), np.uint8)
    if valid.any():
        lo, hi = np.percentile(d[valid], [2, 98])
        g = np.clip((hi - d) / max(hi - lo, 1e-6), 0, 1) * 255
        pic[..., 0] = pic[..., 1] = pic[..., 2] = g.astype(np.uint8)
    pic[~valid] = (200, 0, 0)
    return grid([[rgb, pic]], tile=h, labels=["pins found", "depth (near bright, red: none)"])


def extract_pins(sdir, log=print, detector=None):
    """Find and follow the pins through every trajectory of a session; write pins/. Returns
    the number of trajectories done."""
    meta = read_meta(sdir)
    rows = meta.get("trajectories", [])
    scale = _depth_scale_mm(meta)
    det, rest, first = _detector(sdir, meta, rows)
    if detector is not None:
        det = detector
        rest = det.rest(traj_arrays(sdir, first, ("real_tactip",), meta)["real_tactip"][:3])
    out_dir = os.path.join(sdir, "pins")
    log(f"[pins] {len(rest)} pins (pitch {spacing(rest):.1f} px); following them through "
        f"{len(rows)} trajectories")
    done, zcov = 0, []
    for k, row in enumerate(rows):
        a = traj_arrays(sdir, row, ("real_tactip", "depth", "phase", "waypoint"), meta)
        if "real_tactip" not in a:
            continue
        frames = a["real_tactip"]
        hw = frames.shape[1:3]
        still, _, _ = _untouched(a)
        tracker = MarkerTracker(rest)
        N, P = len(frames), len(rest)
        xy = np.full((N, P, 2), np.nan, np.float32)
        ok = np.zeros((N, P), bool)
        z = np.full((N, P), np.nan, np.float32)
        for i, f in enumerate(frames):
            pos, found = tracker.update(det(f))
            xy[i], ok[i] = pos, found
            if "depth" in a:
                z[i] = depth_at(a["depth"][i], np.where(found[:, None], pos, np.nan), hw, scale)
        xy[~ok] = np.nan
        with np.errstate(all="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)     # pins never seen untouched
            ref_xy = np.nanmean(xy[still], axis=0)
            ref_z = np.nanmean(z[still], axis=0)
        disp = np.concatenate([xy - ref_xy, (z - ref_z)[..., None]], axis=-1).astype(np.float32)
        path = os.path.join(out_dir, row["file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, real_pins_xy=xy, real_pins_ok=ok, real_pins_z=z, real_pins=disp)
        zcov.append(float(np.isfinite(z).mean()))
        done += 1
        if (k + 1) % 20 == 0 or k + 1 == len(rows):
            log(f"[pins] {k + 1}/{len(rows)} trajectories (pins found {ok.mean():.0%} of the time here)")
    reg, why = _registered(meta)
    info = dict(detector=det.to_dict(), rest_px=rest.tolist(), pins=int(len(rest)),
                pitch_px=spacing(rest), depth_coverage=float(np.mean(zcov)) if zcov else 0.0,
                registered=reg, registration=why, trajectories=done)
    with open(os.path.join(out_dir, "pins.json"), "w") as f:
        json.dump(info, f, indent=1)
    log(f"[pins] wrote {out_dir}: {done} trajectories; depth at the pins {info['depth_coverage']:.0%} "
        "of the time")
    return done


def pins_info(sdir):
    """pins/pins.json of a session (or of the session it was re-simulated from); None if
    extract-pins hasn't been run."""
    from .resim import source_of
    for d in (sdir, source_of(sdir)):
        if d and os.path.isfile(os.path.join(d, "pins", "pins.json")):
            with open(os.path.join(d, "pins", "pins.json")) as f:
                return json.load(f)
    return None
