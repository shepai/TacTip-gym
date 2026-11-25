"""
Re-run recorded rig_dataset trials in simulation: the same textures, the same paths and the
same samples, recording simulated readouts the session didn't have (the markers, a depth
camera inside the TacTip, the camera image), and correcting the calibration's z after the
fact.

    python -m tactip_mujoco_gym.sim2real resim data/session_20261117_104611 --z-offset fit
    -> data/session_20261117_104611_resim/   (meta.json + one small .npz per trajectory)

A re-simulated session holds only the new simulated arrays. Its meta.json points at the
recorded session ("resim": {"source_session": ...}), and PairedData and the marker map read
the real arrays (real_tactip, depth, pins) from there, so nothing large is copied. The
re-simulated sim_* arrays replace the recorded ones; sim arrays not re-recorded are left out
rather than mixed (they would be at the old z).

z offset: the real TacTip may have sat a little lower or higher than the simulated one (the
pairing check's "real starts changing X mm before the sim"). --z-offset 0.6 presses the
simulated TacTip 0.6 mm deeper at every sample; "fit" tries a range of offsets on some
trajectories and keeps the one where, along each press, the simulated markers start moving
at the depth where the recorded real images start changing (sobel edges, compared with each
trajectory's untouched frames).
"""
import datetime
import json
import os

import numpy as np

RECORDABLE = ("markers", "depth", "image", "force")
FIT_VALUES = tuple(np.round(np.arange(-0.5, 1.51, 0.25), 2))


def read_meta(sdir):
    with open(os.path.join(sdir, "meta.json")) as f:
        return json.load(f)


def source_of(sdir, meta=None):
    """The recorded session a re-simulated one came from (None if it is a recording)."""
    meta = read_meta(sdir) if meta is None else meta
    rs = meta.get("resim") or {}
    if not rs:
        return None
    for p in (rs.get("source_session"), os.path.join(sdir, rs.get("source_relative") or "")):
        if p and os.path.isfile(os.path.join(p, "meta.json")):
            return p
    raise FileNotFoundError(f"{sdir} was re-simulated from {rs.get('source_session')}, which "
                            "isn't there any more")


def traj_arrays(sdir, row, keys=None, meta=None):
    """One trajectory's arrays, merged: the recorded session's, its extracted pins
    (pins/), then a re-simulated session's own (whose sim_* replace the recording's).
    keys: only these (default all)."""
    meta = read_meta(sdir) if meta is None else meta
    src = source_of(sdir, meta)
    layers = []
    if src:
        layers += [(os.path.join(src, row["file"]), True), (os.path.join(src, "pins", row["file"]), False)]
    layers += [(os.path.join(sdir, row["file"]), False), (os.path.join(sdir, "pins", row["file"]), False)]
    out = {}
    for path, drop_sim in layers:
        if not os.path.exists(path):
            continue
        with np.load(path) as z:
            for k in z.files:
                if drop_sim and k.startswith("sim_"):
                    continue
                if keys is None or k in keys:
                    out[k] = z[k]
    return out


# ============================================================================ the replay
class Replayer:
    """Drives TacRig-v0 (through SimRig) along recorded waypoints as rig_dataset did: to each
    waypoint, settle, sample."""

    def __init__(self, rig, settle_steps=10, z_offset=0.0, record=("markers", "depth")):
        self.rig = rig
        self.settle_steps = int(settle_steps)
        self.z_offset = float(z_offset)
        self.record = tuple(record)
        self.texture = None
        e = rig.e
        self.depth = e.sensors.get("tactile_depth") if "depth" in self.record else None
        self.image = e.sensors.get("tactile_image") if "image" in self.record else None
        for name, s in (("depth", self.depth), ("image", self.image)):
            if name in self.record and (s is None or not getattr(s, "available", True)):
                raise RuntimeError(f"no simulated {name} readout: {getattr(s, 'reason', 'missing')}")

    def target(self, wp):
        x, y, d = (float(v) for v in wp[:3])
        return np.array([x, y, -(self.rig.start_height_mm + d + self.z_offset)])

    def move(self, target):
        for _ in range(5000):
            if self.rig.step_toward(target):
                break
        for _ in range(self.settle_steps):
            self.rig.step_toward(target)

    def sample(self):
        r = self.rig
        s = {"sim_pose": np.asarray(r._raw(), np.float32)}
        if "markers" in self.record:
            s["sim_markers"] = np.asarray(r.displacement(), np.float32)
        if "force" in self.record:
            s["sim_force"] = np.float32(r.e.plate_force)
        if self.depth is not None:
            s["sim_depth"] = np.asarray(self.depth.observe(r.e), np.float32)[..., 0]
        if self.image is not None:
            s["sim_tactip"] = np.asarray(self.image.observe(r.e), np.uint8)
        return s

    def run(self, waypoints, texture=None):
        """Every waypoint in turn -> {name: (N, ...) array}. A new texture resets the rig
        (rig_dataset reset at each texture, not each trajectory)."""
        if texture is not None and texture != self.texture:
            self.rig.reset(texture=texture, verbose=False)
            self.texture = texture
        out = {}
        for wp in np.asarray(waypoints, float):
            self.move(self.target(wp))
            for k, v in self.sample().items():
                out.setdefault(k, []).append(v)
        return {k: np.stack(v) for k, v in out.items()}


def make_rig(meta, record=("markers", "depth"), depth_size=64, image_size=None):
    """A SimRig on TacRig-v0 set up like the session's (textures, skin stiffness), with the
    readouts `record` needs."""
    import gymnasium as gym

    import tactip_mujoco_gym  # noqa: F401  (registers TacRig-v0)
    from .rigs import SimRig
    sim = meta.get("sim") or {}
    textures = sorted({int(r["texture"]) for r in meta.get("trajectories", [])
                       if str(r.get("texture", "")).lstrip("-").isdigit()}) or None
    sensors = {}
    if "depth" in record:
        sensors["tactile_depth"] = {"type": "tactip_depth_camera", "width": int(depth_size),
                                    "height": int(depth_size)}
    if "image" in record:
        s = int(image_size or sim.get("image_size") or 240)
        sensors["tactile_image"] = {"width": s, "height": s}
    kw = dict(sensor_stiffness=float(sim.get("sensor_stiffness", 1.0)), sensors=sensors)
    if textures:
        kw["textures"] = textures
    return SimRig(env=gym.make(sim.get("env", "TacRig-v0"), **kw))


def _settle_steps(meta):
    return int((meta.get("config") or {}).get("sim_settle_steps", 10))


def _real_change(sdir, row, meta, prep):
    """How much each recorded real image differs from its trajectory's untouched frames
    (sobel edges, as the pairing check with --filter sobel)."""
    from .paired import _untouched
    a = traj_arrays(sdir, row, ("real_tactip", "phase", "waypoint"), meta)
    if "real_tactip" not in a:
        return None
    ph = a.get("phase", np.zeros(len(a["real_tactip"]), np.int8))
    depth = a["waypoint"][:, 2] if "waypoint" in a else np.full(len(ph), np.nan)
    y = prep.real(a["real_tactip"])[0].astype(np.float32)
    ref = y[_untouched(ph, depth)].mean(axis=0)
    return np.abs(y - ref).reshape(len(y), -1).mean(axis=1)


def _onset_gap(sims, reals, trajs):
    """Median over descents of (depth where the real image starts changing) - (depth where
    the simulated markers do): 0 when the two touch at the same depth."""
    from .paired import _onset
    gaps = []
    for s, r, (ph, d) in zip(sims, reals, trajs):
        if not np.isfinite(d).all():
            continue
        end = len(d)
        for i in range(1, len(d)):
            if d[i] < d[i - 1] - 1e-3 or ph[i] == 2:
                end = i
                break
        clear = (ph[:end] == 0) & (d[:end] <= -0.5)
        s, r = s[:end], r[:end]
        ns = float(np.percentile(s[clear], 95)) if clear.any() else 0.0
        nr = float(np.percentile(r[clear], 95)) if clear.any() else 0.0
        a, b = _onset(d[:end], s, clear, ns), _onset(d[:end], r, clear, nr)
        if a is not None and b is not None:
            gaps.append(b - a)
    return float(np.median(gaps)) if gaps else float("nan")


def fit_z_offset(sdir, rig=None, values=FIT_VALUES, n_traj=16, log=print, meta=None):
    """Try z offsets on up to n_traj trajectories (presses, taps and strokes first) and
    return (best offset, {offset: (onset gap mm, correlation)}). The best makes the
    simulated markers start moving at the depth where the real images start changing
    (along each press); the correlation of the two is reported too."""
    from .paired import Preprocess, _corr
    meta = read_meta(sdir) if meta is None else meta
    rows = meta.get("trajectories", [])
    pref = [r for r in rows if r.get("kind") in ("press", "tap", "ramp_stroke", "stroke")] or rows
    pick = [pref[i] for i in np.unique(np.linspace(0, len(pref) - 1, min(n_traj, len(pref))).astype(int))]
    prep = Preprocess(real_size=(128, 128), filter="sobel")
    real, wps, trajs = [], [], []
    for row in pick:
        sig = _real_change(sdir, row, meta, prep)
        a = traj_arrays(sdir, row, ("waypoint", "phase"), meta)
        if sig is None or "waypoint" not in a:
            continue
        real.append(sig)
        wps.append((row, a["waypoint"]))
        trajs.append((a.get("phase", np.zeros(len(sig), np.int8)), a["waypoint"][:, 2]))
    if not real:
        raise ValueError(f"{sdir}: no trajectories with real images and waypoints to fit z on")
    rig = rig or make_rig(meta, record=("markers",))
    rep = Replayer(rig, _settle_steps(meta), record=("markers",))
    scores = {}
    log(f"[resim] fitting z on {len(real)} trajectories, offsets {[float(v) for v in values]} mm "
        "(+ = the simulated TacTip deeper)")
    for dz in values:
        rep.z_offset, rep.texture = float(dz), None
        sim = []
        for row, wp in wps:
            m = rep.run(wp, texture=_texture(row))["sim_markers"]
            sim.append(np.linalg.norm(m, axis=2).mean(axis=1))
        gap = _onset_gap(sim, real, trajs)
        scores[float(dz)] = (gap, _corr(np.concatenate(sim), np.concatenate(real)))
        log(f"[resim]   z {dz:+.2f} mm: real starts changing {gap:+.2f} mm from the sim, "
            f"corr {scores[float(dz)][1]:+.3f}")
    finite = {k: v for k, v in scores.items() if np.isfinite(v[0])}
    if finite:
        best = min(finite, key=lambda k: (abs(finite[k][0]), -np.nan_to_num(finite[k][1])))
    else:
        finite = {k: v for k, v in scores.items() if np.isfinite(v[1])}
        if not finite:
            raise ValueError("no offset could be judged (no presses with a clear onset)")
        log("[resim] no clear onsets: choosing by correlation")
        best = max(finite, key=lambda k: finite[k][1])
    if best in (min(values), max(values)):
        log(f"[resim] the best offset ({best:+.2f} mm) is at the end of the range tried: try "
            "wider --fit-values")
    return best, scores


def _texture(row):
    t = row.get("texture")
    return int(t) if str(t).lstrip("-").isdigit() else t


def resimulate(sdir, out=None, record=("markers", "depth"), z_offset=0.0, depth_size=64,
               image_size=None, fit_values=FIT_VALUES, fit_trajectories=16, rig=None, log=print):
    """Replay every trajectory of a recorded session in simulation; write the re-simulated
    session to `out` (default <session>_resim). z_offset: mm (+ = sim deeper) or "fit".
    Returns the output folder."""
    bad = set(record) - set(RECORDABLE)
    if bad:
        raise ValueError(f"can record {RECORDABLE}, not {sorted(bad)}")
    if source_of(sdir):
        raise ValueError(f"{sdir} is itself re-simulated: give the recorded session")
    meta = read_meta(sdir)
    rows = meta.get("trajectories", [])
    if not rows:
        raise ValueError(f"{sdir}: no trajectories recorded")
    out = out or os.path.normpath(sdir) + "_resim"
    rig = rig or make_rig(meta, record, depth_size, image_size)
    fit = None
    if z_offset == "fit":
        z_offset, fit = fit_z_offset(sdir, rig, fit_values, fit_trajectories, log, meta)
        log(f"[resim] z offset: {z_offset:+.2f} mm")
    rep = Replayer(rig, _settle_steps(meta), float(z_offset), record)
    os.makedirs(out, exist_ok=True)
    done = []
    for k, row in enumerate(rows):
        a = traj_arrays(sdir, row, ("waypoint", "phase"), meta)
        if "waypoint" not in a:
            log(f"[resim] {row['file']}: no waypoints, skipped")
            continue
        arrays = rep.run(a["waypoint"], texture=_texture(row))
        arrays["waypoint"] = a["waypoint"]
        if "phase" in a:
            arrays["phase"] = a["phase"]
        path = os.path.join(out, row["file"])
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez_compressed(path, **arrays)
        done.append(row)
        if (k + 1) % 10 == 0 or k + 1 == len(rows):
            log(f"[resim] {k + 1}/{len(rows)} trajectories")
    new = dict(meta)
    new["trajectories"] = done
    new["sim"] = dict(meta.get("sim") or {})
    if "image" in record:
        new["sim"]["image_size"] = int(image_size or new["sim"].get("image_size") or 240)
    else:
        new["sim"]["image_size"] = None
    new["sim"]["depth_size"] = int(depth_size) if "depth" in record else None
    new["sources"] = [{"markers": "sim_markers", "depth": "sim_depth", "image": "sim_tactip",
                       "force": "sim_force"}[r] for r in record] + ["sim_pose"]
    new["resim"] = dict(source_session=os.path.abspath(sdir),
                        source_relative=os.path.relpath(os.path.abspath(sdir), os.path.abspath(out)),
                        z_offset_mm=float(z_offset), record=list(record),
                        fit=None if fit is None else {f"{k:+.2f}": {"onset_gap_mm": v[0], "corr": v[1]}
                                                      for k, v in fit.items()},
                        created=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(new, f, indent=1, default=lambda v: v.tolist() if hasattr(v, "tolist") else str(v))
    log(f"[resim] wrote {out}: {len(done)} trajectories, z offset {z_offset:+.2f} mm, "
        f"recorded {', '.join(record)}")
    return out
