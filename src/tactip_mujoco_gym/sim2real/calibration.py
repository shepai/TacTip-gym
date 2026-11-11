"""
Fit the map from what the real TacTip's camera sees to what the simulated TacTip's markers do,
from the same presses made on the real rig and on the simulated one, and apply it.

    calib = fit_calibration(real_capture, [sim_capture_k1, sim_capture_k2, ...])
    calib.save("rig_calibration.json")

    calib = RigCalibration.load("rig_calibration.json")
    to_sim = RealToSim(calib, real_rig.set_rest())        # each session: a fresh rest image
    obs = to_sim.observation(*real_rig.pins())            # = the simulated "tactile" input

What is fitted, in order:
  1. layout   which real pin is which simulated marker, and pixels -> mm on the sensor
              (layout.match_layouts, the rotation chosen by pick_by_motion)
  2. skin     which simulated skin stiffness (sensor_stiffness) moves the markers as much
              as the real skin, of those simulated (one capture per stiffness)
  3. markers  per matched marker, a linear map to the simulated marker's displacement (mm,
              3D, in the readout's frame) from the real pin's displacement (mm along the
              rig's x, y) and a few components of the whole real displacement field. The
              camera sees the pins move sideways only; how far the skin is pushed in shows
              in the shape of the whole field, so the field components carry it. Each
              marker's map is shrunk towards one shared by all markers; the number of field
              components and the shrinkage are chosen by cross-validation over the presses.
              Simulated markers the camera doesn't see take their neighbours' (inverse-
              distance weighted).
  4. force    the plate's raw total -> newtons, linear, if the real plate was read
Held-out accuracy (5-fold over presses) is reported for the geometric map alone, the shared
map and the per-marker one.
"""
import datetime
import json

import numpy as np

from .layout import PolyMap, lateral, match_layouts, pick_by_motion
from .markers import match_rest, spacing

VERSION = 1


# ============================================================================ the calibration
class RigCalibration:
    """Everything needed to turn real TacTip pin positions into the simulated readout. Saved
    as JSON (arrays as lists)."""

    def __init__(self, **kw):
        self.version = VERSION
        self.created = datetime.datetime.now().isoformat(timespec="seconds")
        # real rig
        self.steps_per_mm = None
        self.axis_sign = None
        self.detector = None                 # MarkerDetector settings
        self.real_rest = None                # (n_real, 2) px, the calibration session's rest
        # simulation
        self.sim = {}                        # env kwargs, n_markers, marker scale / clip
        self.sim_layout = None               # (n_sim, 2) mm, rest markers down the axis
        self.frame_R = None                  # (3, 3) readout frame -> world
        # maps
        self.layout_map = None               # PolyMap px -> mm
        self.real_idx = None                 # matched pins (into real_rest) ...
        self.sim_idx = None                  # ... and markers (into the simulated markers)
        self.W = None                        # (n_matched, 3, 2 + q)
        self.basis = None                    # (2 n_matched, q): whole-field components
        self.fill_idx = None                 # (n_sim, k) into the matched list (-1: none)
        self.fill_w = None                   # (n_sim, k)
        self.own = None                      # (n_sim,) the marker's own matched index or -1
        self.force_map = None                # (gain, offset): N = gain * raw + offset
        self.metrics = {}
        self.__dict__.update(kw)

    # ---- apply
    def lateral_mm(self, disp_px, ok=None):
        """Real pin displacements (n_real, 2) px from the calibration rest -> mm along the
        rig's x, y, for the matched pins: (n_matched, 2)."""
        rest = self.real_rest[self.real_idx]
        base = self.layout_map(rest)
        return self.layout_map(rest + np.asarray(disp_px, float)[self.real_idx]) - base

    def sim_markers(self, disp_px, ok=None):
        """Real pin displacements (n_real, 2) px (calibration pin order; ok: found) -> the
        simulated markers' displacements (n_sim, 3) mm in the readout's frame."""
        L = self.lateral_mm(disp_px)
        valid = np.ones(len(self.real_idx), bool) if ok is None else np.asarray(ok, bool)[self.real_idx]
        X = _features(L[None], valid[None].astype(float), self.basis)[0]
        pred = np.einsum("mij,mj->mi", self.W, X)                  # (n_matched, 3)
        return _fill(pred, valid, self.own, self.fill_idx, self.fill_w)

    def force_newtons(self, raw):
        if self.force_map is None:
            return None
        g, b = self.force_map
        return g * np.asarray(raw, float) + b

    def sim_env_kwargs(self):
        """gym.make("TacRig-v0", **these) builds the simulated rig this was calibrated to."""
        kw = {k: self.sim[k] for k in ("sensor_stiffness", "sensor_xml", "overrides") if k in self.sim}
        return kw

    # ---- save / load
    def to_dict(self):
        def arr(a):
            return None if a is None else np.asarray(a).tolist()
        return dict(version=self.version, created=self.created,
                    steps_per_mm=arr(self.steps_per_mm), axis_sign=arr(self.axis_sign),
                    detector=self.detector, real_rest=arr(self.real_rest), sim=self.sim,
                    sim_layout=arr(self.sim_layout), frame_R=arr(self.frame_R),
                    layout_map=self.layout_map.to_dict() if self.layout_map else None,
                    real_idx=arr(self.real_idx), sim_idx=arr(self.sim_idx), W=arr(self.W),
                    basis=arr(self.basis),
                    fill_idx=arr(self.fill_idx), fill_w=arr(self.fill_w), own=arr(self.own),
                    force_map=arr(self.force_map), metrics=self.metrics)

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        if d.get("version", 0) > VERSION:
            raise ValueError(f"{path}: calibration version {d['version']} is newer than this code")
        c = cls()
        for k, v in d.items():
            setattr(c, k, v)
        for k, dt in (("steps_per_mm", float), ("axis_sign", float), ("real_rest", float),
                      ("sim_layout", float), ("frame_R", float), ("real_idx", int),
                      ("sim_idx", int), ("W", float), ("basis", float), ("fill_idx", int),
                      ("fill_w", float),
                      ("own", int)):
            if getattr(c, k) is not None:
                setattr(c, k, np.asarray(getattr(c, k), dtype=dt))
        if c.layout_map is not None:
            c.layout_map = PolyMap.from_dict(c.layout_map)
        if c.force_map is not None:
            c.force_map = tuple(c.force_map)
        return c

    def summary(self):
        m = self.metrics
        lines = [f"rig calibration ({self.created})",
                 f"  pins matched      {len(self.real_idx)} of {len(self.real_rest)} real, "
                 f"{len(self.sim_idx)} of {self.sim.get('n_markers')} simulated "
                 f"(rest fit {m.get('layout_rms_mm', float('nan')):.3f} mm rms)",
                 f"  camera            {np.degrees(self.layout_map.theta):.1f} deg"
                 f"{', mirrored' if self.layout_map.mirror else ''}, "
                 f"{1000 * self.layout_map.scale:.1f} um/px at the centre",
                 f"  skin stiffness    {self.sim.get('sensor_stiffness')}",
                 f"  marker maps       {m.get('field_components', 0)} whole-field components, "
                 f"shrinkage {m.get('shrinkage', float('nan')):g}"]
        for name in ("geometric", "shared", "per_marker"):
            if name in m.get("heldout", {}):
                h = m["heldout"][name]
                lines.append(f"  held-out {name:<10} R2 {h['r2']:.3f}, rmse {h['rmse_mm']:.4f} mm")
        if self.force_map is not None:
            lines.append(f"  force             N = {self.force_map[0]:.4g} x raw + "
                         f"{self.force_map[1]:.4g} (R2 {m.get('force_r2', float('nan')):.3f})")
        return "\n".join(lines)


def _fill(pred, valid, own, fill_idx, fill_w):
    """Predicted displacements of the matched markers -> every simulated marker."""
    n = len(own)
    out = np.zeros((n, pred.shape[1]))
    for s in range(n):
        o = own[s]
        if o >= 0 and valid[o]:
            out[s] = pred[o]
            continue
        idx, w = fill_idx[s], fill_w[s]
        ok = (idx >= 0) & (w > 0)
        ok[ok] &= valid[idx[ok]]
        if ok.any():
            ww = w[ok] / w[ok].sum()
            out[s] = ww @ pred[idx[ok]]
    return out


def _fill_tables(sim_layout, sim_idx, k=4, reach=2.0):
    """For every simulated marker: its own index in the matched list (-1 if unmatched) and
    its k nearest matched markers with inverse-distance weights (0 beyond reach x pitch)."""
    S = np.asarray(sim_layout, float)
    M = S[sim_idx]
    pitch = spacing(S)
    own = -np.ones(len(S), int)
    own[sim_idx] = np.arange(len(sim_idx))
    D = np.linalg.norm(S[:, None] - M[None], axis=-1)
    D[own >= 0, own[own >= 0]] = np.inf                   # neighbours, not itself
    k = min(k, max(1, len(sim_idx) - 1))
    idx = np.argsort(D, axis=1)[:, :k]
    d = np.take_along_axis(D, idx, axis=1)
    w = 1.0 / (d + 0.1 * pitch) ** 2
    w[d > reach * pitch] = 0.0
    idx[~np.isfinite(d)] = -1
    w[~np.isfinite(d)] = 0.0
    return own, idx, w


# ============================================================================ fitting
def _features(L, V, basis):
    """(P, M, 2) mapped real displacements (V: pin found) -> (P, M, 2 + q): each marker's
    own motion, then the q whole-field components (the same for every marker)."""
    L0 = L * (np.asarray(V) > 0)[..., None]
    if basis is None or basis.shape[1] == 0:
        return L0
    g = L0.reshape(len(L0), -1) @ basis
    return np.concatenate([L0, np.repeat(g[:, None, :], L0.shape[1], axis=1)], axis=2)


def _basis(L, V, q):
    """The q main components of the whole displacement field (uncentred, so no motion -> 0),
    scaled to the size of a single marker's motion."""
    L0 = (L * (np.asarray(V) > 0)[..., None]).reshape(len(L), -1)
    if q == 0:
        return np.zeros((L0.shape[1], 0))
    _, S, Vt = np.linalg.svd(L0, full_matrices=False)
    q = min(q, int((S > 1e-12 * S[0]).sum()))
    per_marker = np.sqrt((L0 ** 2).sum() / max((np.asarray(V) > 0).sum(), 1) / 2)
    return Vt[:q].T / (S[:q] / np.sqrt(len(L0))) * per_marker


def _ridge_maps(X, Y, V, prior, lam):
    """Per marker m: W_m = argmin sum_p v |Y - W X|^2 + lam |W - prior|^2.
    X (P, M, F), Y (P, M, 3), V (P, M) weights, prior (3, F) or (M, 3, F)."""
    P, M, F = X.shape
    prior = np.broadcast_to(prior, (M, 3, F))
    XtX = np.einsum("pmi,pmj->mij", X * V[..., None], X)          # (M, F, F)
    YtX = np.einsum("pmi,pmj->mij", Y * V[..., None], X)          # (M, 3, F)
    A = XtX + lam * np.eye(F)
    B = YtX + lam * prior
    return np.linalg.solve(A.transpose(0, 2, 1), B.transpose(0, 2, 1)).transpose(0, 2, 1)


def _shared_map(X, Y, V, prior, lam=1e-3):
    """One map for every marker (pooled least squares, a little ridge towards `prior`)."""
    F = X.shape[2]
    Xf = (X * np.sqrt(V)[..., None]).reshape(-1, F)
    Yf = (Y * np.sqrt(V)[..., None]).reshape(-1, 3)
    scale = max(float((Xf ** 2).sum()) / max(len(Xf), 1) / F, 1e-12)
    A = Xf.T @ Xf + lam * scale * len(Xf) * np.eye(F)
    B = Yf.T @ Xf + lam * scale * len(Xf) * prior
    return np.linalg.solve(A.T, B.T).T


def _folds(n, k=5, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    return [idx[i::k] for i in range(min(k, n))]


def _scores(pred, true):
    sse = float(((pred - true) ** 2).sum())
    sst = float(((true - true.mean(axis=(0, 1), keepdims=True)) ** 2).sum())
    return dict(r2=1.0 - sse / sst if sst > 0 else float("nan"),
                rmse_mm=float(np.sqrt(sse / true.size)))


def _fit_marker_maps(L, Y, V, prior2, lams, qs, folds):
    """Choose the number of field components q and the shrinkage by cross-validation over
    presses, then fit on all of them. Returns (W per marker, shared map, basis, q, lambda,
    held-out predictions on the matched markers for each method)."""
    P = len(L)
    qs = [q for q in qs if q <= max(0, P - len(folds[0]) - 2)] or [0]

    def prior_for(q):
        return np.concatenate([prior2, np.zeros((3, q))], axis=1)

    def lam_scale(X, V_):
        F = X.shape[2]
        return float((np.einsum("pmi,pmi->pm", X, X) * V_).sum() / max(V_.sum(), 1) / F)

    cv = {(q, lam): 0.0 for q in qs for lam in lams}
    held_pm = {key: np.zeros_like(Y) for key in cv}
    held_sh = {q: np.zeros_like(Y) for q in qs}
    held_geo = np.zeros_like(Y)
    for te in folds:
        tr = np.setdiff1d(np.arange(P), te)
        held_geo[te] = np.einsum("ij,pmj->pmi", prior2, L[te] * (V[te] > 0)[..., None])
        for q in qs:
            B = _basis(L[tr], V[tr], q)
            Xtr, Xte = _features(L[tr], V[tr], B), _features(L[te], V[te], B)
            pr = prior_for(B.shape[1])
            G = _shared_map(Xtr, Y[tr], V[tr], pr)
            held_sh[q][te] = np.einsum("ij,pmj->pmi", G, Xte)
            sc = lam_scale(Xtr, V[tr]) * len(tr)
            for lam in lams:
                W = _ridge_maps(Xtr, Y[tr], V[tr], G, lam * sc)
                p = np.einsum("mij,pmj->pmi", W, Xte)
                held_pm[(q, lam)][te] = p
                cv[(q, lam)] += float((V[te][..., None] * (p - Y[te]) ** 2).sum())
    q, lam = min(cv, key=cv.get)
    B = _basis(L, V, q)
    X = _features(L, V, B)
    G = _shared_map(X, Y, V, prior_for(B.shape[1]))
    W = _ridge_maps(X, Y, V, G, lam * lam_scale(X, V) * P)
    held = {"geometric": held_geo, "shared": held_sh[q], "per_marker": held_pm[(q, lam)]}
    return W, G, B, int(B.shape[1]), float(lam), held


def fit_calibration(real, sims, degree=3, n_candidates=6, lams=(0.01, 0.03, 0.1, 0.3, 1, 3, 10),
                    components=(0, 1, 2, 3, 4, 6, 8), verbose=True):
    """Fit a RigCalibration from the same presses on both rigs.

    real   dict: "probes" (P, 3), "pins" (P, n_real, 2) px, "ok" (P, n_real), "rest"
           (n_real, 2) px, optional "force_raw" (P,), "steps_per_mm", "axis_sign", "detector"
    sims   dict or list of dicts, one per simulated skin stiffness: "probes" (P, 3),
           "markers" (P, n_sim, 3) mm, "lateral" (P, n_sim, 2) mm, "force" (P,), "layout"
           (n_sim, 2) mm, "frame_R" (3, 3), "sensor_stiffness", optional "env" (kwargs),
           "marker_scale", "marker_clip", "velocity"
    (capture_real / capture_sim in sim2real/__main__ write these as .npz)"""
    sims = [sims] if isinstance(sims, dict) else list(sims)
    probes = np.asarray(real["probes"], float)
    for s in sims:
        if np.asarray(s["probes"]).shape != probes.shape or not np.allclose(s["probes"], probes):
            raise ValueError("the real and simulated captures pressed at different probes: "
                             "run the same protocol on both")
    rest = np.asarray(real["rest"], float)
    pins = np.asarray(real["pins"], float)
    ok = np.asarray(real.get("ok", np.ones(pins.shape[:2], bool)), bool)

    # 1. layout (rest positions don't depend on the skin stiffness: any capture will do)
    ref = sims[0]
    cands = match_layouts(rest, ref["layout"], degree=degree, n_candidates=n_candidates,
                          verbose=verbose)
    best = pick_by_motion(cands, rest, pins, ref["lateral"], ok, verbose=verbose)
    if best.motion < 0.3 and verbose:
        print(f"[fit] warning: the best layout's motion agreement is only {best.motion:+.2f}; "
              "check the axis signs (axis_sign) and that both rigs pressed the same points")
    L = lateral(best, rest, pins)                                  # (P, M, 2) mm
    V = ok[:, best.real_idx].astype(float)

    # 2. skin stiffness: the simulation whose lateral displacements are closest in size
    table = []
    for s in sims:
        S = np.asarray(s["lateral"], float)[:, best.sim_idx]
        err = float(np.sqrt((V[..., None] * (L - S) ** 2).sum() / max(2 * V.sum(), 1)))
        ratio = float(np.sqrt((V[..., None] * L ** 2).sum() / max((V[..., None] * S ** 2).sum(), 1e-12)))
        table.append((err, s, ratio))
        if verbose:
            print(f"[fit] stiffness {s.get('sensor_stiffness')}: lateral rms difference "
                  f"{err:.4f} mm, real / sim size {ratio:.2f}")
    table.sort(key=lambda t: t[0])
    sim = table[0][1]

    # 3. per-marker maps
    frame_R = np.asarray(sim["frame_R"], float)
    prior = frame_R.T[:, :2]                       # world x, y displacement -> readout frame
    Y = np.asarray(sim["markers"], float)[:, best.sim_idx]
    folds = _folds(len(probes))
    W, G, basis, q, lam, held = _fit_marker_maps(L, Y, V, prior, lams, components, folds)
    own, fidx, fw = _fill_tables(sim["layout"], best.sim_idx)

    # held-out scores on every simulated marker (the unseen ones filled in)
    Yall = np.asarray(sim["markers"], float)
    heldout = {}
    for name, H in held.items():
        full = np.stack([_fill(H[p], V[p] > 0, own, fidx, fw) for p in range(len(H))])
        heldout[name] = _scores(full, Yall)
        heldout[name]["matched"] = _scores(H, Y)

    # 4. force
    force_map, force_r2 = None, float("nan")
    fr = real.get("force_raw")
    if fr is not None and np.all(np.isfinite(np.asarray(fr, float))):
        fr = np.asarray(fr, float)
        fs = np.asarray(sim["force"], float)
        A = np.stack([fr, np.ones_like(fr)], axis=1)
        coef, *_ = np.linalg.lstsq(A, fs, rcond=None)
        force_map = (float(coef[0]), float(coef[1]))
        force_r2 = _scores((A @ coef)[:, None, None], fs[:, None, None])["r2"]

    c = RigCalibration()
    c.steps_per_mm = None if real.get("steps_per_mm") is None else np.asarray(real["steps_per_mm"], float)
    c.axis_sign = None if real.get("axis_sign") is None else np.asarray(real["axis_sign"], float)
    c.detector = real.get("detector")
    c.real_rest = rest
    c.sim = dict(sim.get("env") or {}, sensor_stiffness=float(sim.get("sensor_stiffness", 1.0)),
                 n_markers=int(np.asarray(sim["markers"]).shape[1]),
                 marker_scale=float(sim.get("marker_scale", 1e-3)),
                 marker_clip=float(sim.get("marker_clip", 20.0)),
                 velocity=bool(sim.get("velocity", True)))
    c.sim_layout = np.asarray(sim["layout"], float)
    c.frame_R = frame_R
    c.layout_map, c.real_idx, c.sim_idx = best.map, best.real_idx, best.sim_idx
    c.W, c.basis, c.own, c.fill_idx, c.fill_w = W, basis, own, fidx, fw
    c.force_map = force_map
    c.metrics = dict(layout_rms_mm=best.rms, motion_agreement=best.motion,
                     candidates=[dict(theta_deg=float(np.degrees(k.map.theta)),
                                      mirror=bool(k.map.mirror), n=int(k.n), rms_mm=float(k.rms),
                                      motion=float(k.motion)) for k in cands],
                     stiffness=[dict(sensor_stiffness=float(s.get("sensor_stiffness", 1.0)),
                                     lateral_rms_mm=e, real_over_sim=r) for e, s, r in table],
                     shrinkage=lam, field_components=q, shared_map=G.tolist(), heldout=heldout,
                     force_r2=force_r2, n_probes=int(len(probes)))
    if verbose:
        print(c.summary())
    return c


# ============================================================================ applying it
class RealToSim:
    """Real pin positions of one session -> the simulated "tactile" input.

    session_rest   the rest pins of this session (RealRig.set_rest()): matched to the
                   calibration's rest, so the pin numbering carries over and the lighting
                   or a small camera shift since calibrating doesn't matter
    velocity       append the change since the previous call, like the simulated readout
                   (which takes it per env step: call at the env's rate, 50 Hz, to match)
    """

    def __init__(self, calib, session_rest, velocity=None):
        self.c = calib
        self.session_rest = np.asarray(session_rest, float)
        self.idx = match_rest(calib.real_rest, self.session_rest)
        self.velocity = calib.sim.get("velocity", True) if velocity is None else bool(velocity)
        self.scale_mm = calib.sim.get("marker_scale", 1e-3) * 1000.0
        self.clip = calib.sim.get("marker_clip", 20.0)
        self.n_seen = int((self.idx >= 0).sum())
        self.reset()

    def reset(self):
        self._last = None

    def markers(self, pins, ok=None):
        """Session pin positions (n_session, 2) px (tracked, rest order) -> simulated marker
        displacements (n_sim, 3) mm in the readout frame."""
        pins = np.asarray(pins, float)
        ok = np.ones(len(pins), bool) if ok is None else np.asarray(ok, bool)
        have = self.idx >= 0
        disp = np.zeros_like(self.c.real_rest)
        disp[have] = pins[self.idx[have]] - self.session_rest[self.idx[have]]
        valid = np.zeros(len(self.idx), bool)
        valid[have] = ok[self.idx[have]]
        return self.c.sim_markers(disp, valid)

    def observation(self, pins, ok=None):
        """The flat "tactile" input: [displacement, change since last call] in the simulated
        readout's units, clipped the same way."""
        d = self.markers(pins, ok)
        parts = [d.ravel() / self.scale_mm]
        if self.velocity:
            delta = np.zeros_like(d) if self._last is None else d - self._last
            parts.append(delta.ravel() / self.scale_mm)
        self._last = d
        return np.clip(np.concatenate(parts), -self.clip, self.clip).astype(np.float32)
