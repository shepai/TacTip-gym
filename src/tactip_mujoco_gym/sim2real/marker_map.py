"""
Marker-to-marker map: the simulated TacTip's markers <-> the real TacTip's pins.

The sim side is the simulated markers' displacements (n x 3, mm), as rig_dataset records
them (or a re-simulated session gives them). The real side is the real pins' movement from
each trajectory's untouched frames, from extract-pins: x, y in px of the camera image, and z
in mm when the depth image has it (use_z). Both are a few hundred numbers instead of images,
and both are already "change from untouched", so the camera's light and exposure don't come
into it.

    python -m tactip_mujoco_gym.sim2real extract-pins data/session_20261117_104611
    python -m tactip_mujoco_gym.sim2real train-marker-map data/session_20261117_104611 --out models/pins

    from tactip_mujoco_gym.sim2real import load_marker_map
    mm = load_marker_map("models/pins")
    real = mm.sim_to_real(sim_disp)          # (n, 3) mm -> (P, 2 or 3) real pin movement
    sim = mm.real_to_sim(real_pins)          # (P, 2 or 3), nan where a pin wasn't found -> (n, 3)

    gym.make("TacRig-v0", sensors={"real_pins": {"model": "models/pins"}}, obs_modes=["real_pins"])

Backends: "linear" (ridge regression both ways, numpy; the strength chosen on validation)
and "mlp" (two small PyTorch networks, trained with missing pins left out of the loss).
Validation holds out whole motions, as for the image mapping. Scores are normalised MSE
(1 = always answering the average) and RMSE in px / mm.

Pins are numbered by the first session's rest image; other sessions' pins are matched to
them (match_rest), so several sessions can be trained on together.
"""
import datetime
import json
import os

import numpy as np

from .markers import match_rest
from .paired import PHASE_CODES, _corr, find_sessions
from .real_pins import pins_info
from .resim import read_meta, traj_arrays

CONFIG = "marker_map.json"


class MarkerPairs:
    """sim (N, n, 3) mm; real (N, P, k) pin movement (nan where unknown); traj, phase per
    sample; rows: one per trajectory."""

    def __init__(self, sim, real, traj, phase, rows, sessions, use_z, rest_px):
        self.sim, self.real, self.traj, self.phase = sim, real, traj, phase
        self.rows, self.sessions, self.use_z, self.rest_px = rows, sessions, use_z, rest_px

    def __len__(self):
        return len(self.sim)

    @classmethod
    def load(cls, paths, use_z="auto", phases=None, every=1, log=print):
        sessions = find_sessions(paths)
        want = None if not phases else {PHASE_CODES.get(p, p) for p in phases}
        sims, reals, trajs, phs, rows = [], [], [], [], []
        ref = None
        for si, sdir in enumerate(sessions):
            info = pins_info(sdir)
            if info is None:
                log(f"[pins] {sdir}: no pins extracted (run extract-pins first), skipped")
                continue
            rest = np.asarray(info["rest_px"], float)
            if ref is None:
                ref, index = rest, np.arange(len(rest))
            else:
                index = match_rest(ref, rest)
            meta = read_meta(sdir)
            n0 = sum(len(s) for s in sims)
            for row in meta.get("trajectories", []):
                a = traj_arrays(sdir, row, ("sim_markers", "real_pins", "phase"), meta)
                if "sim_markers" not in a or "real_pins" not in a:
                    continue
                keep = np.arange(len(a["real_pins"]))[::max(1, int(every))]
                ph = a.get("phase", np.full(len(a["real_pins"]), -1, np.int8))
                if want is not None:
                    keep = keep[np.isin(ph[keep], list(want))]
                if not len(keep):
                    continue
                rp = a["real_pins"][keep]
                real = np.full((len(keep), len(ref), 3), np.nan, np.float32)
                m = index >= 0
                real[:, m] = rp[:, index[m]]
                rows.append(dict(row, session=sdir, session_index=si))
                sims.append(a["sim_markers"][keep].astype(np.float32))
                reals.append(real)
                trajs.append(np.full(len(keep), len(rows) - 1, np.int32))
                phs.append(ph[keep].astype(np.int8))
            log(f"[pins] {sdir}: {sum(len(s) for s in sims) - n0} samples")
        if not sims:
            raise ValueError("no samples with both sim_markers and extracted pins")
        sim, real = np.concatenate(sims), np.concatenate(reals)
        if use_z == "auto":
            xy = np.isfinite(real[..., 0])
            cover = float(np.isfinite(real[..., 2])[xy].mean()) if xy.any() else 0.0
            use_z = cover >= 0.5
            log(f"[pins] depth at the pins {cover:.0%} of the time: "
                + ("using x, y and z" if use_z else "using x and y only"))
        if not use_z:
            real = real[..., :2]
        return cls(sim, real, np.concatenate(trajs), np.concatenate(phs), rows, sessions,
                   bool(use_z), ref)

    def split(self, val=0.15, seed=0):
        """(train, val) indices, whole motions (all their repeats) at a time."""
        keys = []
        for r in self.rows:
            m = r.get("motion")
            keys.append((r["session_index"], str(r.get("texture")), str(m)) if m not in (None, "")
                        else (r["session_index"], r["file"]))
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        g = np.array([ids[k] for k in keys])[self.traj]
        uniq = np.unique(g)
        rng = np.random.default_rng(seed)
        if len(uniq) < 2:
            idx = rng.permutation(len(g))
            n = max(1, int(round(val * len(g))))
            return np.sort(idx[n:]), np.sort(idx[:n])
        order = rng.permutation(uniq)
        n_val = min(len(uniq) - 1, max(1, int(round(val * len(uniq)))))
        v = np.isin(g, order[:n_val])
        return np.flatnonzero(~v), np.flatnonzero(v)


# ============================================================================ the map
def _masked_nmse(pred, true):
    ok = np.isfinite(true)
    if not ok.any():
        return float("nan")
    t = np.where(ok, true, 0.0)
    cnt = np.maximum(ok.sum(axis=0), 1)
    mean = t.sum(axis=0) / cnt
    var = (((t - mean) ** 2) * ok).sum(axis=0) / cnt
    err = (((pred - t) ** 2) * ok).sum() / ok.sum()
    return float(err / max(var[ok.any(axis=0)].mean(), 1e-12))


class MarkerMap:
    """Standardised sim and real vectors, maps both ways (ridge or MLP)."""

    def __init__(self, cfg, params=None, nets=None):
        self.cfg = cfg
        self.p = params or {}
        self.nets = nets                     # (sim->real, real->sim) torch modules, or None
        self.device = "cpu"

    def __repr__(self):
        c = self.cfg
        return (f"MarkerMap({c['backend']}: sim {c['n_markers']} markers x 3 <-> real "
                f"{c['pins']} pins x {c['dims']})")

    @property
    def use_z(self):
        return self.cfg["dims"] == 3

    # ---- vectors
    def _x(self, sim):
        x = np.asarray(sim, np.float32).reshape(len(sim), -1)
        return (x - self.p["x_mean"]) / self.p["x_std"]

    def _y(self, real):
        y = np.asarray(real, np.float32).reshape(len(real), -1)
        return (y - self.p["y_mean"]) / self.p["y_std"]

    def _sr(self, xs):
        if self.nets is not None:
            return self._run(self.nets[0], xs)
        return xs @ self.p["W_sr"]

    def _rs(self, ys):
        ys = np.where(np.isfinite(ys), ys, 0.0)          # a missing pin: its average movement
        if self.nets is not None:
            return self._run(self.nets[1], ys)
        return ys @ self.p["W_rs"]

    def _run(self, net, a):
        import torch
        net.eval()
        with torch.no_grad():
            return net(torch.as_tensor(np.asarray(a, np.float32), device=self.device)).cpu().numpy()

    # ---- use
    def sim_to_real(self, sim):
        """Simulated marker displacements (n, 3) mm (or (N, n, 3)) -> real pin movement
        (P, dims): x, y px [, z mm]."""
        sim = np.asarray(sim, np.float32)
        single = sim.ndim == 2 and sim.shape == (self.cfg["n_markers"], 3) or sim.ndim == 1
        sim = sim.reshape(1 if single else len(sim), -1)
        y = self._sr(self._x(sim)) * self.p["y_std"] + self.p["y_mean"]
        y = y.reshape(len(y), self.cfg["pins"], self.cfg["dims"])
        return y[0] if single else y

    __call__ = sim_to_real

    def real_to_sim(self, real):
        """Real pin movement (P, dims) (nan where a pin wasn't found; or (N, P, dims)) ->
        simulated marker displacements (n, 3) mm."""
        real = np.asarray(real, np.float32)
        single = real.ndim == 2
        real = real[None] if single else real
        real = real[..., :self.cfg["dims"]]
        x = self._rs(self._y(real)) * self.p["x_std"] + self.p["x_mean"]
        x = x.reshape(len(x), self.cfg["n_markers"], 3)
        return x[0] if single else x

    def evaluate(self, pairs, idx):
        X, Y = pairs.sim[idx], pairs.real[idx]
        pr, ps = self.sim_to_real(X), self.real_to_sim(Y)
        ok = np.isfinite(Y)
        out = {"samples": int(len(idx)),
               "sim2real_nmse": _masked_nmse(pr.reshape(len(idx), -1), Y.reshape(len(idx), -1)),
               "real2sim_nmse": _masked_nmse(ps.reshape(len(idx), -1), X.reshape(len(idx), -1)),
               "real2sim_rmse_mm": float(np.sqrt(np.mean((ps - X) ** 2)))}
        err = np.where(ok, pr - np.where(ok, Y, 0.0), np.nan)
        out["sim2real_rmse_xy_px"] = float(np.sqrt(np.nanmean(err[..., :2] ** 2)))
        if self.use_z:
            out["sim2real_rmse_z_mm"] = float(np.sqrt(np.nanmean(err[..., 2] ** 2)))
        out["movement_corr"] = _corr(np.linalg.norm(X, axis=2).mean(axis=1),
                                     np.nanmean(np.linalg.norm(np.nan_to_num(Y), axis=2), axis=1))
        return out

    # ---- save / load
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        cfg = dict(self.cfg)
        with open(os.path.join(path, CONFIG), "w") as f:
            json.dump(_jsonable(cfg), f, indent=1)
        np.savez(os.path.join(path, "weights.npz"), **self.p)
        if self.nets is not None:
            import torch
            torch.save({"sr": self.nets[0].state_dict(), "rs": self.nets[1].state_dict()},
                       os.path.join(path, "mlp.pt"))
        return path

    @classmethod
    def load(cls, path, device="cpu"):
        f = path if os.path.isfile(path) else os.path.join(path, CONFIG)
        if not os.path.isfile(f):
            raise FileNotFoundError(f"no marker map at {path} (train one with python -m "
                                    "tactip_mujoco_gym.sim2real train-marker-map)")
        d = os.path.dirname(os.path.abspath(f))
        with open(f) as fh:
            cfg = json.load(fh)
        with np.load(os.path.join(d, "weights.npz")) as z:
            p = {k: z[k] for k in z.files}
        m = cls(cfg, p)
        if cfg["backend"] == "mlp":
            import torch
            nets = _mlps(cfg)
            try:
                st = torch.load(os.path.join(d, "mlp.pt"), map_location=device, weights_only=True)
            except TypeError:
                st = torch.load(os.path.join(d, "mlp.pt"), map_location=device)
            nets[0].load_state_dict(st["sr"])
            nets[1].load_state_dict(st["rs"])
            m.nets = (nets[0].to(device), nets[1].to(device))
            m.device = device
        return m


def load_marker_map(path, device="cpu"):
    """Load a trained marker map: sim markers in -> real pins out (and back)."""
    return MarkerMap.load(path, device)


def read_marker_map_config(path):
    f = path if os.path.isfile(path) else os.path.join(path, CONFIG)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"no marker map at {path} (expected {CONFIG}; train one with "
                                "python -m tactip_mujoco_gym.sim2real train-marker-map)")
    with open(f) as fh:
        cfg = json.load(fh)
    cfg["dir"] = os.path.dirname(os.path.abspath(f))
    return cfg


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return _jsonable(v.tolist())
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        v = v.item()
    if isinstance(v, float) and not np.isfinite(v):
        return None
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


def _mlps(cfg):
    import torch.nn as nn
    dx, dy, h = 3 * cfg["n_markers"], cfg["pins"] * cfg["dims"], int(cfg.get("hidden", 256))

    def mlp(a, b):
        return nn.Sequential(nn.Linear(a, h), nn.SiLU(), nn.Dropout(0.1), nn.Linear(h, h),
                             nn.SiLU(), nn.Linear(h, b))
    return mlp(dx, dy), mlp(dy, dx)


def train_marker_map(pairs, out=None, backend="linear", val=0.15, seed=0, hidden=256,
                     epochs=300, lr=1e-3, batch=256, patience=30, device=None, log=print):
    """Fit the maps on MarkerPairs; save to `out`. Returns the MarkerMap."""
    if backend not in ("linear", "mlp"):
        raise ValueError("backend is 'linear' or 'mlp'")
    tr, va = pairs.split(val, seed)
    X = pairs.sim.reshape(len(pairs), -1)
    Y = pairs.real.reshape(len(pairs), -1)
    x_mean, x_std = X[tr].mean(axis=0), np.maximum(X[tr].std(axis=0), 1e-3)
    with np.errstate(all="ignore"):
        y_mean = np.nan_to_num(np.nanmean(Y[tr], axis=0))
        y_std = np.maximum(np.nan_to_num(np.nanstd(Y[tr], axis=0)), 1e-3)
    cfg = dict(version=1, backend=backend, n_markers=int(pairs.sim.shape[1]),
               pins=int(pairs.real.shape[1]), dims=int(pairs.real.shape[2]), hidden=int(hidden),
               rest_px=pairs.rest_px, sessions=[os.path.abspath(s) for s in pairs.sessions],
               train_samples=int(len(tr)), val_samples=int(len(va)),
               created=datetime.datetime.now().isoformat(timespec="seconds"))
    m = MarkerMap(cfg, dict(x_mean=x_mean.astype(np.float32), x_std=x_std.astype(np.float32),
                            y_mean=y_mean.astype(np.float32), y_std=y_std.astype(np.float32)))
    Xs, Ys = m._x(X), m._y(Y)
    ok = np.isfinite(Ys)
    Yf = np.where(ok, Ys, 0.0)
    log(f"[pins] {m}: train {len(tr)}, validate {len(va)} (whole motions held out)")
    vi = va if len(va) else tr
    if backend == "linear":
        for src, dst, A, B, name, okd in (("x", "y", Xs, Yf, "W_sr", ok), ("y", "x", Yf, Xs, "W_rs", None)):
            best = None
            for ridge in (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0):
                W = np.linalg.solve(A[tr].T @ A[tr] + ridge * len(tr) * np.eye(A.shape[1]),
                                    A[tr].T @ B[tr]).astype(np.float32)
                pred = A[vi] @ W
                tgt = np.where(okd[vi], B[vi], np.nan) if okd is not None else B[vi]
                err = _masked_nmse(pred, tgt)
                if best is None or err < best[0]:
                    best = (err, ridge, W)
            m.p[name] = best[2]
            log(f"[pins] {'sim -> real' if name == 'W_sr' else 'real -> sim'}: ridge {best[1]:g}, "
                f"val {best[0]:.3f}")
    else:
        m.nets, m.device = _fit_mlps(cfg, Xs, Yf, ok, tr, va, epochs, lr, batch, patience, device, seed, log)
    cfg["metrics"] = m.evaluate(pairs, vi)
    mt = cfg["metrics"]
    log(f"[pins] on {mt['samples']} validation samples: sim->real {mt['sim2real_nmse']:.3f} "
        f"(pins off by {mt['sim2real_rmse_xy_px']:.2f} px"
        + (f", {mt['sim2real_rmse_z_mm']:.2f} mm in z" if "sim2real_rmse_z_mm" in mt else "")
        + f"), real->sim {mt['real2sim_nmse']:.3f} ({mt['real2sim_rmse_mm']:.3f} mm); "
        "normalised MSE, 1 = always the average")
    if out:
        m.save(out)
        log(f"[pins] saved {out}")
    return m


def _fit_mlps(cfg, Xs, Yf, ok, tr, va, epochs, lr, batch, patience, device, seed, log):
    import torch
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sr, rs = (n.to(dev) for n in _mlps(cfg))
    T = lambda a: torch.as_tensor(np.asarray(a, np.float32), device=dev)  # noqa: E731
    X, Y, M = T(Xs), T(Yf), T(ok)
    vi = va if len(va) else tr
    best = []
    for name, net, src, dst, mask in (("sim -> real", sr, X, Y, M), ("real -> sim", rs, Y, X, None)):
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)

        def loss(idx):
            p = net(src[idx])
            if mask is None:
                return ((p - dst[idx]) ** 2).mean()
            w = mask[idx]
            return (((p - dst[idx]) ** 2) * w).sum() / w.sum().clamp(min=1)
        top, state, bad = np.inf, None, 0
        for ep in range(int(epochs)):
            net.train()
            perm = tr[rng.permutation(len(tr))]
            for i in range(0, len(perm), batch):
                idx = torch.as_tensor(perm[i:i + batch], device=dev)
                opt.zero_grad(set_to_none=True)
                loss(idx).backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(loss(torch.as_tensor(vi, device=dev)))
            if v < top - 1e-6:
                top, bad = v, 0
                state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        net.load_state_dict(state)
        net.eval()
        best.append(top)
        log(f"[pins] {name}: {ep + 1} epochs, val loss {top:.3f} (standardised)")
    return (sr, rs), dev
