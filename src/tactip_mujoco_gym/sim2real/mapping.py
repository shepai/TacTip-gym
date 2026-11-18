"""
Learned sim <-> real TacTip mapping: an autoencoder for each sensor, and maps between their
latent spaces.

    sim input  --E_sim-->  z_sim  --map_sr-->  z_real  --D_real-->  real TacTip image
    real frame --E_real--> z_real --map_rs-->  z_sim   --D_sim-->   sim input

The sim input is the simulated TacTip camera image, or the simulated markers'
displacements. Training (train_mapping, or `python -m tactip_mujoco_gym.sim2real
train-mapping`) uses paired rig_dataset sessions (paired.py), in four stages:

    1. sim autoencoder    E_sim, D_sim reconstruct the simulated inputs
    2. real autoencoder   E_real, D_real reconstruct the real TacTip images
    3. latent maps        map_sr (and map_rs back) on the paired samples, both autoencoders
                          frozen: the mapped latent's error and the decoded image's error
    4. joint              everything together, a little: both reconstructions + both mapped
                          images, so the latent spaces settle where the maps work best

Every stage keeps its epoch with the lowest validation loss. Validation is on whole motions
training never saw (PairedData.split). Losses are "normalised MSE": the squared error over
the data's variance, so 1.0 is no better than always answering the average image and 0 is
perfect.

Backends:
    "torch"   convolutional autoencoders (an MLP one for markers) and MLP maps. Needs
              PyTorch (pip install torch). The real thing.
    "linear"  PCA autoencoders and ridge-regression maps, numpy only. Trains in seconds:
              a baseline, and a way to check the pipeline before a long run.

Use:
    from tactip_mujoco_gym.sim2real import load_mapping
    m = load_mapping("models/sim2real")
    real = m.sim_to_real(sim_image)        # simulated TacTip image (H, W, 3) RGB -> (h, w, 3)
    real = m(sim_images)                   # batches too, (N, H, W, 3) -> (N, h, w, 3)
    sim = m.real_to_sim(frame)             # a real camera frame as captured (BGR) -> the
                                           # simulated image (or markers, mm) it corresponds to

In the gym: sensors={"real_tactip": {"model": "models/sim2real"}} (sensors/real_tactip.py).

A trained mapping is a folder: mapping.json (preprocessing, network sizes, training
settings, validation scores), weights.pt (torch) or weights.npz (linear), and from training
preview.png (validation examples) and history.csv (losses per epoch).
"""
import csv
import datetime
import json
import os

import numpy as np

from .paired import Preprocess, draw_markers, grid, save_png, to_uint8

CONFIG = "mapping.json"
VERSION = 1

DEFAULTS = dict(
    latent_sim=32,          # size of the sim autoencoder's latent
    latent_real=64,         # size of the real autoencoder's latent
    base=32,                # channels of the first conv layer (doubling per layer, <= max_ch)
    max_ch=256,
    map_hidden=256,         # width of the latent maps' two hidden layers
    dropout=0.1,            # in the maps
    epochs_ae=30,           # each autoencoder
    epochs_map=60,
    epochs_joint=20,
    batch=64,
    lr=1e-3,
    joint_lr=3e-4,
    weight_decay=1e-4,
    latent_weight=0.1,      # stage 3: weight of the latent error next to the image error
    patience=10,            # stop a stage after this many epochs without a better val loss
    reverse=True,           # also learn real -> sim
    seed=0,
    max_fit=4000,           # linear backend: samples the PCA is fitted on
)


# ====================================================================== the mapping
class Sim2RealMapping:
    """A trained mapping: preprocessing + autoencoders + latent maps. Make one with
    train_mapping(), or load_mapping(path)."""

    def __init__(self, prep, backend, norm=None, info=None):
        self.prep = prep
        self.backend = backend
        self.norm = norm            # markers: {"mean": [...], "std": [...]} (mm)
        self.info = dict(info or {})

    def __repr__(self):
        p = self.prep
        src = (f"sim {p.sim_size}x{p.sim_size} image" if p.sim_input == "image"
               else f"sim {p.n_markers} markers")
        return (f"Sim2RealMapping({self.backend.name}: {src} <-> real {p.real_size[0]}x"
                f"{p.real_size[1]}{' grey' if p.gray else ''})")

    @property
    def sim_input(self):
        return self.prep.sim_input

    @property
    def real_shape(self):
        """(h, w, channels) of the real images it makes."""
        return self.prep.real_shape

    # ---------------------------------------------------------------- array plumbing
    def _sim_in(self, x):
        """Model-ready sim rows (uint8 images / mm) -> network floats."""
        if self.prep.sim_input == "image":
            return np.asarray(x, np.float32) / 255.0
        return ((np.asarray(x, np.float32) - self._mean) / self._std).astype(np.float32)

    def _sim_out(self, x):
        if self.prep.sim_input == "image":
            return to_uint8(np.asarray(x) * 255.0)
        mm = np.asarray(x, np.float32) * self._std + self._mean
        return mm.reshape(len(mm), -1, 3)

    @staticmethod
    def _real_in(y):
        return np.asarray(y, np.float32) / 255.0

    @staticmethod
    def _real_out(y):
        return to_uint8(np.asarray(y) * 255.0)

    @property
    def _mean(self):
        return np.asarray(self.norm["mean"], np.float32)

    @property
    def _std(self):
        return np.asarray(self.norm["std"], np.float32)

    def _batched(self, fn, x, n=256):
        if len(x) <= n:
            return fn(x)
        return np.concatenate([fn(x[i:i + n]) for i in range(0, len(x), n)])

    # ---------------------------------------------------------------- use
    def sim_to_real(self, sim):
        """Simulated TacTip image(s) (H, W, 3) RGB uint8 as rendered, any size; or marker
        displacements (n, 3) mm -> the real TacTip image(s) (h, w, C) uint8, RGB.
        One input in, one out; a batch (N, ...) in, (N, h, w, C) out."""
        x, single = self.prep.sim(sim)
        y = self._real_out(self._batched(self.backend.sim_to_real, self._sim_in(x)))
        return y[0] if single else y

    __call__ = sim_to_real

    def real_to_sim(self, frames, raw=True):
        """Real TacTip frame(s) as the camera gives them (BGR, full size; raw=False: already
        preprocessed like PairedData.real) -> the simulated image(s), or marker displacements
        (n, 3) mm, that match. Needs a mapping trained with reverse=True."""
        if not self.info.get("reverse", True):
            raise RuntimeError("this mapping was trained without the real -> sim direction")
        y, single = self.prep.real(frames, raw=raw)
        x = self._sim_out(self._batched(self.backend.real_to_sim, self._real_in(y)))
        return x[0] if single else x

    def reconstruct_real(self, frames, raw=True):
        """The real autoencoder's version of real frame(s) (how much detail it keeps)."""
        y, single = self.prep.real(frames, raw=raw)
        out = self._real_out(self._batched(self.backend.reconstruct_real, self._real_in(y)))
        return out[0] if single else out

    def reconstruct_sim(self, sim):
        x, single = self.prep.sim(sim)
        out = self._sim_out(self._batched(self.backend.reconstruct_sim, self._sim_in(x)))
        return out[0] if single else out

    def encode_sim(self, sim):
        """Latent code(s) of simulated input(s): (latent_sim,) or (N, latent_sim)."""
        x, single = self.prep.sim(sim)
        z = self._batched(self.backend.encode_sim, self._sim_in(x))
        return z[0] if single else z

    def encode_real(self, frames, raw=True):
        y, single = self.prep.real(frames, raw=raw)
        z = self._batched(self.backend.encode_real, self._real_in(y))
        return z[0] if single else z

    # ---------------------------------------------------------------- scores and pictures
    def evaluate(self, data, idx=None, max_n=2000, seed=0):
        """Scores on samples `idx` of PairedData `data` (normalised MSE: 1 = the average
        image, 0 = perfect; PSNR in dB; markers also as RMSE in mm)."""
        idx = np.arange(len(data)) if idx is None else np.asarray(idx)
        if len(idx) > max_n:
            idx = np.sort(np.random.default_rng(seed).choice(idx, max_n, replace=False))
        if not len(idx):
            return {}
        X = self._sim_in(data.sim[idx])
        Y = self._real_in(data.real[idx])
        b = self.backend

        def nmse(pred, true):
            return float(np.mean((pred - true) ** 2) / max(true.var(axis=0).mean(), 1e-12))
        out = {"samples": int(len(idx))}
        pred = self._batched(b.sim_to_real, X)
        out["sim2real_nmse"] = nmse(pred, Y)
        out["sim2real_psnr_db"] = float(10 * np.log10(1.0 / max(np.mean((pred - Y) ** 2), 1e-12)))
        out["real_rec_nmse"] = nmse(self._batched(b.reconstruct_real, Y), Y)
        rec = self._batched(b.reconstruct_sim, X)
        out["sim_rec_nmse"] = nmse(rec, X)
        back = self._batched(b.real_to_sim, Y) if self.info.get("reverse", True) else None
        if back is not None:
            out["real2sim_nmse"] = nmse(back, X)
        if self.prep.sim_input == "markers":
            mm = self._sim_out(X)
            out["sim_rec_rmse_mm"] = float(np.sqrt(np.mean((self._sim_out(rec) - mm) ** 2)))
            if back is not None:
                out["real2sim_rmse_mm"] = float(np.sqrt(np.mean((self._sim_out(back) - mm) ** 2)))
        return out

    def preview(self, data, idx=None, n=8, path=None, tile=96):
        """A picture of n samples, one per row: sim input | real | sim->real | real
        autoencoder | real->sim. Returns the RGB array (and writes a PNG if path)."""
        idx = np.arange(len(data)) if idx is None else np.asarray(idx)
        if not len(idx):
            return None
        touch = idx[data.phase[idx] == 1]
        pool = touch if len(touch) >= n else idx
        pick = pool[np.linspace(0, len(pool) - 1, min(n, len(pool))).astype(int)]
        X = self._sim_in(data.sim[pick])
        Y = self._real_in(data.real[pick])
        b = self.backend
        fwd = self._real_out(b.sim_to_real(X))
        rec = self._real_out(b.reconstruct_real(Y))
        rev = self._sim_out(b.real_to_sim(Y)) if self.info.get("reverse", True) else None

        def sim_pic(v):
            if self.prep.sim_input == "image":
                return v
            rest = self.info.get("marker_rest_mm")
            if rest is None:
                return np.full((tile, tile, 3), 230, np.uint8)
            return draw_markers(v, rest, size=tile)
        rows = []
        for k in range(len(pick)):
            r = [sim_pic(data.sim[pick[k]] if self.prep.sim_input == "image"
                         else self._sim_out(X[k:k + 1])[0]),
                 data.real[pick[k]], fwd[k], rec[k]]
            if rev is not None:
                r.append(sim_pic(rev[k]))
            rows.append(r)
        labels = ["sim", "real", "sim->real", "real AE"] + (["real->sim"] if rev is not None else [])
        pic = grid(rows, tile=tile, labels=labels)
        if path:
            save_png(path, pic)
        return pic

    # ---------------------------------------------------------------- save / load
    def config(self):
        return dict(version=VERSION, backend=self.backend.name, arch=self.backend.arch,
                    preprocess=self.prep.to_dict(), norm=self.norm, info=self.info)

    def save(self, path):
        """Write the folder (mapping.json + weights). Returns the folder."""
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, CONFIG), "w") as f:
            json.dump(_jsonable(self.config()), f, indent=1)
        self.backend.save(path)
        return path

    @classmethod
    def load(cls, path, device=None):
        cfg = read_config(path)
        kind = cfg.get("backend", "torch")
        if kind not in BACKENDS:
            raise ValueError(f"{path}: unknown backend {kind!r}")
        backend = BACKENDS[kind](arch=cfg["arch"], device=device)
        backend.load(cfg["dir"])
        m = cls(Preprocess.from_dict(cfg["preprocess"]), backend, norm=cfg.get("norm"),
                info=cfg.get("info"))
        m.path = cfg["dir"]
        return m


def read_config(path):
    """mapping.json of a saved mapping (path: its folder or the file), plus "dir"."""
    path = os.fspath(path)
    f = path if os.path.isfile(path) else os.path.join(path, CONFIG)
    if not os.path.isfile(f):
        raise FileNotFoundError(f"no trained sim2real mapping at {path} (expected {CONFIG}; train "
                                "one with python -m tactip_mujoco_gym.sim2real train-mapping)")
    with open(f) as fh:
        cfg = json.load(fh)
    cfg["dir"] = os.path.dirname(os.path.abspath(f))
    return cfg


def load_mapping(path, device=None):
    """Load a trained mapping: sim in -> map -> real out (and back).

        m = load_mapping("models/sim2real")
        real_image = m.sim_to_real(sim_image)

    device: torch device for the "torch" backend ("cpu", "cuda", ...; default: the GPU if
    there is one)."""
    return Sim2RealMapping.load(path, device=device)


def _jsonable(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return v


# ====================================================================== training
def train_mapping(data, out=None, backend="torch", val=0.15, split="motion", device=None,
                  log=print, **settings):
    """Train sim AE, real AE and the latent maps on PairedData `data`; save to `out`.

    val, split   fraction of the groups held out for validation, and what a group is:
                 "motion" (default), "trajectory", "texture" or "session" (paired.py)
    settings     any of DEFAULTS (latent sizes, epochs per stage, learning rates, ...)

    Returns the Sim2RealMapping; with `out`, it is saved there with preview.png (validation
    examples) and history.csv (losses per epoch)."""
    unknown = set(settings) - set(DEFAULTS)
    if unknown:
        raise TypeError(f"unknown training settings {sorted(unknown)}; known: {sorted(DEFAULTS)}")
    s = dict(DEFAULTS, **settings)
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {sorted(BACKENDS)}, not {backend!r}")
    tr, va = data.split(val, split, s["seed"])
    if not len(tr):
        raise ValueError("no training samples")
    prep = data.prep
    norm = None
    if prep.sim_input == "markers":
        x = np.asarray(data.sim[tr], np.float32)
        norm = {"mean": x.mean(axis=0), "std": np.maximum(x.std(axis=0), 1e-3)}
    arch = dict(sim_kind=prep.sim_input, sim_shape=list(prep.sim_shape),
                real_shape=list(prep.real_shape), latent_sim=int(s["latent_sim"]),
                latent_real=int(s["latent_real"]), base=int(s["base"]), max_ch=int(s["max_ch"]),
                map_hidden=int(s["map_hidden"]), dropout=float(s["dropout"]))
    info = dict(created=datetime.datetime.now().isoformat(timespec="seconds"),
                sessions=[os.path.abspath(p) for p in data.sessions], samples=len(data),
                train_samples=int(len(tr)), val_samples=int(len(va)), split=split, val=val,
                reverse=bool(s["reverse"]), settings=s,
                textures=sorted({int(t) for t in data.texture if t >= 0}),
                marker_rest_mm=None if data.marker_rest is None else data.marker_rest)
    m = Sim2RealMapping(prep, BACKENDS[backend](arch=arch, device=device), norm=norm, info=info)
    m.backend.build()
    log(f"[mapping] {data.summary()}")
    log(f"[mapping] {backend} backend{m.backend.where()}: train {len(tr)}, validate {len(va)} "
        f"(split by {split}); losses are normalised MSE (1 = the average image), train/val")
    rng = np.random.default_rng(s["seed"])
    sub = tr if len(tr) <= 2000 else rng.choice(tr, 2000, replace=False)
    var = dict(sim=float(m._sim_in(data.sim[sub]).var(axis=0).mean()) + 1e-8,
               real=float(m._real_in(data.real[sub]).var(axis=0).mean()) + 1e-8)
    history = m.backend.fit(data.sim, data.real, tr, va, m._sim_in, m._real_in, s, var, log)
    m.info["metrics"] = m.evaluate(data, va if len(va) else tr)
    m.info["metrics_on"] = "validation" if len(va) else "training (no validation samples)"
    if out:
        m.save(out)
        m.preview(data, va if len(va) else tr, path=os.path.join(out, "preview.png"))
        _write_history(os.path.join(out, "history.csv"), history)
        log(f"[mapping] saved {out}  ({CONFIG}, weights, preview.png, history.csv)")
    log("[mapping] " + describe_metrics(m.info["metrics"], "on " + m.info["metrics_on"]))
    return m


def describe_metrics(mt, on="on validation"):
    if not mt:
        return "no scores"
    parts = [f"sim->real {mt['sim2real_nmse']:.3f} ({mt['sim2real_psnr_db']:.1f} dB)",
             f"real AE {mt['real_rec_nmse']:.3f}", f"sim AE {mt['sim_rec_nmse']:.3f}"]
    if "real2sim_nmse" in mt:
        parts.append(f"real->sim {mt['real2sim_nmse']:.3f}")
    if "real2sim_rmse_mm" in mt:
        parts.append(f"real->sim markers {mt['real2sim_rmse_mm']:.3f} mm rms")
    return (f"{on} ({mt.get('samples', '?')} samples), normalised MSE (1 = the average image): "
            + ", ".join(parts))


def _write_history(path, history):
    if not history:
        return
    keys = list(dict.fromkeys(k for h in history for k in h))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for h in history:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in h.items()})


def _batches(idx, size, rng):
    idx = rng.permutation(idx)
    for i in range(0, len(idx), size):
        yield np.sort(idx[i:i + size])


# ====================================================================== linear backend
def _pca(X, k, rng, iters=3):
    """(mean, components (k, D)) of rows X; randomized SVD for large X. Directions the data
    doesn't vary along are left out (so k can come back smaller)."""
    mean = X.mean(axis=0)
    Xc = X - mean
    k = int(min(k, min(Xc.shape)))
    if min(Xc.shape) <= 1500:
        _, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    else:
        Q = Xc @ rng.standard_normal((Xc.shape[1], k + 10)).astype(np.float32)
        for _ in range(iters):
            Q, _ = np.linalg.qr(Q)
            Q = Xc @ (Xc.T @ Q)
        Q, _ = np.linalg.qr(Q)
        _, sv, vt = np.linalg.svd(Q.T @ Xc, full_matrices=False)
    k = max(1, min(k, int((sv > 1e-4 * max(sv[0], 1e-12)).sum())))
    return mean, vt[:k].astype(np.float32)


class LinearBackend:
    """PCA autoencoders + ridge-regression maps between the standardised latents."""
    name = "linear"

    def __init__(self, arch=None, device=None):
        self.arch = arch
        self.p = {}

    def where(self):
        return ""

    def build(self):
        pass

    # ---- pieces
    def _flat(self, x):
        return np.asarray(x, np.float32).reshape(len(x), -1)

    def _shape(self, x, which):
        return x.reshape(len(x), *self.arch[which])

    def encode_sim(self, x):
        return (self._flat(x) - self.p["sim_mean"]) @ self.p["sim_pc"].T

    def decode_sim(self, z):
        x = z @ self.p["sim_pc"] + self.p["sim_mean"]
        if self.arch["sim_kind"] == "image":
            x = np.clip(x, 0.0, 1.0)
        return self._shape(x, "sim_shape")

    def encode_real(self, y):
        return (self._flat(y) - self.p["real_mean"]) @ self.p["real_pc"].T

    def decode_real(self, z):
        return self._shape(np.clip(z @ self.p["real_pc"] + self.p["real_mean"], 0.0, 1.0), "real_shape")

    def _map(self, z, a, b):
        p = self.p
        return ((z - p[f"{a}_zm"]) / p[f"{a}_zs"]) @ p[f"W_{a}{b}"] * p[f"{b}_zs"] + p[f"{b}_zm"]

    def map_sr(self, z):
        return self._map(z, "sim", "real")

    def map_rs(self, z):
        return self._map(z, "real", "sim")

    def sim_to_real(self, x):
        return self.decode_real(self.map_sr(self.encode_sim(x)))

    def real_to_sim(self, y):
        return self.decode_sim(self.map_rs(self.encode_real(y)))

    def reconstruct_sim(self, x):
        return self.decode_sim(self.encode_sim(x))

    def reconstruct_real(self, y):
        return self.decode_real(self.encode_real(y))

    # ---- training
    def fit(self, sim, real, tr, va, sim_in, real_in, s, var, log):
        rng = np.random.default_rng(s["seed"])
        a = self.arch
        fit_idx = tr if len(tr) <= s["max_fit"] else np.sort(rng.choice(tr, s["max_fit"], replace=False))
        if len(fit_idx) < len(tr):
            log(f"[linear] PCA on {len(fit_idx)} of the {len(tr)} training samples (max_fit)")
        X, Y = self._flat(sim_in(sim[fit_idx])), self._flat(real_in(real[fit_idx]))
        self.p["sim_mean"], self.p["sim_pc"] = _pca(X, a["latent_sim"], rng)
        self.p["real_mean"], self.p["real_pc"] = _pca(Y, a["latent_real"], rng)
        a["latent_sim"], a["latent_real"] = len(self.p["sim_pc"]), len(self.p["real_pc"])
        Zs, Zr = self.encode_sim(X), self.encode_real(Y)
        for name, Z in (("sim", Zs), ("real", Zr)):
            self.p[f"{name}_zm"] = Z.mean(axis=0)
            sd = Z.std(axis=0)
            self.p[f"{name}_zs"] = np.maximum(sd, 1e-2 * sd.max()) + 1e-8
        vidx = va if len(va) <= s["max_fit"] else np.sort(rng.choice(va, s["max_fit"], replace=False))
        Xv, Yv = (sim_in(sim[vidx]), real_in(real[vidx])) if len(vidx) else (sim_in(sim[fit_idx]), real_in(real[fit_idx]))
        history = []

        def nmse(p, t, v):
            return float(np.mean((p - t) ** 2) / v)
        history.append(dict(stage="sim_ae", epoch=1, val_sim_rec=nmse(self.reconstruct_sim(Xv), Xv, var["sim"])))
        history.append(dict(stage="real_ae", epoch=1, val_real_rec=nmse(self.reconstruct_real(Yv), Yv, var["real"])))
        log(f"[linear] PCA: sim {a['latent_sim']} components (val {history[0]['val_sim_rec']:.4f}), "
            f"real {a['latent_real']} (val {history[1]['val_real_rec']:.4f})")
        n = len(Zs)
        S = (Zs - self.p["sim_zm"]) / self.p["sim_zs"]
        R = (Zr - self.p["real_zm"]) / self.p["real_zs"]
        pairs = [("sim", "real", S, R, lambda: self.sim_to_real(Xv), Yv, var["real"], "sim2real")]
        if s["reverse"]:
            pairs.append(("real", "sim", R, S, lambda: self.real_to_sim(Yv), Xv, var["sim"], "real2sim"))
        for src, dst, A, B, predict, target, v, label in pairs:
            best = None
            for ridge in (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0):
                self.p[f"W_{src}{dst}"] = np.linalg.solve(A.T @ A + ridge * n * np.eye(A.shape[1]),
                                                          A.T @ B).astype(np.float32)
                err = nmse(predict(), target, v)
                history.append({"stage": "map", "epoch": len(history), "ridge": ridge, f"val_{label}": err})
                if best is None or err < best[0]:
                    best = (err, ridge, self.p[f"W_{src}{dst}"])
            self.p[f"W_{src}{dst}"] = best[2]
            log(f"[linear] {src} -> {dst}: ridge {best[1]:g}, val {best[0]:.4f}")
        return history

    # ---- save / load
    def save(self, path):
        np.savez(os.path.join(path, "weights.npz"), **self.p)

    def load(self, path):
        with np.load(os.path.join(path, "weights.npz")) as z:
            self.p = {k: z[k] for k in z.files}


# ====================================================================== torch backend
def _torch():
    try:
        import torch
    except ImportError as err:
        raise ImportError("the 'torch' backend needs PyTorch: pip install torch "
                          "(or train with --backend linear)") from err
    return torch


def _pick_device(torch, device=None):
    if device not in (None, "auto"):
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def n_down(h, w):
    """How many stride-2 layers an h x w image gets (down to 4-7 px, at most 5)."""
    n = 0
    while n < 5 and h % 2 == 0 and w % 2 == 0 and min(h, w) // 2 >= 4:
        h, w, n = h // 2, w // 2, n + 1
    return n


def check_image_size(w, h, what="image"):
    if n_down(h, w) < 2:
        raise ValueError(f"{what} {w} x {h}: use sizes divisible by 8 (e.g. 64 or 128) so the "
                         "autoencoder can halve them a few times")


def build_networks(arch):
    """The torch modules for an architecture (the "arch" dict saved in mapping.json):
    enc_sim, dec_sim, enc_real, dec_real, map_sr, map_rs."""
    torch = _torch()
    nn = torch.nn
    base, max_ch = int(arch["base"]), int(arch["max_ch"])
    if base % 8 or max_ch % 8:
        raise ValueError("base and max_ch channels must be multiples of 8")

    def mlp(sizes, dropout=0.0):
        layers = []
        for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
            layers.append(nn.Linear(a, b))
            if i < len(sizes) - 2:
                layers.append(nn.SiLU())
                if dropout:
                    layers.append(nn.Dropout(dropout))
        return nn.Sequential(*layers)

    def chans(n):
        return [min(base * 2 ** i, max_ch) for i in range(n)]

    class ConvEncoder(nn.Module):
        def __init__(self, shape, latent):
            super().__init__()
            h, w, c = shape
            n = n_down(h, w)
            layers, ch = [], c
            for out in chans(n):
                layers += [nn.Conv2d(ch, out, 4, 2, 1), nn.GroupNorm(8, out), nn.LeakyReLU(0.2)]
                ch = out
            self.conv = nn.Sequential(*layers)
            self.fc = nn.Linear(ch * (h >> n) * (w >> n), latent)

        def forward(self, x):                       # (B, C, H, W) in [0, 1]
            return self.fc(self.conv(x * 2.0 - 1.0).flatten(1))

    class ConvDecoder(nn.Module):
        def __init__(self, shape, latent):
            super().__init__()
            h, w, c = shape
            n = n_down(h, w)
            ch = chans(n)
            self.start = (ch[-1], h >> n, w >> n)
            self.fc = nn.Linear(latent, ch[-1] * (h >> n) * (w >> n))
            layers = [nn.LeakyReLU(0.2)]
            for i in reversed(range(n)):
                out = ch[i - 1] if i > 0 else base
                layers += [nn.ConvTranspose2d(ch[i], out, 4, 2, 1), nn.GroupNorm(8, out),
                           nn.LeakyReLU(0.2)]
            layers += [nn.Conv2d(base, c, 3, 1, 1), nn.Sigmoid()]
            self.net = nn.Sequential(*layers)

        def forward(self, z):                       # -> (B, C, H, W) in [0, 1]
            return self.net(self.fc(z).view(-1, *self.start))

    class Networks(nn.Module):
        def __init__(self):
            super().__init__()
            ls, lr, hid = int(arch["latent_sim"]), int(arch["latent_real"]), int(arch["map_hidden"])
            if arch["sim_kind"] == "image":
                self.enc_sim = ConvEncoder(arch["sim_shape"], ls)
                self.dec_sim = ConvDecoder(arch["sim_shape"], ls)
            else:
                d = int(arch["sim_shape"][0])
                self.enc_sim = mlp([d, 256, 128, ls])
                self.dec_sim = mlp([ls, 128, 256, d])
            self.enc_real = ConvEncoder(arch["real_shape"], lr)
            self.dec_real = ConvDecoder(arch["real_shape"], lr)
            self.map_sr = mlp([ls, hid, hid, lr], float(arch["dropout"]))
            self.map_rs = mlp([lr, hid, hid, ls], float(arch["dropout"]))

    return Networks()


class TorchBackend:
    """Conv autoencoders (MLP for markers) + MLP latent maps, in PyTorch."""
    name = "torch"

    def __init__(self, arch=None, device=None):
        self.arch = arch
        self.device_name = device
        self.net = None

    def build(self):
        self.torch = _torch()
        for key in ("sim_shape", "real_shape"):
            if len(self.arch[key]) == 3:
                h, w, _ = self.arch[key]
                check_image_size(w, h, key.split("_")[0] + " image")
        self.device = _pick_device(self.torch, self.device_name)
        self.net = build_networks(self.arch).to(self.device)
        self.net.eval()

    def where(self):
        n = sum(p.numel() for p in self.net.parameters()) if self.net is not None else 0
        return f" on {self.device} ({n / 1e6:.2f} M parameters)"

    # ---- numpy <-> torch
    def _t(self, x):
        t = self.torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).to(self.device)
        return t.permute(0, 3, 1, 2).contiguous() if t.ndim == 4 else t

    @staticmethod
    def _np(t):
        if t.ndim == 4:
            t = t.permute(0, 2, 3, 1)
        return t.detach().float().cpu().numpy()

    def _run(self, fn, x):
        self.net.eval()
        with self.torch.no_grad():
            return self._np(fn(self._t(x)))

    # ---- pieces (numpy in, numpy out)
    def encode_sim(self, x):
        return self._run(self.net.enc_sim, x)

    def decode_sim(self, z):
        return self._run(self.net.dec_sim, z)

    def encode_real(self, y):
        return self._run(self.net.enc_real, y)

    def decode_real(self, z):
        return self._run(self.net.dec_real, z)

    def sim_to_real(self, x):
        n = self.net
        return self._run(lambda t: n.dec_real(n.map_sr(n.enc_sim(t))), x)

    def real_to_sim(self, y):
        n = self.net
        return self._run(lambda t: n.dec_sim(n.map_rs(n.enc_real(t))), y)

    def reconstruct_sim(self, x):
        n = self.net
        return self._run(lambda t: n.dec_sim(n.enc_sim(t)), x)

    def reconstruct_real(self, y):
        n = self.net
        return self._run(lambda t: n.dec_real(n.enc_real(t)), y)

    # ---- training
    def fit(self, sim, real, tr, va, sim_in, real_in, s, var, log):
        torch, net = self.torch, self.net
        mse = torch.nn.functional.mse_loss
        rng = np.random.default_rng(s["seed"])
        torch.manual_seed(int(s["seed"]))
        vs, vr = var["sim"], var["real"]
        history = []

        def batch(idx):
            return self._t(sim_in(sim[idx])), self._t(real_in(real[idx]))

        def trainable(modules, on):
            for mod in modules:
                for p in mod.parameters():
                    p.requires_grad_(on)

        def loss_sim_ae(x, y):
            r = mse(net.dec_sim(net.enc_sim(x)), x) / vs
            return r, {"sim_rec": r}

        def loss_real_ae(x, y):
            r = mse(net.dec_real(net.enc_real(y)), y) / vr
            return r, {"real_rec": r}

        def loss_map(x, y):
            with torch.no_grad():
                zs, zr = net.enc_sim(x), net.enc_real(y)
            to_r = net.map_sr(zs)
            fwd = mse(net.dec_real(to_r), y) / vr
            lat = mse(to_r, zr) / (zr.var() + 1e-6)
            total, parts = fwd + s["latent_weight"] * lat, {"sim2real": fwd, "latent": lat}
            if s["reverse"]:
                to_s = net.map_rs(zr)
                back = mse(net.dec_sim(to_s), x) / vs
                total = total + back + s["latent_weight"] * mse(to_s, zs) / (zs.var() + 1e-6)
                parts["real2sim"] = back
            return total, parts

        def loss_joint(x, y):
            zs, zr = net.enc_sim(x), net.enc_real(y)
            rs = mse(net.dec_sim(zs), x) / vs
            rr = mse(net.dec_real(zr), y) / vr
            fwd = mse(net.dec_real(net.map_sr(zs)), y) / vr
            total = rs + rr + fwd
            parts = {"sim_rec": rs, "real_rec": rr, "sim2real": fwd}
            if s["reverse"]:
                back = mse(net.dec_sim(net.map_rs(zr)), x) / vs
                total = total + back
                parts["real2sim"] = back
            return total, parts

        def evaluate(loss_fn, idx):
            net.eval()
            tot, n = {}, 0
            with torch.no_grad():
                for i in range(0, len(idx), s["batch"]):
                    b = idx[i:i + s["batch"]]
                    _, parts = loss_fn(*batch(b))
                    for k, v in parts.items():
                        tot[k] = tot.get(k, 0.0) + float(v) * len(b)
                    n += len(b)
            return {k: v / n for k, v in tot.items()}

        def stage(name, modules, loss_fn, epochs, lr, select):
            epochs = int(epochs)
            if epochs <= 0:
                return
            trainable(net.children(), False)
            trainable(modules, True)
            params = [p for mod in modules for p in mod.parameters()]
            opt = torch.optim.AdamW(params, lr=lr, weight_decay=s["weight_decay"])
            best, best_state, bad = np.inf, None, 0
            for ep in range(1, epochs + 1):
                net.train()
                tot, n = {}, 0
                for idx in _batches(tr, int(s["batch"]), rng):
                    loss, parts = loss_fn(*batch(idx))
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                    for k, v in parts.items():
                        tot[k] = tot.get(k, 0.0) + float(v) * len(idx)
                    n += len(idx)
                train = {k: v / n for k, v in tot.items()}
                val = evaluate(loss_fn, va) if len(va) else train
                score = sum(val[k] for k in select)
                row = {"stage": name, "epoch": ep}
                row.update({f"train_{k}": v for k, v in train.items()})
                row.update({f"val_{k}": v for k, v in val.items()})
                history.append(row)
                mark = ""
                if score < best - 1e-7:
                    best, bad, mark = score, 0, "  *"
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                log(f"[{name}] {ep}/{epochs}  " +
                    "  ".join(f"{k} {train[k]:.4f}/{val[k]:.4f}" for k in train) + mark)
                if s["patience"] and bad >= int(s["patience"]):
                    log(f"[{name}] no better for {bad} epochs: next stage")
                    break
            if best_state is not None:
                net.load_state_dict(best_state)
            trainable(net.children(), True)

        lr = float(s["lr"])
        stage("sim_ae", [net.enc_sim, net.dec_sim], loss_sim_ae, s["epochs_ae"], lr, ["sim_rec"])
        stage("real_ae", [net.enc_real, net.dec_real], loss_real_ae, s["epochs_ae"], lr, ["real_rec"])
        maps = [net.map_sr] + ([net.map_rs] if s["reverse"] else [])
        stage("map", maps, loss_map, s["epochs_map"], lr,
              ["sim2real"] + (["real2sim"] if s["reverse"] else []))
        everything = [net.enc_sim, net.dec_sim, net.enc_real, net.dec_real] + maps
        stage("joint", everything, loss_joint, s["epochs_joint"], float(s["joint_lr"]),
              ["sim_rec", "real_rec", "sim2real"] + (["real2sim"] if s["reverse"] else []))
        net.eval()
        return history

    # ---- save / load
    def save(self, path):
        self.torch.save(self.net.state_dict(), os.path.join(path, "weights.pt"))

    def load(self, path):
        self.build()
        f = os.path.join(path, "weights.pt")
        try:
            state = self.torch.load(f, map_location=self.device, weights_only=True)
        except TypeError:                           # torch < 1.13: no weights_only
            state = self.torch.load(f, map_location=self.device)
        self.net.load_state_dict(state)
        self.net.eval()


BACKENDS = {"torch": TorchBackend, "linear": LinearBackend}
