"""
Paired sim / real TacTip data for learning the image mapping (mapping.py).

Reads sessions recorded with rig_dataset (Rig-controller repo): every sample there is the
same pose on both rigs, so sample k gives (simulated TacTip, real TacTip). Nothing here
needs rig_dataset installed: a session is a folder with meta.json and one .npz per
trajectory.

    data = PairedData.load(["data/session_20261113_140000"], sim_input="image")
    train, val = data.split(val=0.15, by="motion")
    data.sim[train], data.real[train]       # model-ready uint8 images (markers: float mm)

The same preprocessing (Preprocess) is saved with a trained mapping and applied again when
it is used, in the gym (sensors/real_tactip.py) or on live camera frames:

    sim image   the simulated TacTip camera image (RGB, any size) -> area-resized to
                sim_size x sim_size (grey if gray=True)
    sim markers the simulated markers' displacements (n, 3) mm -> one row of 3n numbers
    real image  the real TacTip frame as the camera gives it (BGR, as OpenCV and the
                RealSense store it) -> cropped (default: the centre square) -> area-resized
                to real_size, RGB (grey if gray=True)

Only numpy is used, so the resizing is identical on every machine.
"""
import json
import os
import struct
import zlib

import numpy as np

GRAY = np.array([0.299, 0.587, 0.114], np.float32)          # as sensors.TacTipCamera(mode="gray")
PHASE_CODES = {"approach": 0, "contact": 1, "retract": 2}


# ====================================================================== image helpers
_WEIGHTS = {}


def _area_weights(n_in, n_out):
    """(n_out, n_in) matrix averaging the input pixels each output pixel covers."""
    key = (int(n_in), int(n_out))
    if key not in _WEIGHTS:
        W = np.zeros((n_out, n_in), np.float32)
        scale = n_in / n_out
        for i in range(n_out):
            a, b = i * scale, (i + 1) * scale
            for j in range(int(np.floor(a)), min(int(np.ceil(b)), n_in)):
                W[i, j] = min(b, j + 1) - max(a, j)
            W[i] /= W[i].sum()
        _WEIGHTS[key] = W
    return _WEIGHTS[key]


def resize_area(img, size):
    """Area-average resize of (H, W[, C]) or (N, H, W, C) to size = (w, h); float32 out."""
    img = np.asarray(img)
    w, h = int(size[0]), int(size[1])
    single = img.ndim < 4
    x = img[None] if single else img
    if x.ndim == 3:                                          # (N, H, W) -> (N, H, W, 1)
        x = x[..., None]
    x = x.astype(np.float32, copy=False)
    H, W = x.shape[1:3]
    if (H, W) != (h, w):
        x = np.tensordot(_area_weights(H, h), x, axes=([1], [1])).transpose(1, 0, 2, 3)
        x = np.tensordot(_area_weights(W, w), x, axes=([1], [2])).transpose(1, 2, 0, 3)
    if img.ndim == 2:
        x = x[..., 0]
    return x[0] if single else x


def to_uint8(x):
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def to_gray(rgb):
    """(..., 3) RGB -> (..., 1)."""
    return (np.asarray(rgb, np.float32) @ GRAY)[..., None]


def save_png(path, img):
    """Write a uint8 (H, W), (H, W, 1) or (H, W, 3) image as PNG (no imaging library needed)."""
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = to_uint8(img)
    if img.ndim == 3 and img.shape[2] == 1:
        img = img[..., 0]
    h, w = img.shape[:2]
    ctype = 0 if img.ndim == 2 else {3: 2, 4: 6}[img.shape[2]]
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))

    def chunk(t, data):
        return (struct.pack(">I", len(data)) + t + data +
                struct.pack(">I", zlib.crc32(t + data) & 0xFFFFFFFF))
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ctype, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))
    return path


def _rgb(img):
    img = np.asarray(img)
    if img.ndim == 2:
        img = img[..., None]
    return np.repeat(img, 3, axis=2) if img.shape[2] == 1 else img[..., :3]


def draw_markers(disp_mm, rest_mm, size=96, scale=4.0):
    """A picture of marker displacements: rest positions (grey) and where each marker is,
    its displacement drawn `scale` times larger, coloured by the push into the skin."""
    rest = np.asarray(rest_mm, float).reshape(-1, 3)
    disp = np.asarray(disp_mm, float).reshape(-1, 3)
    c = rest.mean(axis=0)
    _, _, vt = np.linalg.svd(rest - c, full_matrices=False)  # in-plane axes, then the normal
    p0 = (rest - c) @ vt.T
    p1 = (rest - c + scale * disp) @ vt.T
    span = max(np.abs(p0[:, :2]).max(), 1e-6) * 1.15
    canvas = np.full((size, size, 3), 255, np.uint8)

    def px(p):
        return np.clip(np.rint((p[:, :2] / span * 0.5 + 0.5) * (size - 1)), 0, size - 1).astype(int)
    for (u, v) in px(p0):
        canvas[max(v - 1, 0):v + 1, max(u - 1, 0):u + 1] = 190
    push = disp @ vt[2]
    m = max(np.abs(push).max(), 1e-6)
    for (u, v), z in zip(px(p1), push):
        t = 0.5 + 0.5 * z / m
        canvas[max(v - 1, 0):v + 2, max(u - 1, 0):u + 2] = (int(255 * t), 40, int(255 * (1 - t)))
    return canvas


def grid(rows, tile=96, pad=2, labels=None):
    """Rows of images (any size, grey or RGB) -> one RGB uint8 picture, `tile` px squares."""
    rows = [[_rgb(im) for im in r] for r in rows]
    ncol = max(len(r) for r in rows)
    top = 14 if labels else 0
    out = np.full((top + len(rows) * (tile + pad) + pad, ncol * (tile + pad) + pad, 3), 255, np.uint8)
    for i, r in enumerate(rows):
        for j, im in enumerate(r):
            t = to_uint8(resize_area(im, (tile, tile))) if im.shape[:2] != (tile, tile) else im
            y, x = top + pad + i * (tile + pad), pad + j * (tile + pad)
            out[y:y + tile, x:x + tile] = t
    if labels:
        try:
            import cv2
            for j, s in enumerate(labels):
                cv2.putText(out, s, (pad + j * (tile + pad) + 2, 11), cv2.FONT_HERSHEY_SIMPLEX,
                            0.33, (0, 0, 0), 1, cv2.LINE_AA)
        except ImportError:
            pass
    return out


def _chunked(fn, x, n=32):
    """fn over a batch, n at a time (keeps float copies of large frames small)."""
    if len(x) <= n:
        return fn(x)
    return np.concatenate([fn(x[i:i + n]) for i in range(0, len(x), n)])


# ====================================================================== preprocessing
class Preprocess:
    """How simulated inputs and real frames become the arrays the mapping sees (saved with
    the model, so training and use match exactly).

    sim_input    "image" (the simulated TacTip camera) or "markers" (marker displacements)
    sim_size     side of the square sim image the model sees (px)
    sim_render   size the simulated camera renders at before resizing (the dataset's; the
                 gym readout renders at this size)
    n_markers    markers per sensor (sim_input="markers")
    real_size    (w, h) of the real image the model sees and outputs
    real_crop    (x, y, w, h) of the camera frame to keep; None: the centre square;
                 "full": the whole frame (squashed to real_size)
    real_frame   (w, h) of the camera frames the crop refers to (another size scales it)
    real_bgr     camera frames are BGR (OpenCV, RealSense bgr8): converted to RGB
    gray         one grey channel instead of RGB, for both
    """

    def __init__(self, sim_input="image", sim_size=64, sim_render=None, n_markers=None,
                 real_size=(64, 64), real_crop=None, real_frame=None, real_bgr=True, gray=False):
        if sim_input not in ("image", "markers"):
            raise ValueError(f"sim_input must be 'image' or 'markers', not {sim_input!r}")
        self.sim_input = sim_input
        self.sim_size = int(sim_size)
        self.sim_render = None if sim_render is None else int(sim_render)
        self.n_markers = None if n_markers is None else int(n_markers)
        self.real_size = (int(real_size[0]), int(real_size[1]))
        self.real_crop = (real_crop if real_crop in (None, "full") else
                          tuple(int(v) for v in real_crop))
        self.real_frame = None if real_frame is None else (int(real_frame[0]), int(real_frame[1]))
        self.real_bgr = bool(real_bgr)
        self.gray = bool(gray)

    @property
    def channels(self):
        return 1 if self.gray else 3

    @property
    def sim_shape(self):
        if self.sim_input == "image":
            return (self.sim_size, self.sim_size, self.channels)
        return (3 * (self.n_markers or 0),)

    @property
    def real_shape(self):
        return (self.real_size[1], self.real_size[0], self.channels)

    def to_dict(self):
        return dict(sim_input=self.sim_input, sim_size=self.sim_size, sim_render=self.sim_render,
                    n_markers=self.n_markers, real_size=list(self.real_size),
                    real_crop=(self.real_crop if self.real_crop in (None, "full")
                               else list(self.real_crop)),
                    real_frame=None if self.real_frame is None else list(self.real_frame),
                    real_bgr=self.real_bgr, gray=self.gray)

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("sim_input", "sim_size", "sim_render", "n_markers",
                                        "real_size", "real_crop", "real_frame", "real_bgr",
                                        "gray") if k in d})

    # ---- sim
    def sim(self, x):
        """Simulated input(s) -> model-ready rows: uint8 (N, s, s, C) images, or float32
        (N, 3n) marker displacements in mm. Returns (array, was_single)."""
        x = np.asarray(x)
        if self.sim_input == "image":
            single = x.ndim == 3
            x = x[None] if single else x
            if x.shape[-1] not in (1, 3):
                raise ValueError(f"a simulated TacTip image is (H, W, 3); got {x.shape[1:]}")
            return _chunked(self._sim_images, x), single
        single = x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 3 and
                                 (self.n_markers is None or x.shape[0] == self.n_markers))
        x = x.reshape(1 if single else len(x), -1).astype(np.float32)
        if self.n_markers and x.shape[1] != 3 * self.n_markers:
            raise ValueError(f"expected {self.n_markers} markers (x, y, z), got {x.shape[1] / 3:g}")
        return x, single

    def _sim_images(self, x):
        if x.shape[1:3] != (self.sim_size, self.sim_size):
            x = resize_area(x, (self.sim_size, self.sim_size))
        if self.gray and x.shape[-1] == 3:
            x = to_gray(x)
        elif not self.gray and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return to_uint8(x) if x.dtype != np.uint8 else x

    # ---- real
    def crop_box(self, frame_hw):
        """(x, y, w, h) of a camera frame of size (H, W) to keep."""
        H, W = int(frame_hw[0]), int(frame_hw[1])
        if self.real_crop == "full":
            return (0, 0, W, H)
        if self.real_crop is None:
            s = min(H, W)
            return ((W - s) // 2, (H - s) // 2, s, s)
        x, y, w, h = self.real_crop
        if self.real_frame and tuple(self.real_frame) != (W, H):       # another frame size
            fx, fy = W / self.real_frame[0], H / self.real_frame[1]
            x, y, w, h = int(round(x * fx)), int(round(y * fy)), int(round(w * fx)), int(round(h * fy))
        x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
        return (x, y, max(1, min(w, W - x)), max(1, min(h, H - y)))

    def real(self, frames, raw=True):
        """Real TacTip frame(s) -> uint8 (N, h, w, C) RGB / grey. raw=False: already
        preprocessed (the arrays this class makes), passed through. Returns (array, single)."""
        x = np.asarray(frames)
        single = x.ndim == 2 or (x.ndim == 3 and x.shape[-1] in (1, 3, 4) and x.shape[0] > 4)
        if x.ndim == 2:
            x = x[..., None]
        x = x[None] if single else x
        if x.ndim == 3:                                       # (N, H, W) grey frames
            x = x[..., None]
        if not raw:
            return x.astype(np.uint8, copy=False), single
        if self.real_frame is None:
            self.real_frame = (x.shape[2], x.shape[1])
        if self.real_crop in (None, "full"):            # fixed by the first frames seen
            self.real_crop = self.crop_box(x.shape[1:3])
        cx, cy, cw, ch = self.crop_box(x.shape[1:3])
        return _chunked(self._real_frames, x[:, cy:cy + ch, cx:cx + cw, :3]), single

    def _real_frames(self, x):
        if x.shape[-1] == 3 and self.real_bgr:
            x = x[..., ::-1]
        x = resize_area(x, self.real_size)
        if self.gray and x.shape[-1] == 3:
            x = to_gray(x)
        elif not self.gray and x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=-1)
        return to_uint8(x)


# ====================================================================== sessions
def find_sessions(paths):
    """Session folders (with meta.json) from session folders or folders holding them."""
    out = []
    for p in ([paths] if isinstance(paths, (str, os.PathLike)) else paths):
        p = os.fspath(p)
        if os.path.isfile(os.path.join(p, "meta.json")):
            out.append(p)
        elif os.path.isdir(p):
            subs = sorted(os.path.join(p, d) for d in os.listdir(p)
                          if os.path.isfile(os.path.join(p, d, "meta.json")))
            if not subs:
                raise FileNotFoundError(f"{p}: no recorded sessions in it (a session folder "
                                        "has meta.json; record one with python -m rig_dataset collect)")
            out += subs
        else:
            raise FileNotFoundError(f"{p}: no such folder")
    return out


class PairedData:
    """Paired samples from one or more sessions, preprocessed.

    sim      uint8 (N, s, s, C) simulated images, or float32 (N, 3n) markers (mm)
    real     uint8 (N, h, w, C) real images
    group    (N,) int, the unit kept together when splitting (see split)
    phase, texture, step   per sample; rows: one dict per trajectory
    """

    def __init__(self, sim, real, prep, rows, traj, phase, texture, step, sessions,
                 marker_rest=None):
        self.sim, self.real, self.prep = sim, real, prep
        self.rows, self.traj = rows, traj
        self.phase, self.texture, self.step = phase, texture, step
        self.sessions = sessions
        self.marker_rest = marker_rest

    def __len__(self):
        return len(self.real)

    @classmethod
    def load(cls, paths, sim_input="image", sim_size=64, real_size=(64, 64), real_crop=None,
             gray=False, real_bgr=True, phases=None, textures=None, every=1, max_samples=None,
             prep=None, log=print):
        """Read sessions (folders, or folders of them). phases: e.g. ["contact"] (default
        all, so the model also learns the untouched sensor); textures: numbers to keep;
        every: keep every k-th sample along a trajectory; prep: a Preprocess to use instead
        of the settings before it (a trained mapping's, to test it on new sessions)."""
        sessions = find_sessions(paths)
        if prep is not None:
            prep = Preprocess.from_dict(prep.to_dict())
            sim_input = prep.sim_input
        else:
            prep = Preprocess(sim_input=sim_input, sim_size=sim_size, real_size=real_size,
                              real_crop=real_crop, real_bgr=real_bgr, gray=gray)
        want = None if not phases else {PHASE_CODES.get(p, p) for p in phases}
        sims, reals, rows, traj, phase, tex, step = [], [], [], [], [], [], []
        renders, marker_rest, skipped = [], None, {}
        sim_key = "sim_tactip" if sim_input == "image" else "sim_markers"
        for si, sdir in enumerate(sessions):
            with open(os.path.join(sdir, "meta.json")) as f:
                meta = json.load(f)
            simmeta = meta.get("sim") or {}
            if simmeta.get("image_size"):
                renders.append(int(simmeta["image_size"]))
            if marker_rest is None and simmeta.get("marker_rest_mm") is not None:
                marker_rest = np.asarray(simmeta["marker_rest_mm"], np.float32)
            n_before = sum(len(r) for r in reals)
            for row in meta.get("trajectories", []):
                if textures is not None and row.get("texture") not in textures:
                    continue
                path = os.path.join(sdir, row["file"])
                if not os.path.exists(path):
                    skipped["file missing"] = skipped.get("file missing", 0) + 1
                    continue
                with np.load(path) as z:
                    if sim_key not in z.files or "real_tactip" not in z.files:
                        why = f"no {sim_key if sim_key not in z.files else 'real_tactip'}"
                        skipped[why] = skipped.get(why, 0) + 1
                        continue
                    s, r = z[sim_key], z["real_tactip"]
                    ph = z["phase"] if "phase" in z.files else np.full(len(r), -1, np.int8)
                keep = np.arange(len(r))[::max(1, int(every))]
                if want is not None:
                    keep = keep[np.isin(ph[keep], list(want))]
                if not len(keep):
                    continue
                if sim_input == "markers":
                    if prep.n_markers is None:
                        prep.n_markers = s.shape[1]
                    elif s.shape[1] != prep.n_markers:
                        raise ValueError(f"{path}: {s.shape[1]} markers, earlier sessions have "
                                         f"{prep.n_markers}; train on one sensor model")
                sims.append(prep.sim(s[keep])[0])
                reals.append(prep.real(r[keep])[0])
                k = len(rows)
                rows.append(dict(row, session=sdir, session_index=si))
                traj.append(np.full(len(keep), k, np.int32))
                phase.append(ph[keep].astype(np.int8))
                tex.append(np.full(len(keep), int(row.get("texture", -1)) if
                                   str(row.get("texture", "")).lstrip("-").isdigit() else -1, np.int16))
                step.append(keep.astype(np.int32))
                if max_samples and sum(len(v) for v in reals) >= max_samples:
                    break
            n = sum(len(r) for r in reals) - n_before
            log(f"[mapping] {sdir}: {n} samples")
            if max_samples and sum(len(v) for v in reals) >= max_samples:
                break
        for why, n in skipped.items():
            log(f"[mapping] skipped {n} trajectories: {why}")
        if not reals:
            raise ValueError("no paired samples found (the sessions need real_tactip and "
                             f"{sim_key}; record with the real rig, not --sim-only)")
        if renders and prep.sim_render is None:
            prep.sim_render = max(set(renders), key=renders.count)
            if len(set(renders)) > 1:
                log(f"[mapping] sessions rendered the sim at {sorted(set(renders))} px; all are "
                    f"resized to {prep.sim_size} (the gym will render at {prep.sim_render})")
        data = cls(np.concatenate(sims), np.concatenate(reals), prep, rows, np.concatenate(traj),
                   np.concatenate(phase), np.concatenate(tex), np.concatenate(step), sessions,
                   marker_rest)
        if max_samples and len(data) > max_samples:
            data = data.subset(np.arange(max_samples))
        return data

    def subset(self, idx):
        idx = np.asarray(idx)
        return PairedData(self.sim[idx], self.real[idx], self.prep, self.rows, self.traj[idx],
                          self.phase[idx], self.texture[idx], self.step[idx], self.sessions,
                          self.marker_rest)

    def groups(self, by="motion"):
        """(N,) group id per sample. by: "motion" (every repeat of one motion on one
        texture together, so the validation motions are new ones), "trajectory", "texture"
        (held-out textures) or "session"."""
        keys = []
        for r in self.rows:
            if by == "motion":
                m = r.get("motion")
                keys.append((r["session_index"], str(r.get("texture")), "m", str(m))
                            if m not in (None, "") else (r["session_index"], r["file"]))
            elif by == "trajectory":
                keys.append((r["session_index"], r["file"]))
            elif by == "texture":
                keys.append(str(r.get("texture")))
            elif by == "session":
                keys.append(r["session_index"])
            else:
                raise ValueError(f"split by 'motion', 'trajectory', 'texture' or 'session', not {by!r}")
        ids = {k: i for i, k in enumerate(dict.fromkeys(keys))}
        per_row = np.array([ids[k] for k in keys], np.int32)
        return per_row[self.traj]

    def split(self, val=0.15, by="motion", seed=0):
        """(train indices, validation indices), whole groups at a time."""
        g = self.groups(by)
        uniq = np.unique(g)
        if len(uniq) < 2 or val <= 0:
            if val > 0:
                print(f"[mapping] only one {by} group: validating on a random {val:.0%} of samples instead")
                rng = np.random.default_rng(seed)
                idx = rng.permutation(len(g))
                n = max(1, int(round(val * len(g))))
                return np.sort(idx[n:]), np.sort(idx[:n])
            return np.arange(len(g)), np.arange(0)
        rng = np.random.default_rng(seed)
        order = rng.permutation(uniq)
        n_val = min(len(uniq) - 1, max(1, int(round(val * len(uniq)))))
        vmask = np.isin(g, order[:n_val])
        return np.flatnonzero(~vmask), np.flatnonzero(vmask)

    def summary(self):
        kinds = {}
        for r in self.rows:
            kinds[r.get("kind", "?")] = kinds.get(r.get("kind", "?"), 0) + 1
        tex = sorted({int(t) for t in self.texture if t >= 0})
        ph = {name: int((self.phase == c).sum()) for name, c in PHASE_CODES.items()}
        what = (f"sim images {self.sim.shape[1]}x{self.sim.shape[2]}" if self.prep.sim_input == "image"
                else f"sim markers {self.prep.n_markers} x 3")
        return (f"{len(self)} paired samples from {len(self.rows)} trajectories in "
                f"{len(self.sessions)} session(s); textures {tex}; {kinds}\n"
                f"  phases {ph}; {what} -> real images {self.real.shape[2]}x{self.real.shape[1]}"
                f"{' grey' if self.prep.gray else ''}, crop {self.prep.real_crop} of "
                f"{self.prep.real_frame} frames")


def crop_check(data, path, width=320):
    """A picture to check the real crop: one raw real frame with the crop box drawn (as
    the camera gives it), next to what the mapping sees, real and simulated. Uses a sample
    in contact if there is one."""
    i = int(np.flatnonzero(data.phase == 1)[0]) if (data.phase == 1).any() else 0
    row = data.rows[int(data.traj[i])]
    with np.load(os.path.join(row["session"], row["file"])) as z:
        frame = np.asarray(z["real_tactip"][int(data.step[i])])
    if frame.ndim == 2:
        frame = frame[..., None]
    rgb = _rgb(frame[..., ::-1] if frame.shape[-1] == 3 and data.prep.real_bgr else frame).copy()
    x, y, w, h = data.prep.crop_box(rgb.shape[:2])
    t = max(1, rgb.shape[1] // 200)
    for (a, b), (c, d) in (((y, y + t), (x, x + w)), ((y + h - t, y + h), (x, x + w)),
                           ((y, y + h), (x, x + t)), ((y, y + h), (x + w - t, x + w))):
        rgb[a:b, c:d] = (255, 0, 255)
    scale = width / rgb.shape[1]
    big = to_uint8(resize_area(rgb, (width, max(1, int(round(rgb.shape[0] * scale))))))
    side = big.shape[0]
    sim = data.sim[i] if data.prep.sim_input == "image" else (
        draw_markers(data.sim[i], data.marker_rest, size=side) if data.marker_rest is not None
        else np.full((side, side, 3), 230, np.uint8))
    tiles = [big, to_uint8(resize_area(_rgb(data.real[i]), (side, side))),
             to_uint8(resize_area(_rgb(sim), (side, side)))]
    out = np.full((side + 4, sum(t.shape[1] for t in tiles) + 4 * len(tiles), 3), 255, np.uint8)
    x0 = 2
    for tile in tiles:
        out[2:2 + side, x0:x0 + tile.shape[1]] = tile
        x0 += tile.shape[1] + 4
    return save_png(path, out)
