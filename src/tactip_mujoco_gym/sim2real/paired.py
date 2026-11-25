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

from .resim import traj_arrays

GRAY = np.array([0.299, 0.587, 0.114], np.float32)          # as sensors.TacTipCamera(mode="gray")
PHASE_CODES = {"approach": 0, "contact": 1, "retract": 2}
REFERENCES = ("trajectory", "session", "none")
SIM_INPUTS = ("image", "markers", "depth")
SIM_KEYS = {"image": "sim_tactip", "markers": "sim_markers", "depth": "sim_depth"}
FILTERS = ("none", "sobel")


def sobel(gray):
    """Edge strength of (N, H, W, 1) grey images, each scaled so its 99th percentile is 200:
    the pins' outlines, without the light level or exposure."""
    g = np.pad(gray[..., 0].astype(np.float32), ((0, 0), (1, 1), (1, 1)), mode="edge")
    gx = (g[:, :-2, 2:] + 2 * g[:, 1:-1, 2:] + g[:, 2:, 2:]
          - g[:, :-2, :-2] - 2 * g[:, 1:-1, :-2] - g[:, 2:, :-2])
    gy = (g[:, 2:, :-2] + 2 * g[:, 2:, 1:-1] + g[:, 2:, 2:]
          - g[:, :-2, :-2] - 2 * g[:, :-2, 1:-1] - g[:, :-2, 2:])
    m = np.sqrt(gx ** 2 + gy ** 2)
    top = np.percentile(m.reshape(len(m), -1), 99, axis=1).reshape(-1, 1, 1)
    return (m / np.maximum(top, 1e-6) * 200.0)[..., None]


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
    reference    how real images are made comparable across sessions: "trajectory" (each
                 is shifted by its trajectory's untouched frames, onto the average untouched
                 sensor), "session" (the same with each session's), or "none". The camera's
                 exposure, light and position drift between and within sessions; the
                 simulator can't know about that, so it is taken out
    """

    def __init__(self, sim_input="image", sim_size=64, sim_render=None, n_markers=None,
                 real_size=(64, 64), real_crop=None, real_frame=None, real_bgr=True, gray=False,
                 reference="none", filter="none", depth_range=None):
        if sim_input not in SIM_INPUTS:
            raise ValueError(f"sim_input must be one of {SIM_INPUTS}, not {sim_input!r}")
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
        if reference not in REFERENCES:
            raise ValueError(f"reference must be one of {REFERENCES}, not {reference!r}")
        self.reference = reference
        if filter not in FILTERS:
            raise ValueError(f"filter must be one of {FILTERS}, not {filter!r}")
        self.filter = filter
        self.depth_range = None if depth_range is None else (float(depth_range[0]), float(depth_range[1]))
        if filter == "sobel":
            self.gray = True                       # edges are one channel

    @property
    def channels(self):
        return 1 if self.gray else 3

    @property
    def sim_is_image(self):
        """The simulated side is a picture (camera image, or depth shown as one)."""
        return self.sim_input in ("image", "depth")

    @property
    def sim_channels(self):
        return 1 if self.gray or self.sim_input == "depth" else 3

    @property
    def sim_shape(self):
        if self.sim_is_image:
            return (self.sim_size, self.sim_size, self.sim_channels)
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
                    real_bgr=self.real_bgr, gray=self.gray, reference=self.reference,
                    filter=self.filter, depth_range=None if self.depth_range is None else list(self.depth_range))

    @classmethod
    def from_dict(cls, d):
        return cls(**{k: d[k] for k in ("sim_input", "sim_size", "sim_render", "n_markers",
                                        "real_size", "real_crop", "real_frame", "real_bgr",
                                        "gray", "reference", "filter", "depth_range") if k in d})

    # ---- sim
    def sim(self, x):
        """Simulated input(s) -> model-ready rows: uint8 (N, s, s, C) images, or float32
        (N, 3n) marker displacements in mm. Returns (array, was_single)."""
        x = np.asarray(x)
        if self.sim_input == "depth":               # mm -> a picture: near bright, nothing 0
            single = x.ndim == 2 or (x.ndim == 3 and x.shape[-1] == 1)
            x = x[None] if single else x
            if x.ndim == 3:
                x = x[..., None]
            return _chunked(lambda b: self._sim_images(self.depth_picture(b)), x), single
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

    def depth_picture(self, mm):
        """(N, H, W, 1) depth in mm -> float 0-255: depth_range[0] (near) bright,
        depth_range[1] (far) dark, no reading 0."""
        if self.depth_range is None:
            raise ValueError("depth_range isn't set (PairedData.load sets it from the data)")
        lo, hi = self.depth_range
        mm = np.asarray(mm, np.float32)
        v = np.clip((hi - mm) / max(hi - lo, 1e-6), 0.0, 1.0) * 255.0
        v[mm <= 0] = 0.0
        return v

    def _sim_images(self, x):
        want = self.sim_channels
        if self.filter == "sobel":                 # edges at twice the size, then shrink
            x = resize_area(x, (2 * self.sim_size, 2 * self.sim_size))
            x = to_gray(x) if x.shape[-1] == 3 else x
            return to_uint8(resize_area(sobel(x), (self.sim_size, self.sim_size)))
        if x.shape[1:3] != (self.sim_size, self.sim_size):
            x = resize_area(x, (self.sim_size, self.sim_size))
        if want == 1 and x.shape[-1] == 3:
            x = to_gray(x)
        elif want == 3 and x.shape[-1] == 1:
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
        if self.filter == "sobel":
            w, h = self.real_size
            x = resize_area(x, (2 * w, 2 * h))
            x = to_gray(x) if x.shape[-1] == 3 else x
            return to_uint8(resize_area(sobel(x), self.real_size))
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

    sim          uint8 (N, s, s, C) simulated images, or float32 (N, 3n) markers (mm)
    real         uint8 (N, h, w, C) real images; with prep.reference, each moved from its
                 trajectory's (or session's) untouched frames onto real_rest
    traj         (N,) trajectory number: rows[traj] is its index.csv row plus its session
    phase, texture, step, depth   per sample (depth: the plan's, mm below the block's top)
    real_signal  (N,) how much the real image differs from its trajectory's untouched frames
                 (mean |difference|, grey levels, before any reference shift)
    sim_signal   (N,) the same for the simulated image (markers: mean displacement, mm)
    real_ref     (trajectories, h, w, C) each trajectory's untouched real image
    real_rest    (h, w, C) the untouched sensor the real images are moved onto (None: not)
    pairing      the pairing check per session (pairing_report)
    """

    PER_SAMPLE = ("sim", "real", "traj", "phase", "texture", "step", "depth", "real_signal",
                  "sim_signal")

    def __init__(self, prep, rows, sessions, marker_rest=None, real_ref=None, real_rest=None,
                 calibrations=None, **arrays):
        self.prep, self.rows, self.sessions = prep, rows, sessions
        self.marker_rest = marker_rest
        self.real_ref, self.real_rest = real_ref, real_rest
        self.calibrations = calibrations or [None] * len(sessions)
        self.pairing = None
        n = len(arrays["real"])
        for k in self.PER_SAMPLE:
            v = arrays.get(k)
            if v is None:                               # optional per-sample values
                v = np.full(n, np.nan, np.float32)
            setattr(self, k, v)

    def __len__(self):
        return len(self.real)

    @property
    def session(self):
        """(N,) the session number of each sample (an index into sessions)."""
        return np.array([r["session_index"] for r in self.rows], np.int32)[self.traj]

    def subset(self, idx):
        idx = np.asarray(idx)
        d = PairedData(self.prep, self.rows, self.sessions, self.marker_rest, self.real_ref,
                       self.real_rest, self.calibrations,
                       **{k: getattr(self, k)[idx] for k in self.PER_SAMPLE})
        d.pairing = self.pairing
        return d

    @classmethod
    def load(cls, paths, sim_input="image", sim_size=64, real_size=(64, 64), real_crop=None,
             gray=False, real_bgr=True, reference="trajectory", phases=None, textures=None,
             every=1, max_samples=None, min_corr=None, prep=None, real_rest=None, log=print,
             filter="none"):
        """Read sessions (folders, or folders of them).

        reference  "trajectory" (default), "session" or "none": see Preprocess
        phases     e.g. ["contact"] (default all, so the untouched sensor is learnt too)
        textures   texture numbers to keep; every: every k-th sample along a trajectory
        min_corr   leave out sessions whose real images follow the simulated ones less than
                   this (the pairing check's correlation, e.g. 0.4)
        prep, real_rest   a trained mapping's, to test it on new sessions
        """
        sessions = find_sessions(paths)
        if prep is not None:
            prep = Preprocess.from_dict(prep.to_dict())
            sim_input = prep.sim_input
        else:
            prep = Preprocess(sim_input=sim_input, sim_size=sim_size, real_size=real_size,
                              real_crop=real_crop, real_bgr=real_bgr, gray=gray,
                              reference=reference, filter=filter)
        want = None if not phases else {PHASE_CODES.get(p, p) for p in phases}
        cols = {k: [] for k in cls.PER_SAMPLE}
        rows, refs, renders, calibrations, skipped = [], [], [], [], {}
        marker_rest, total = None, 0
        sim_key = SIM_KEYS[sim_input]
        for si, sdir in enumerate(sessions):
            with open(os.path.join(sdir, "meta.json")) as f:
                meta = json.load(f)
            calibrations.append(_calibration_note(meta))
            simmeta = meta.get("sim") or {}
            size = simmeta.get("depth_size" if sim_input == "depth" else "image_size")
            if size and sim_input != "markers":
                renders.append(int(size))
            if marker_rest is None and simmeta.get("marker_rest_mm") is not None:
                marker_rest = np.asarray(simmeta["marker_rest_mm"], np.float32)
            n_before = total
            trajs = meta.get("trajectories", [])
            for row in trajs:
                if textures is not None and row.get("texture") not in textures:
                    continue
                z = traj_arrays(sdir, row, (sim_key, "real_tactip", "phase", "waypoint"), meta)
                if sim_key not in z or "real_tactip" not in z:
                    why = f"no {sim_key if sim_key not in z else 'real_tactip'}"
                    if sim_key not in z and sim_input != "markers" and (meta.get("resim") or {}):
                        why += " (re-simulate with --record " + {"image": "image", "depth": "depth"}[sim_input] + ")"
                    skipped[why] = skipped.get(why, 0) + 1
                    continue
                s, r = z[sim_key], z["real_tactip"]
                ph = z["phase"] if "phase" in z else np.full(len(r), -1, np.int8)
                wp = z.get("waypoint")
                if sim_input == "depth" and prep.depth_range is None:
                    v = s[s > 0]
                    if len(v):
                        lo, hi = np.percentile(v, [1, 99])
                        pad = 0.1 * (hi - lo) + 0.5
                        prep.depth_range = (float(max(lo - pad, 0.0)), float(hi + pad))
                depth = (wp[:, 2].astype(np.float32) if wp is not None and wp.ndim == 2 and
                         wp.shape[1] >= 3 and len(wp) == len(r) else np.full(len(r), np.nan, np.float32))
                keep = np.arange(len(r))[::max(1, int(every))]
                if want is not None:
                    keep = keep[np.isin(ph[keep], list(want))]
                if not len(keep):
                    continue
                if sim_input == "markers":
                    if prep.n_markers is None:
                        prep.n_markers = s.shape[1]
                    elif s.shape[1] != prep.n_markers:
                        raise ValueError(f"{os.path.join(sdir, row['file'])}: {s.shape[1]} markers, earlier sessions have "
                                         f"{prep.n_markers}; train on one sensor model")
                still = _untouched(ph, depth)
                real_k = prep.real(r[keep])[0]
                still_r = prep.real(r[still])[0].astype(np.float32)
                ref_r = still_r.mean(axis=0)
                sim_k = prep.sim(s[keep])[0]
                if prep.sim_is_image:
                    ref_s = prep.sim(s[still])[0].astype(np.float32).mean(axis=0)
                    sim_sig = np.abs(sim_k.astype(np.float32) - ref_s).reshape(len(keep), -1).mean(axis=1)
                else:
                    sim_sig = np.linalg.norm(sim_k.reshape(len(keep), -1, 3), axis=2).mean(axis=1)
                real_sig = np.abs(real_k.astype(np.float32) - ref_r).reshape(len(keep), -1).mean(axis=1)
                if len(still) > 1:              # the reference frames: against the others only
                    for j, fi in enumerate(still):
                        pos = np.flatnonzero(keep == fi)
                        if len(pos):
                            others = (ref_r * len(still) - still_r[j]) / (len(still) - 1)
                            real_sig[pos] = np.abs(still_r[j] - others).mean()
                k = len(rows)
                rows.append(dict(row, session=sdir, session_index=si))
                refs.append(ref_r)
                t = row.get("texture", -1)
                cols["sim"].append(sim_k)
                cols["real"].append(real_k)
                cols["traj"].append(np.full(len(keep), k, np.int32))
                cols["phase"].append(ph[keep].astype(np.int8))
                cols["texture"].append(np.full(len(keep), int(t) if str(t).lstrip("-").isdigit() else -1,
                                               np.int16))
                cols["step"].append(keep.astype(np.int32))
                cols["depth"].append(depth[keep])
                cols["real_signal"].append(real_sig.astype(np.float32))
                cols["sim_signal"].append(sim_sig.astype(np.float32))
                total += len(keep)
                if max_samples and total >= max_samples:
                    break
            n = total - n_before
            note = "" if n else (" (no trajectories recorded)" if not trajs else " (none usable)")
            log(f"[mapping] {sdir}: {n} samples{note}")
            if max_samples and total >= max_samples:
                break
        for why, n in skipped.items():
            log(f"[mapping] skipped {n} trajectories: {why}")
        if not rows:
            raise ValueError("no paired samples found (the sessions need real_tactip and "
                             f"{sim_key}; record with the real rig, not --sim-only)")
        if renders and prep.sim_render is None:
            prep.sim_render = max(set(renders), key=renders.count)
            if len(set(renders)) > 1:
                log(f"[mapping] sessions rendered the sim at {sorted(set(renders))} px; all are "
                    f"resized to {prep.sim_size} (the gym will render at {prep.sim_render})")
        data = cls(prep, rows, sessions, marker_rest, real_ref=np.stack(refs),
                   calibrations=calibrations, **{k: np.concatenate(v) for k, v in cols.items()})
        if max_samples and len(data) > max_samples:
            data = data.subset(np.arange(max_samples))
        data.pairing = pairing_report(data)
        log(describe_pairing(data.pairing))
        if min_corr is not None:
            good = [p["index"] for p in data.pairing if np.isfinite(p["corr"]) and p["corr"] >= min_corr]
            dropped = [p["session"] for p in data.pairing if p["index"] not in good]
            if not good:
                raise ValueError(f"no session reaches --min-corr {min_corr}")
            if dropped:
                log(f"[mapping] left out (pairing below {min_corr}): {', '.join(dropped)}")
                data = data.subset(np.flatnonzero(np.isin(data.session, good)))
        return data.apply_reference(real_rest)

    def _refs(self):
        """Each trajectory's untouched real image as prep.reference uses it."""
        refs = self.real_ref
        if self.prep.reference == "session":
            refs = refs.copy()
            sess = np.array([r["session_index"] for r in self.rows])
            for si in np.unique(sess):
                refs[sess == si] = self.real_ref[sess == si].mean(axis=0)
        return refs

    def apply_reference(self, rest=None):
        """Move every real image from its trajectory's (or session's) untouched frames onto
        `rest` (default: the average untouched frame of the trajectories here). Done once,
        by load()."""
        if self.prep.reference == "none" or self.real_ref is None or getattr(self, "_moved", False):
            return self
        refs = self._refs()
        used = np.unique(self.traj)
        self.real_rest = np.asarray(refs[used].mean(axis=0) if rest is None else rest, np.float32)
        order = np.argsort(self.traj, kind="stable")
        bounds = np.flatnonzero(np.diff(self.traj[order])) + 1
        for block in np.split(order, bounds):
            k = self.traj[block[0]]
            self.real[block] = to_uint8(self.real[block].astype(np.float32) - refs[k] + self.real_rest)
        self._moved = True
        return self

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
        for k in np.unique(self.traj):
            kind = self.rows[k].get("kind", "?")
            kinds[kind] = kinds.get(kind, 0) + 1
        tex = sorted({int(t) for t in self.texture if t >= 0})
        ph = {name: int((self.phase == c).sum()) for name, c in PHASE_CODES.items()}
        what = (f"sim {'depth ' if self.prep.sim_input == 'depth' else ''}images "
                f"{self.sim.shape[1]}x{self.sim.shape[2]}" if self.prep.sim_is_image
                else f"sim markers {self.prep.n_markers} x 3")
        used = len(np.unique(self.traj))
        ref = {"trajectory": "each trajectory's untouched frames", "session": "each session's "
               "untouched frames", "none": "nothing"}[self.prep.reference]
        return (f"{len(self)} paired samples from {used} trajectories in "
                f"{len(np.unique(self.session))} session(s); textures {tex}; {kinds}\n"
                f"  phases {ph}; {what} -> real images {self.real.shape[2]}x{self.real.shape[1]}"
                f"{' grey' if self.prep.gray else ''}, crop {self.prep.real_crop} of "
                f"{self.prep.real_frame} frames; real images referenced to {ref}")


def _untouched(phase, depth, n=8):
    """A trajectory's frames with the sensor clear of the block: approach frames at least
    0.5 mm above the top (any approach frames if depth is unknown; else the first few)."""
    idx = np.flatnonzero((phase == 0) & ~(depth > -0.5))
    if not len(idx):
        idx = np.flatnonzero(phase == 0)
    if not len(idx):
        idx = np.arange(min(3, len(phase)))
    return idx[:n]


def _calibration_note(meta):
    c = meta.get("calibration") or {}
    pm = c.get("pose_map") or {}
    if not pm:
        return None
    bits = [f"{pm.get('kind') or '?'} pose map"]
    if pm.get("known_spm"):
        bits.append("measured steps/mm")
    if pm.get("created"):
        bits.append(f"made {pm['created']}")
    trim = c.get("trim_steps")
    if trim and any(trim):
        bits.append(f"nudged {list(trim)} steps")
    return ", ".join(bits)


# ====================================================================== pairing check
def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def pairing_report(data):
    """Does each session's real TacTip change when the simulated one does? One dict per
    session:

    corr       correlation between how much the sim and the real image change from their
               untouched frames, sample by sample (1: they change together; near 0: the real
               images don't follow the simulation, e.g. a wrong calibration)
    lag        samples by which the real images line up best later (+) or earlier (-)
    onset_mm   along the first descent of each trajectory: depth where the real image has
               made 15 % of its change minus where the sim's has (negative: the real TacTip
               touches first, so it sits lower than the simulated one); onset_n descents
    real_snr   real change in contact / its change while clear of the block
    """
    out = []
    sess = data.session
    for si, sdir in enumerate(data.sessions):
        idx = np.flatnonzero(sess == si)
        if len(idx):
            out.append(_pairing(data, idx, si, sdir))
    return out


def _pairing(data, idx, si, sdir):
    rs, ss = data.real_signal[idx], data.sim_signal[idx]
    ph, depth, traj = data.phase[idx], data.depth[idx], data.traj[idx]
    r = dict(session=os.path.basename(os.path.normpath(sdir)), path=sdir, index=si,
             samples=int(len(idx)), calibration=data.calibrations[si], corr=_corr(ss, rs))
    blocks = [np.flatnonzero(traj == k) for k in np.unique(traj)]
    lags = {}
    for lag in range(-3, 4):
        a, b = [], []
        for bl in blocks:
            if len(bl) <= abs(lag):
                continue
            if lag >= 0:
                a.append(ss[bl[:len(bl) - lag]])
                b.append(rs[bl[lag:]])
            else:
                a.append(ss[bl[-lag:]])
                b.append(rs[bl[:len(bl) + lag]])
        lags[lag] = _corr(np.concatenate(a), np.concatenate(b)) if a else float("nan")
    finite = {k: v for k, v in lags.items() if np.isfinite(v)}
    r["lag"] = max(finite, key=finite.get) if finite else 0
    r["lag_corr"] = finite.get(r["lag"], float("nan"))
    if np.isfinite(r["corr"]) and r["lag_corr"] - r["corr"] < 0.03:      # not clearly better
        r["lag"], r["lag_corr"] = 0, r["corr"]
    clear = (ph == 0) & ~(depth > -0.5)
    clear = clear if clear.any() else ph == 0
    touch = ph == 1

    rest_r = float(np.median(rs[clear])) if clear.any() else float("nan")
    r["real_snr"] = (float(np.median(rs[touch])) / max(rest_r, 0.05)
                     if touch.any() and np.isfinite(rest_r) else float("nan"))
    noise_r = float(np.percentile(rs[clear], 95)) if clear.any() else 0.0
    noise_s = float(np.percentile(ss[clear], 95)) if clear.any() else 0.0
    diffs = []
    for bl in blocks:
        d = depth[bl]
        if not np.isfinite(d).all():
            continue
        end = len(bl)
        for i in range(1, len(bl)):
            if d[i] < d[i - 1] - 1e-3 or ph[bl[i]] == 2:       # the first descent only
                end = i
                break
        a = _onset(d[:end], ss[bl[:end]], clear[bl[:end]], noise_s)
        b = _onset(d[:end], rs[bl[:end]], clear[bl[:end]], noise_r)
        if a is not None and b is not None:
            diffs.append(b - a)
    r["onset_mm"] = float(np.median(diffs)) if diffs else float("nan")
    r["onset_n"] = len(diffs)
    c = r["corr"]
    r["verdict"] = "?" if not np.isfinite(c) else "good" if c >= 0.7 else "fair" if c >= 0.4 else "poor"
    return r


def _onset(depth, sig, clear, noise, frac=0.15):
    """Depth (interpolated) where `sig` has made `frac` of its rise along a descent; None if
    it hardly rises above its noise."""
    if len(sig) < 3:
        return None
    base = float(np.median(sig[clear])) if clear.any() else float(sig[0])
    top = float(sig.max())
    if top - base <= 2 * max(noise - base, 1e-9) or top - base <= 1e-6:
        return None
    level = base + frac * (top - base)
    i = int(np.argmax(sig >= level))
    if i == 0:
        return None
    x0, x1 = sig[i - 1], sig[i]
    t = 0.0 if x1 == x0 else (level - x0) / (x1 - x0)
    return float(depth[i - 1] + t * (depth[i] - depth[i - 1]))


def describe_pairing(report):
    """The pairing check in words, with what to do about problems."""
    lines = ["[mapping] pairing check: does the real TacTip change when the simulated one does?",
             "          corr: how closely the two images' change from untouched goes together "
             "(1 = perfectly, 0 = not at all)"]
    for r in report:
        bits = [f"{r['samples']:6d} samples", f"corr {r['corr']:+.2f}"]
        if r["lag"]:
            bits.append(f"best ({r['lag_corr']:+.2f}) with the real images {abs(r['lag'])} "
                        f"sample(s) {'later' if r['lag'] > 0 else 'earlier'}")
        if np.isfinite(r["onset_mm"]):
            bits.append(f"real starts changing {abs(r['onset_mm']):.1f} mm "
                        f"{'before' if r['onset_mm'] < 0 else 'after'} the sim "
                        f"({r['onset_n']} descents)")
        if np.isfinite(r["real_snr"]):
            bits.append(f"real contact change {r['real_snr']:.1f}x its noise")
        lines.append(f"  {r['verdict']:4s}  {r['session']}: " + ", ".join(bits))
        if r["calibration"]:
            lines.append(f"          calibration: {r['calibration']}")
    poor = [r["session"] for r in report if r["verdict"] in ("poor", "?")]
    if poor and len(poor) < len(report):
        lines.append(f"  -> {', '.join(poor)}: the real images hardly follow the simulation (often "
                     "an old or wrong calibration). Leave them out: list only the good sessions, or "
                     "add --min-corr 0.4")
    elif poor:
        lines.append("  -> no session's real images follow the simulation well. Check the pose "
                     "calibration (sim2real real-setup) and the crop (crop_check.png)")
    ok = [r for r in report if r["verdict"] in ("good", "fair")]
    off = [r for r in ok if np.isfinite(r["onset_mm"]) and abs(r["onset_mm"]) >= 0.4 and not r["lag"]]
    if off:
        lines.append("  -> the real TacTip starts touching at a different depth from the simulated one "
                     "in " + ", ".join(f"{r['session']} ({r['onset_mm']:+.1f} mm: real "
                                       f"{'lower' if r['onset_mm'] < 0 else 'higher'})" for r in off)
                     + ". The calibration's z is off by roughly that much (redo real-setup, or nudge "
                     "z when rig_dataset aligns)")
    lagged = [r["session"] for r in report if r["lag"] > 0]
    if lagged:
        lines.append(f"  -> {', '.join(lagged)}: the real images lag the simulated ones (the rig "
                     "hadn't settled). Record with a longer rig_dataset collect --settle, e.g. 0.3")
    weak = [r["session"] for r in ok if np.isfinite(r["real_snr"]) and r["real_snr"] < 1.5]
    if weak:
        lines.append(f"  -> {', '.join(weak)}: the real image barely changes on contact at this "
                     "size. Try --real-size 128 128 or a tighter --real-crop around the pins")
    return "\n".join(lines)


def pairing_plot(data, path, per_session=3):
    """Plot, for a few trajectories per session, how much the real and simulated images
    change from untouched (each scaled to its session's range) and the planned depth.
    Needs matplotlib; returns the path or None."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    sess = data.session
    report = [r for r in (data.pairing or pairing_report(data)) if (sess == r["index"]).any()]
    if not report:
        return None
    fig, axes = plt.subplots(len(report), per_session, figsize=(4 * per_session, 2.2 * len(report)),
                             squeeze=False)
    for row, r in zip(axes, report):
        idx = np.flatnonzero(sess == r["index"])
        traj = data.traj[idx]
        ks, counts = np.unique(traj[data.phase[idx] == 1], return_counts=True)
        pick = ks[np.argsort(-counts)][:per_session] if len(ks) else np.unique(traj)[:per_session]
        top_s = np.percentile(data.sim_signal[idx], 99) or 1.0
        top_r = np.percentile(data.real_signal[idx], 99) or 1.0
        for ax, k in zip(row, pick):
            j = idx[traj == k]
            ax.plot(data.sim_signal[j] / top_s, label="sim change", color="tab:blue")
            ax.plot(data.real_signal[j] / top_r, label="real change", color="tab:red")
            if np.isfinite(data.depth[j]).any():
                ax2 = ax.twinx()
                ax2.plot(data.depth[j], color="0.6", lw=0.8, ls="--")
                ax2.invert_yaxis()
                ax2.tick_params(labelsize=6)
            ax.set_title(f"{r['session']} #{k} {data.rows[k].get('kind', '')} (corr {r['corr']:+.2f})",
                         fontsize=7)
            ax.tick_params(labelsize=6)
        for ax in row[len(pick):]:
            ax.axis("off")
    axes[0][0].legend(fontsize=6, loc="upper left")
    fig.suptitle("how much each image changes from untouched (scaled), by sample; dashed: planned depth",
                 fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def crop_check(data, path, width=320):
    """A picture to check the real crop: one raw real frame with the crop box drawn (as
    the camera gives it), next to what the mapping sees, real and simulated. Uses a sample
    in contact if there is one."""
    i = int(np.flatnonzero(data.phase == 1)[0]) if (data.phase == 1).any() else 0
    row = data.rows[int(data.traj[i])]
    frame = np.asarray(traj_arrays(row["session"], row, ("real_tactip",))["real_tactip"][int(data.step[i])])
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
    sim = data.sim[i] if data.prep.sim_is_image else (
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
