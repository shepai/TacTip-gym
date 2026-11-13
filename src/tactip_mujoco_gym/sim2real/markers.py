"""
Pins in a real TacTip's camera image: find them, and follow each one from frame to frame.

    det = MarkerDetector()                       # bright pins on a dark background
    rest = det.rest(frames)                      # averaged over a few untouched frames
    tracker = MarkerTracker(rest)
    xy, ok = tracker.update(det(frame))          # (n, 2) px, in the order of `rest`

Only needs numpy and OpenCV (pip install opencv-python). The detector thresholds the image
locally (pins are usually bright dots on the dark inside of the skin; invert=True for dark
pins), keeps round blobs of a plausible size and returns their centroids in pixels.
"""
from dataclasses import asdict, dataclass

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as err:                                   # pragma: no cover
        raise ImportError("the real TacTip side needs OpenCV: pip install opencv-python") from err
    return cv2


def to_gray(frame):
    """uint8 grayscale image from a BGR / RGB / gray frame."""
    f = np.asarray(frame)
    if f.ndim == 3 and f.shape[2] == 1:
        f = f[..., 0]
    if f.ndim == 3:
        f = (f[..., :3].astype(np.float32) @ np.array([0.114, 0.587, 0.299], np.float32))
    if f.dtype != np.uint8:
        f = np.clip(f, 0, 255).astype(np.uint8)
    return f


@dataclass
class MarkerDetector:
    """Pin centroids in a TacTip image.

    invert        True for dark pins on a light background
    block         adaptive-threshold neighbourhood (px, odd): about 2 x the pin spacing
    offset        how much brighter than its neighbourhood a pixel must be (grey levels)
    blur          Gaussian blur sigma (px) before thresholding (0 = none)
    min_area, max_area   pin size limits (px^2)
    min_circularity      4 pi area / perimeter^2 (1 = a perfect disc)
    roi           (cx, cy, r): only keep pins within r px of (cx, cy); None = whole image
    """
    invert: bool = False
    block: int = 31
    offset: float = -8.0
    blur: float = 1.5
    min_area: float = 6.0
    max_area: float = 900.0
    min_circularity: float = 0.4
    roi: tuple | None = None

    def __call__(self, frame):
        """(n, 2) float array of pin centres (x, y) in pixels."""
        cv2 = _cv2()
        g = to_gray(frame)
        if self.invert:
            g = 255 - g
        if self.blur > 0:
            g = cv2.GaussianBlur(g, (0, 0), self.blur)
        block = max(3, int(self.block) | 1)
        # pixels brighter than the local mean by -offset
        bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                   block, self.offset)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        pts = []
        for c in contours:
            a = cv2.contourArea(c)
            if a < self.min_area or a > self.max_area:
                continue
            per = cv2.arcLength(c, True)
            if per <= 0 or 4 * np.pi * a / per ** 2 < self.min_circularity:
                continue
            m = cv2.moments(c)
            if m["m00"] <= 0:
                continue
            pts.append((m["m10"] / m["m00"], m["m01"] / m["m00"]))
        P = np.array(pts, dtype=np.float64).reshape(-1, 2)
        if self.roi is not None and len(P):
            cx, cy, r = self.roi
            P = P[np.hypot(P[:, 0] - cx, P[:, 1] - cy) <= r]
        return P

    def rest(self, frames, min_seen=0.8):
        """Rest pin positions from several untouched frames: the pins of the first frame,
        each averaged over the frames it is found in (within a third of the pin spacing),
        keeping those found in at least `min_seen` of the frames."""
        frames = list(frames)
        if not frames:
            raise ValueError("no frames")
        ref = self(frames[0])
        if len(ref) < 3:
            raise RuntimeError(f"only {len(ref)} pins found: adjust the MarkerDetector settings "
                               "(invert, block, offset, min_area, max_area)")
        gate = spacing(ref) / 3
        acc, cnt = np.zeros_like(ref), np.zeros(len(ref))
        for f in frames:
            P = self(f)
            if not len(P):
                continue
            d, j = _nearest(ref, P)
            ok = d < gate
            acc[ok] += P[j[ok]]
            cnt[ok] += 1
        keep = cnt >= max(1, min_seen * len(frames))
        return acc[keep] / cnt[keep, None]

    def to_dict(self):
        d = asdict(self)
        d["roi"] = list(self.roi) if self.roi is not None else None
        return d


def _nearest(A, B):
    """For each row of A: distance to and index of the nearest row of B."""
    if len(B) == 0:
        return np.full(len(A), np.inf), np.zeros(len(A), int)
    D = np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)
    j = D.argmin(axis=1)
    return D[np.arange(len(A)), j], j


def regular_pins(P, tol=0.35):
    """How many points sit on a regular pin grid: their nearest neighbour is within tol of
    the median spacing. Scores a detection: real pins are evenly spaced, noise isn't."""
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) < 6:
        return 0
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    d = D.min(axis=1)
    med = np.median(d)
    return int((np.abs(d - med) < tol * med).sum())


def auto_tune(frame, roi=None, verbose=False):
    """MarkerDetector settings that suit this image: tries bright and dark pins over a
    range of pin sizes and thresholds, and keeps the one that finds the most evenly spaced
    pins. Returns (detector, number of regular pins)."""
    best, best_n = None, -1
    for invert in (False, True):
        for s in (0.5, 1, 1.5, 2, 3, 4, 6, 8):
            for offset in (-4, -8, -16):
                det = MarkerDetector(invert=invert, block=int(31 * s) | 1, offset=offset,
                                     blur=1.5 * s, min_area=max(3.0, 6.0 * s * s),
                                     max_area=900.0 * s * s, roi=roi)
                try:
                    n = regular_pins(det(frame))
                except Exception:
                    continue
                if n > best_n:
                    best, best_n = det, n
    if verbose and best is not None:
        print(f"[markers] auto-tuned: {best_n} evenly spaced pins with {best.to_dict()}")
    return best, best_n


def spacing(P):
    """Median nearest-neighbour distance of a point set (the pin pitch)."""
    P = np.asarray(P, float)
    if len(P) < 2:
        return 1.0
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    return float(np.median(D.min(axis=1)))


class MarkerTracker:
    """Follows every rest pin from frame to frame: each takes the nearest detection to where
    it was last (within max_jump px, default half the pin pitch), each detection going to
    one pin only; pins still unclaimed then try from their rest position (so they come back
    after the skin springs back between frames). Pins not found keep their last position
    and are flagged in `ok`. Read often while the skin moves: a pin that moves more than
    max_jump between two frames is lost until it is back near rest."""

    def __init__(self, rest, max_jump=None):
        self.rest = np.asarray(rest, float).copy()
        self.max_jump = float(max_jump if max_jump is not None else 0.5 * spacing(self.rest))
        self.reset()

    def reset(self):
        self.pos = self.rest.copy()
        self.ok = np.ones(len(self.rest), bool)

    def _assign(self, From, P, pins, dets, new):
        """Greedy one-to-one, closest pairs first, among free pins and free detections."""
        D = np.linalg.norm(From[:, None] - P[None], axis=-1)
        D[pins, :] = np.inf
        D[:, dets] = np.inf
        for flat in np.argsort(D, axis=None):
            i, j = divmod(int(flat), len(P))
            if D[i, j] > self.max_jump:
                break
            if pins[i] or dets[j]:
                continue
            pins[i] = dets[j] = True
            new[i] = P[j]

    def update(self, detections):
        P = np.asarray(detections, float).reshape(-1, 2)
        pins = np.zeros(len(self.pos), bool)
        if len(P):
            dets = np.zeros(len(P), bool)
            new = self.pos.copy()
            self._assign(self.pos, P, pins, dets, new)      # from where they were
            self._assign(self.rest, P, pins, dets, new)     # the rest from rest
            self.pos = new
        self.ok = pins
        return self.pos.copy(), pins.copy()

    def displacement(self):
        return self.pos - self.rest


def match_rest(old_rest, new_rest, gate=None):
    """Index into new_rest for every pin of old_rest (-1 where none is within `gate` px):
    carries a calibration's pin numbering over to a new session's rest image."""
    old_rest, new_rest = np.asarray(old_rest, float), np.asarray(new_rest, float)
    gate = 0.5 * spacing(old_rest) if gate is None else gate
    d, j = _nearest(old_rest, new_rest)
    j = np.where(d < gate, j, -1)
    # one-to-one: a new pin claimed twice goes to the closer old pin
    for k in np.unique(j[j >= 0]):
        idx = np.flatnonzero(j == k)
        if len(idx) > 1:
            keep = idx[np.argmin(d[idx])]
            j[idx[idx != keep]] = -1
    return j
