"""
Surfaces to touch, as heightfields: the texture blocks of the 3D Printable Tactile Dataset
(TacRig) and wave surfaces with or without those textures on them (TacArm-surface).

The dataset (Shepherd et al., "A 3D-Printable Dataset for Fair Testing and Comparisons of
Tactile Sensors", arXiv:2606.25886; STL files: doi.org/10.25377/sussex.30256453; generator:
github.com/shepai/3D-textures, Generator/Generator.py) is six textures z = f(x, y), each
printed as a solid block. x, y and z are in millimetres at the printed scale:

    1  sin x                      ridges in one direction
    2  sin x + cos y              sinusoidal in both directions
    3  tri(x) + tri(y)            triangle wave (Fourier series, 13 odd terms), both directions
    4  sq(x) + sq(y)              square wave (51 odd terms), both directions
    5  tri(x)                     triangle wave, one direction
    6  sq(x)                      square wave, one direction
    7 .. 20                       the extension textures listed on the dataset page

Block (BLOCK_MM): 29.3 x 30.0 mm, 7 mm tall at its highest point. That is the generator's
grid (x in [0, 6 pi + 10.5), y in [0, 6 pi + 11.2), every 0.1), its STL files and the
pocket of the rig's texture holder (31.1 x 30 mm). The paper's text gives 23.3 x 30 mm;
change BLOCK_MM if that is the size you print. Every block is the texture on a base that
makes it 7 mm tall: top = f + 7 - max(f) (heights in mm above the block's underside).

Other blocks: texture_heights() also takes an STL file (any of the dataset's, or your own
block): its top surface is read off the mesh and scaled to the block (STL meshes are not
used for contact directly: MuJoCo collides meshes as their convex hull, which would flatten
a texture).
"""
import os
import struct

import numpy as np

BLOCK_MM = (29.3, 30.0, 7.0)              # x, y footprint and height of a printed block
GRID_STEP_MM = 0.1                        # the generator's sampling step
DATASET = (1, 2, 3, 4, 5, 6)              # the benchmark textures
N_TERMS = {3: 25, 4: 101, 5: 25, 6: 101}  # highest odd harmonic, as in Generator.py

TEXTURE_NAMES = {
    1: "sine ridges (x)", 2: "sine (x and y)", 3: "triangle (x and y)",
    4: "square (x and y)", 5: "triangle ridges (x)", 6: "square ridges (x)",
    7: "rippling waves", 8: "concentric rings", 9: "diamond grid", 10: "woven fabric",
    11: "hexagonal", 12: "sharp ridges", 13: "chipped rock", 14: "sand dunes",
    15: "cellular dimples", 16: "fish scales", 17: "knurled", 18: "swirling fluid",
    19: "chevron", 20: "organic coral",
}


def _triangle(u, n):
    out = np.zeros_like(u)
    for i in range(1, n + 1, 2):
        out += (8 / np.pi ** 2) * (-1) ** ((i - 1) // 2) / i ** 2 * np.sin(i * u)
    return out


def _square(u, n):
    out = np.zeros_like(u)
    for i in range(1, n + 1, 2):
        out += 4 / (np.pi * i) * np.sin(i * u)
    return out


def texture_function(texture, n_terms=None):
    """f(x, y) of a dataset texture (1..20), x and y in mm. n_terms: highest odd harmonic
    for textures 3-6 (default: as in the generator)."""
    t = int(texture)
    n = int(n_terms or N_TERMS.get(t, 25))
    s3 = np.sqrt(3.0)
    table = {
        1: lambda x, y: np.sin(x) + 0.0 * y,
        2: lambda x, y: np.sin(x) + np.sin(y + np.pi / 2),
        3: lambda x, y: _triangle(x, n) + _triangle(y, n),
        4: lambda x, y: _square(x, n) + _square(y, n),
        5: lambda x, y: _triangle(x, n) + 0.0 * y,
        6: lambda x, y: _square(x, n) + 0.0 * y,
        7: lambda x, y: np.sin(x + y) + 0.5 * np.cos(2 * x - 2 * y),
        8: lambda x, y: np.sin(np.sqrt(np.sin(x) ** 2 + np.cos(y) ** 2) * np.pi),
        9: lambda x, y: np.abs(np.sin(x)) + np.abs(np.cos(y)),
        10: lambda x, y: np.sin(x) * np.sin(y) + 0.3 * np.cos(3 * x) * np.cos(3 * y),
        11: lambda x, y: np.cos(x) + np.cos(0.5 * x + s3 / 2 * y) + np.cos(0.5 * x - s3 / 2 * y),
        12: lambda x, y: 1.0 - np.abs(np.sin(x) * np.cos(y)),
        13: lambda x, y: np.maximum(np.sin(x), np.cos(y)) + 0.5 * np.minimum(np.sin(2 * x), np.cos(2 * y)),
        14: lambda x, y: np.exp(np.sin(x)) * np.cos(y),
        15: lambda x, y: -np.sqrt(np.sin(x / 2) ** 2 * np.cos(y / 2) ** 2),
        16: lambda x, y: np.sin(x + np.sin(y)) * np.cos(y),
        17: lambda x, y: np.arcsin(np.clip(np.sin(x) * np.cos(y), -1, 1)),
        18: lambda x, y: np.sin(x + np.cos(y)) + np.cos(y + np.sin(x)),
        19: lambda x, y: np.arccos(np.clip(np.cos(x + np.arcsin(np.sin(y))), -1, 1)),
        20: lambda x, y: (np.sin(x) * np.cos(y) + 0.5 * np.sin(2 * x + np.pi / 4) * np.cos(2 * y)
                          + 0.25 * np.sin(4 * x) * np.sin(4 * y)),
    }
    if t not in table:
        raise ValueError(f"texture {texture!r}: the dataset has textures 1..20 (benchmark 1..6)")
    return table[t]


def generator_grid():
    """The generator's x, y sample points (mm)."""
    x = np.arange(0, 6 * np.pi + 10.5, GRID_STEP_MM)
    y = np.arange(0, 6 * np.pi + 11.2, GRID_STEP_MM)
    return x, y


_RANGE = {}


def texture_range(texture, n_terms=None):
    """(min, max) of f over the generator's grid: sets the block's base thickness."""
    key = (int(texture), n_terms)
    if key not in _RANGE:
        x, y = generator_grid()
        X, Y = np.meshgrid(x, y)
        z = texture_function(texture, n_terms)(X, Y)
        _RANGE[key] = (float(z.min()), float(z.max()))
    return _RANGE[key]


def node_grid(res_mm, size_mm=BLOCK_MM[:2]):
    """Heightfield node positions covering a block: (x, y) in mm from its corner, with at
    most res_mm between nodes."""
    nx = int(np.ceil(size_mm[0] / res_mm)) + 1
    ny = int(np.ceil(size_mm[1] / res_mm)) + 1
    return np.linspace(0, size_mm[0], nx), np.linspace(0, size_mm[1], ny)


def texture_heights(texture, res_mm=0.25, block_mm=BLOCK_MM, n_terms=None, blur_mm=0.0,
                    noise_mm=0.0, rng=None):
    """Top of a block, (rows = y, cols = x) in mm above its underside, on nodes res_mm apart.

    texture   1..20 (the dataset's equations, block as printed: highest point 7 mm), or the
              path of an STL block (its top surface, scaled to block_mm)
    blur_mm, noise_mm   imitate a print: Gaussian blur (sigma, mm) of the surface and
              height noise (sd, mm). Default: the ideal surface."""
    xs, ys = node_grid(res_mm, block_mm[:2])
    if isinstance(texture, (str, os.PathLike)) and not str(texture).isdigit():
        H = stl_heights(texture, xs, ys, block_mm)
    else:
        lo, hi = texture_range(texture, n_terms)
        X, Y = np.meshgrid(xs * BLOCK_MM[0] / block_mm[0], ys * BLOCK_MM[1] / block_mm[1])
        H = (texture_function(texture, n_terms)(X, Y) + BLOCK_MM[2] - hi) * block_mm[2] / BLOCK_MM[2]
    if blur_mm > 0:
        H = gaussian_blur(H, blur_mm / (xs[1] - xs[0]), blur_mm / (ys[1] - ys[0]))
    if noise_mm > 0:
        rng = np.random.default_rng() if rng is None else rng
        H = H + rng.normal(0.0, noise_mm, H.shape)
    return np.clip(H, 0.0, block_mm[2])


def gaussian_blur(H, sx, sy):
    """Separable Gaussian blur, sigma in samples along x (columns) and y (rows); edges
    extended."""
    out = np.asarray(H, float)
    for axis, s in ((1, sx), (0, sy)):
        if s <= 0:
            continue
        r = max(1, int(np.ceil(3 * s)))
        k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
        k /= k.sum()
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        P = np.pad(out, pad, mode="edge")
        out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), axis, P)
    return out


# ----------------------------------------------------------------------------- STL blocks
def read_stl(path):
    """Vertices of an STL file (binary or ASCII), (n, 3) float."""
    with open(path, "rb") as f:
        data = f.read()
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if len(data) == 84 + 50 * n:
            rec = np.frombuffer(data, dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]),
                                count=n, offset=84)
            return rec["v"].reshape(-1, 3).astype(float)
    import re
    v = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", data)
    if not v:
        raise ValueError(f"{path}: not an STL file")
    return np.array(v, dtype=float)


def stl_heights(path, xs, ys, block_mm=BLOCK_MM):
    """Top surface of an STL block on the nodes (xs, ys) (mm, from the block's corner): the
    mesh's footprint is scaled to block_mm[:2] and its height to block_mm[2]. Each node takes
    the height of the top-surface vertex closest to it (in x, y); nodes that are no vertex's
    closest take their neighbours' (for meshes coarser than the nodes)."""
    V = read_stl(path)
    lo, hi = V.min(axis=0), V.max(axis=0)
    span = np.maximum(hi - lo, 1e-9)
    P = (V - lo) / span * np.asarray(block_mm, float)
    top = P[P[:, 2] > 1e-6 * block_mm[2]]                  # the underside's vertices don't count
    dx, dy = xs[1] - xs[0], ys[1] - ys[0]
    i = np.clip(np.rint(top[:, 0] / dx).astype(int), 0, len(xs) - 1)
    j = np.clip(np.rint(top[:, 1] / dy).astype(int), 0, len(ys) - 1)
    dist = np.hypot(top[:, 0] - xs[i], top[:, 1] - ys[j])
    node = j * len(xs) + i
    order = np.lexsort((-top[:, 2], dist, node))           # per node: closest (then highest) first
    nodes, first = np.unique(node[order], return_index=True)
    H = np.full((len(ys), len(xs)), -np.inf)
    H.flat[nodes] = top[order[first], 2]
    for _ in range(max(len(xs), len(ys))):                 # fill gaps from the neighbours
        empty = ~np.isfinite(H)
        if not empty.any():
            break
        P_ = np.pad(H, 1, constant_values=-np.inf)
        nb = np.stack([P_[:-2, 1:-1], P_[2:, 1:-1], P_[1:-1, :-2], P_[1:-1, 2:]])
        H[empty] = nb.max(axis=0)[empty]
    H[~np.isfinite(H)] = 0.0
    return H


# ----------------------------------------------------------------------------- wave surfaces
WAVES = ("uniform", "chirp", "multi")


def wave_profile(kind, s, rng, amp=(0.004, 0.010), wavelength=(0.04, 0.10)):
    """Height (m) along a path s (m, from 0) for a wave surface:
        "flat"     no wave
        "uniform"  one sine: constant amplitude and wavelength
        "chirp"    wavelength and amplitude change steadily along the path (non-uniform)
        "multi"    two or three sines of different wavelengths added (non-uniform)
    amp, wavelength: ranges the parameters are drawn from. Returns (heights, params)."""
    s = np.asarray(s, float)
    L = max(float(s[-1] - s[0]), 1e-9)
    u = s - s[0]
    if kind == "flat":
        return np.zeros_like(s), {}
    if kind == "uniform":
        A, lam, ph = rng.uniform(*amp), rng.uniform(*wavelength), rng.uniform(0, 2 * np.pi)
        return A * np.sin(2 * np.pi * u / lam + ph), dict(amp=A, wavelength=lam, phase=ph)
    if kind == "chirp":
        l0, l1 = rng.uniform(*wavelength, size=2)
        while abs(l1 - l0) < 0.3 * (wavelength[1] - wavelength[0]):
            l0, l1 = rng.uniform(*wavelength, size=2)
        a0, a1 = rng.uniform(*amp, size=2)
        lam = l0 + (l1 - l0) * u / L
        phase = 2 * np.pi * L / (l1 - l0) * np.log(lam / l0) + rng.uniform(0, 2 * np.pi)
        return (a0 + (a1 - a0) * u / L) * np.sin(phase), dict(amp=(a0, a1), wavelength=(l0, l1))
    if kind == "multi":
        k = int(rng.integers(2, 4))
        A = rng.uniform(*amp, size=k)
        A *= amp[1] / max(A.sum(), amp[1])                    # the sum stays within amp[1]
        lam = np.exp(rng.uniform(np.log(wavelength[0] / 2), np.log(wavelength[1]), size=k))
        ph = rng.uniform(0, 2 * np.pi, size=k)
        h = sum(a * np.sin(2 * np.pi * u / w + p) for a, w, p in zip(A, lam, ph, strict=True))
        return h, dict(amp=tuple(A), wavelength=tuple(lam), phase=tuple(ph))
    raise ValueError(f"wave {kind!r}: choose from {('flat',) + WAVES}")


def texture_patch(texture, x_m, y_m, scale=1.0, n_terms=None):
    """A dataset texture as relief (m, zero mean over a period, peak-to-peak as printed x
    scale) at points x_m, y_m (m): for putting on other surfaces."""
    f = texture_function(texture, n_terms)
    lo, hi = texture_range(texture, n_terms)
    z_mm = f(np.asarray(x_m) * 1e3, np.asarray(y_m) * 1e3) - 0.5 * (lo + hi)
    return scale * z_mm * 1e-3


def smooth_window(s, ramp_in, ramp_out, flat_in=0.0, flat_out=0.0):
    """1 along the path, 0 for flat_in at the start and flat_out at the end, easing between
    over ramp_in / ramp_out."""
    s = np.asarray(s, float)
    u = s - s[0] - flat_in
    v = s[-1] - s - flat_out
    w = np.ones_like(s)
    w[u < 0] = 0.0
    w[v < 0] = 0.0
    if ramp_in > 0:
        w *= np.where(u < ramp_in, 0.5 - 0.5 * np.cos(np.pi * np.clip(u / ramp_in, 0, 1)), 1.0)
    if ramp_out > 0:
        w *= np.where(v < ramp_out, 0.5 - 0.5 * np.cos(np.pi * np.clip(v / ramp_out, 0, 1)), 1.0)
    return w


# ----------------------------------------------------------------------------- heightfields
def write_hfield(model, hfield_id, heights, elevation, offset=0.0):
    """Write heights (rows = y, cols = x; metres) into a heightfield of `model`, as
    (heights - offset) / elevation, which must stay in [0, 1]: the heightfield's z size is
    `elevation`. Values outside are clipped (the compiled bounding box assumes [0, 1])."""
    H = np.asarray(heights, float)
    nrow, ncol = int(model.hfield_nrow[hfield_id]), int(model.hfield_ncol[hfield_id])
    if H.shape != (nrow, ncol):
        raise ValueError(f"heights are {H.shape}, the heightfield is {(nrow, ncol)}")
    adr = int(model.hfield_adr[hfield_id])
    model.hfield_data[adr:adr + H.size] = np.clip((H - offset) / elevation, 0.0, 1.0).ravel()


def upload_hfields(model, renderer, hfield_ids):
    """Send changed heightfields to a mujoco.Renderer (or a passive viewer handle)."""
    import mujoco
    if renderer is None:
        return
    if hasattr(renderer, "update_hfield"):                    # mujoco.viewer handle
        for h in hfield_ids:
            renderer.update_hfield(int(h))
        return
    ctx = getattr(renderer, "_mjr_context", None)
    gl = getattr(renderer, "_gl_context", None)
    if ctx is None:
        return
    if gl is not None:
        gl.make_current()
    for h in hfield_ids:
        mujoco.mjr_uploadHField(model, ctx, int(h))


class Bilinear:
    """Height and slope of a heightfield grid at any (x, y): heights H (rows = y,
    cols = x) on the box [x0, x1] x [y0, y1]."""

    def __init__(self, H, x0, x1, y0, y1):
        self.H = np.asarray(H, float)
        self.x0, self.y0 = float(x0), float(y0)
        self.dx = (x1 - x0) / (self.H.shape[1] - 1)
        self.dy = (y1 - y0) / (self.H.shape[0] - 1)

    def _cell(self, x, y):
        fx = np.clip((np.asarray(x, float) - self.x0) / self.dx, 0, self.H.shape[1] - 1 - 1e-9)
        fy = np.clip((np.asarray(y, float) - self.y0) / self.dy, 0, self.H.shape[0] - 1 - 1e-9)
        i, j = np.floor(fx).astype(int), np.floor(fy).astype(int)
        return i, j, fx - i, fy - j

    def __call__(self, x, y):
        i, j, u, v = self._cell(x, y)
        H = self.H
        return ((1 - u) * (1 - v) * H[j, i] + u * (1 - v) * H[j, i + 1]
                + (1 - u) * v * H[j + 1, i] + u * v * H[j + 1, i + 1])

    def gradient(self, x, y):
        i, j, u, v = self._cell(x, y)
        H = self.H
        gx = ((1 - v) * (H[j, i + 1] - H[j, i]) + v * (H[j + 1, i + 1] - H[j + 1, i])) / self.dx
        gy = ((1 - u) * (H[j + 1, i] - H[j, i]) + u * (H[j + 1, i + 1] - H[j, i + 1])) / self.dy
        return gx, gy

    def normal(self, x, y):
        gx, gy = self.gradient(x, y)
        n = np.array([-gx, -gy, 1.0])
        return n / np.linalg.norm(n)
