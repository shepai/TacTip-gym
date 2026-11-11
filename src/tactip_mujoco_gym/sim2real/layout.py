"""
Which real pin is which simulated marker, and where each pixel of the real camera image lands
on the simulated sensor.

The real camera sees the pins through a lens and the curved skin; the simulation has the
markers' 3D positions. Both are compared looking down the sensor's axis: the simulated
markers' rest positions projected onto the rig's x-y plane (mm, centred on the tip), and the
real pins' rest centroids (px). match_layouts() finds the transform between the two:

    1. a similarity (rotation, scale, optional mirror: the camera may be mounted turned or
       flipped), searched over every angle on the central pins, where the lens bends least;
    2. refined by ICP (closest-pair matching, then refitting) on the central pins;
    3. then a 2D polynomial map (degree 3: lens and dome distortion) by ICP on all pins.

A TacTip's pin pattern is nearly symmetric (hexagonal / rings), so several rotations fit the
rest image almost equally well. match_layouts() returns each of them as a candidate, and
pick_by_motion() picks the one under which the real pins move the way the simulated markers
do in the same presses - the press data settles what the rest image cannot.
"""
from dataclasses import dataclass, field

import numpy as np

from .markers import spacing


# ============================================================================ 2D maps
def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _poly_features(U, degree):
    """[1, u, v, u^2, uv, v^2, ...] up to `degree`, for U (n, 2) already normalised."""
    u, v = U[:, 0], U[:, 1]
    cols = [u ** (d - k) * v ** k for d in range(degree + 1) for k in range(d + 1)]
    return np.stack(cols, axis=1)


@dataclass
class PolyMap:
    """(x, y) -> (x', y'): a polynomial of `degree` in normalised (x, y), after a similarity
    (scale s, rotation theta, mirror flips x first) that does most of the work, so the
    polynomial only has to correct distortion. Fitted by least squares with a small ridge on
    the non-linear terms."""
    centre: np.ndarray = field(default_factory=lambda: np.zeros(2))   # source centre (px)
    scale: float = 1.0                                                 # mm per px
    theta: float = 0.0
    mirror: bool = False
    offset: np.ndarray = field(default_factory=lambda: np.zeros(2))   # target point of centre
    degree: int = 0
    coef: np.ndarray | None = None          # (n_terms, 2), on top of the similarity
    norm: float = 1.0                       # normalisation radius (mm) for the polynomial

    def similarity(self, P):
        P = np.asarray(P, float).reshape(-1, 2) - self.centre
        if self.mirror:
            P = P * np.array([-1.0, 1.0])
        return self.scale * P @ _rot(self.theta).T + self.offset

    def __call__(self, P):
        S = self.similarity(P)
        if self.degree and self.coef is not None:
            S = S + _poly_features(S / self.norm, self.degree) @ self.coef
        return S

    def fit_similarity(self, P, Q):
        """Refit scale, rotation and offset (mirror kept) to pairs P (px) -> Q (mm)."""
        A = np.asarray(P, float) - self.centre
        if self.mirror:
            A = A * np.array([-1.0, 1.0])
        s, R, t = umeyama(A, np.asarray(Q, float))
        self.scale, self.theta, self.offset = s, float(np.arctan2(R[1, 0], R[0, 0])), t
        return self

    def fit_poly(self, P, Q, degree=3, ridge=1e-3):
        """Fit the distortion polynomial on top of the current similarity."""
        S = self.similarity(P)
        self.norm = float(max(np.abs(S).max(), 1e-9))
        F = _poly_features(S / self.norm, degree)
        R = np.asarray(Q, float) - S
        reg = ridge * len(F) * np.eye(F.shape[1])
        reg[0, 0] = 0.0                                  # the constant term is free
        self.coef = np.linalg.solve(F.T @ F + reg, F.T @ R)
        self.degree = int(degree)
        return self

    def to_dict(self):
        return dict(centre=self.centre.tolist(), scale=float(self.scale), theta=float(self.theta),
                    mirror=bool(self.mirror), offset=self.offset.tolist(), degree=int(self.degree),
                    coef=None if self.coef is None else self.coef.tolist(), norm=float(self.norm))

    @classmethod
    def from_dict(cls, d):
        return cls(centre=np.array(d["centre"], float), scale=float(d["scale"]),
                   theta=float(d["theta"]), mirror=bool(d["mirror"]),
                   offset=np.array(d["offset"], float), degree=int(d["degree"]),
                   coef=None if d.get("coef") is None else np.array(d["coef"], float),
                   norm=float(d.get("norm", 1.0)))


def umeyama(A, B):
    """Scale s, rotation R (no reflection) and offset t minimising |s R a + t - b|^2."""
    ma, mb = A.mean(axis=0), B.mean(axis=0)
    A0, B0 = A - ma, B - mb
    U, S, Vt = np.linalg.svd(B0.T @ A0 / len(A))
    D = np.eye(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        D[1, 1] = -1
    R = U @ D @ Vt
    var = (A0 ** 2).sum() / len(A)
    s = float(np.trace(np.diag(S) @ D) / max(var, 1e-12))
    return s, R, mb - s * R @ ma


# ============================================================================ matching
def _dists(A, B):
    return np.linalg.norm(A[:, None, :] - B[None, :, :], axis=-1)


def mutual_pairs(A, B, gate):
    """Pairs (i, j) where A[i] and B[j] are each other's nearest point, closer than gate."""
    if len(A) == 0 or len(B) == 0:
        return np.zeros(0, int), np.zeros(0, int)
    D = _dists(A, B)
    j = D.argmin(axis=1)
    i_back = D.argmin(axis=0)
    i = np.arange(len(A))
    ok = (i_back[j] == i) & (D[i, j] < gate)
    return i[ok], j[ok]


@dataclass
class LayoutMatch:
    """One way of laying the real pins on the simulated markers."""
    map: PolyMap
    real_idx: np.ndarray            # matched real pins ...
    sim_idx: np.ndarray             # ... and their simulated markers
    rms: float                      # mm, after the polynomial map
    score: float                    # rest-image fit, 0 (perfect) .. 1
    motion: float = float("nan")    # agreement of the press displacements (pick_by_motion)

    @property
    def n(self):
        return len(self.real_idx)


def _angle_scores(R0, S, gate, n_angles, mirror):
    """Rest-image fit of R0 (real pins, centred and scaled) rotated by each angle onto S."""
    thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False)
    A = R0 * np.array([-1.0, 1.0]) if mirror else R0
    out = np.empty(n_angles)
    for k, th in enumerate(thetas):
        T = A @ _rot(th).T
        d = _dists(T, S).min(axis=1)
        out[k] = np.minimum(d / gate, 1.0).mean()
    return thetas, out


def match_layouts(real_xy, sim_xy, degree=3, n_angles=360, n_candidates=6, central=0.6,
                  iterations=15, scale=None, verbose=False):
    """Candidate matches of real rest pins (px) onto simulated rest markers (mm, x-y),
    best rest-image fit first. See the module docstring.

    central     fraction of each layout's radius used for the rotation search and the
                similarity ICP (the lens distorts the rim)
    scale       mm per px to start from (default: the ratio of the pin pitches)
    """
    P = np.asarray(real_xy, float)
    Q = np.asarray(sim_xy, float)
    if len(P) < 6 or len(Q) < 6:
        raise ValueError(f"too few points to match ({len(P)} real, {len(Q)} simulated)")
    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    rp = np.linalg.norm(P - cp, axis=1)
    rq = np.linalg.norm(Q - cq, axis=1)
    Pc = P[rp <= central * rp.max()]
    Qc = Q[rq <= central * rq.max()]
    sp_q = spacing(Q)
    s0 = float(scale) if scale else spacing(Qc) / max(spacing(Pc), 1e-9)
    gate0 = 0.5 * sp_q

    # 1. rotation search (both handednesses) on the central pins
    cands = []
    for mirror in (False, True):
        thetas, sc = _angle_scores(s0 * (Pc - cp), Qc - cq, gate0, n_angles, mirror)
        is_min = (sc <= np.roll(sc, 1)) & (sc <= np.roll(sc, -1))
        for k in np.flatnonzero(is_min):
            cands.append((float(sc[k]), mirror, float(thetas[k])))
    cands.sort()
    # keep distinct minima (a plateau gives neighbouring equal minima)
    chosen = []
    for sc, mirror, th in cands:
        if any(m == mirror and abs((th - t + np.pi) % (2 * np.pi) - np.pi) < np.radians(5)
               for _, m, t in chosen):
            continue
        chosen.append((sc, mirror, th))
        if len(chosen) >= n_candidates:
            break

    out = []
    for sc, mirror, th in chosen:
        m = PolyMap(centre=cp.copy(), scale=s0, theta=th, mirror=mirror, offset=cq.copy())
        # 2. similarity ICP on the central pins
        gate = gate0
        for _ in range(iterations):
            i, j = mutual_pairs(m(Pc), Q, gate)
            if len(i) < 4:
                break
            m.fit_similarity(Pc[i], Q[j])
            gate = max(0.3 * sp_q, gate * 0.9)
        # 3. polynomial ICP on all pins
        if degree:
            gate = 0.45 * sp_q
            for _ in range(iterations):
                i, j = mutual_pairs(m(P), Q, gate)
                n_terms = (degree + 1) * (degree + 2) // 2
                if len(i) < 2 * n_terms:
                    break
                m.fit_poly(P[i], Q[j], degree)
                gate = max(0.3 * sp_q, gate * 0.95)
        i, j = mutual_pairs(m(P), Q, 0.35 * sp_q)
        rms = float(np.sqrt(((m(P[i]) - Q[j]) ** 2).sum(axis=1).mean())) if len(i) else np.inf
        out.append(LayoutMatch(map=m, real_idx=i, sim_idx=j, rms=rms, score=sc))
        if verbose:
            print(f"[layout] mirror={mirror} theta={np.degrees(m.theta):7.2f} deg  "
                  f"scale={m.scale:.4f} mm/px  matched {len(i)}/{len(P)} real, rms {rms:.3f} mm")
    out.sort(key=lambda c: (-c.n, c.rms))
    return out


def lateral(match, real_rest, real_pos):
    """Real pin displacements mapped onto the sensor: (probes, n matched, 2) mm in the rig's
    x-y, for real_pos (probes, n real, 2) px."""
    rest = np.asarray(real_rest, float)[match.real_idx]
    base = match.map(rest)
    pos = np.asarray(real_pos, float)[:, match.real_idx]
    return np.stack([match.map(p) - base for p in pos])


def pick_by_motion(matches, real_rest, real_pos, sim_lateral, real_ok=None, verbose=False):
    """Choose among match_layouts() candidates the one whose mapped real displacements point
    the same way as the simulated markers' in the same presses.

    real_pos     (probes, n real, 2) px     real pin positions at each press
    sim_lateral  (probes, n sim, 2) mm      simulated marker displacements, rig x-y
    real_ok      (probes, n real) bool      pins found in each frame (default: all)
    Sets each candidate's .motion (cosine similarity, -1..1) and returns the best."""
    if not matches:
        raise ValueError("no candidate matches")
    for c in matches:
        L = lateral(c, real_rest, real_pos)
        S = np.asarray(sim_lateral, float)[:, c.sim_idx]
        w = np.ones(L.shape[:2]) if real_ok is None else np.asarray(real_ok)[:, c.real_idx]
        num = (w[..., None] * L * S).sum()
        den = np.sqrt((w[..., None] * L * L).sum() * (w[..., None] * S * S).sum())
        c.motion = float(num / den) if den > 0 else 0.0
        if verbose:
            print(f"[layout] candidate mirror={c.map.mirror} theta={np.degrees(c.map.theta):7.2f}"
                  f" deg: {c.n} pins, rest rms {c.rms:.3f} mm, motion agreement {c.motion:+.3f}")
    # the motion has the last word; among (near) ties the better rest fit
    best = max(matches, key=lambda c: (round(c.motion, 2), c.n, -c.rms))
    return best
