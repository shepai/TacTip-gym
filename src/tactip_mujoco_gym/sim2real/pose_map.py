"""
Where to turn the real rig's steppers to put the sensor where the simulated one is.

Recorded by hand (teleop.pose_calibrate): both rigs reset, you drive each with the keyboard
until they are in the same place relative to their blocks, press Enter, and repeat at a few
poses. Each pair is (simulated position, mm; real stepper position, steps from home). The
fit maps one to the other:

    axis    (default) per axis: steps = k * mm + b. Needs two poses that differ by
            >= 1 mm along every axis (an axis you didn't vary takes its scale from a known
            steps/mm, e.g. rig_setup.json, if given).
    affine  steps = A mm + b, a full 3 x 3: also captures axes that aren't square to the
            simulator's. Needs four poses not all in one plane; use six or more.

    pm = PoseMap.load("calib/pose_calibration.json")
    steps = pm.to_steps(sim_mm)        # where the steppers go (steps from home)
    sim_mm = pm.to_sim(steps)

Simulated positions are TacRig's commanded axes (SimRig.pos before contact is found, i.e.
the spindle: x, y from the block centre, z = 0 at the reset height, 10 mm above the block's
top). Real positions count from home (every limit switch pressed), so they are <= 0.
"""
import datetime
import json

import numpy as np

VERSION = 1


class PoseMap:
    def __init__(self, A=None, b=None, kind=None, points=None, known_spm=None, known_sign=None):
        self.A = None if A is None else np.asarray(A, float)       # (3, 3) steps per mm
        self.b = None if b is None else np.asarray(b, float)       # (3,) steps at sim 0
        self.kind = kind
        self.points = list(points or [])                           # [{"sim_mm", "steps"}]
        self.known_spm = None if known_spm is None else np.asarray(known_spm, float)
        self.known_sign = None if known_sign is None else np.asarray(known_sign, float)
        self.residual_mm = None
        self.created = datetime.datetime.now().isoformat(timespec="seconds")

    # ---- points
    def add(self, sim_mm, steps):
        self.points.append({"sim_mm": [float(v) for v in sim_mm],
                            "steps": [int(round(v)) for v in steps]})

    def undo(self):
        return self.points.pop() if self.points else None

    def arrays(self):
        S = np.array([p["sim_mm"] for p in self.points], float).reshape(-1, 3)
        T = np.array([p["steps"] for p in self.points], float).reshape(-1, 3)
        return S, T

    # ---- fit
    def can_fit(self, kind="axis"):
        try:
            self._solve(kind)
            return True
        except ValueError:
            return False

    def _solve(self, kind):
        S, T = self.arrays()
        n = len(S)
        if n == 0:
            raise ValueError("no poses recorded")
        if kind == "affine":
            if n < 4:
                raise ValueError(f"affine needs 4+ poses ({n} so far)")
            X = np.c_[S, np.ones(n)]
            if np.linalg.matrix_rank(S - S.mean(axis=0), tol=0.5) < 3:
                raise ValueError("affine needs poses spread out in x, y and z (not all in a plane)")
            coef, *_ = np.linalg.lstsq(X, T, rcond=None)              # (4, 3)
            return coef[:3].T, coef[3]
        if kind != "axis":
            raise ValueError(f"unknown kind {kind!r} (axis or affine)")
        A, b = np.zeros((3, 3)), np.zeros(3)
        for i, ax in enumerate("xyz"):
            s, t = S[:, i], T[:, i]
            if np.ptp(s) >= 1.0:                                      # varied: fit the scale
                k, c = np.polyfit(s, t, 1)
            elif self.known_spm is not None:                          # not varied: known scale
                sign = self.known_sign[i] if self.known_sign is not None else 1.0
                k = sign * self.known_spm[i]
                c = float(np.mean(t - k * s))
            else:
                raise ValueError(f"{ax}: record two poses at least 1 mm apart along {ax} "
                                 "(or give the steps/mm, e.g. from rig_setup.json)")
            A[i, i], b[i] = k, c
        return A, b

    def fit(self, kind="axis"):
        self.A, self.b = self._solve(kind)
        self.kind = kind
        S, T = self.arrays()
        if len(S):
            err = self.to_sim(T) - S                                   # in sim mm
            self.residual_mm = np.linalg.norm(err, axis=1)
        return self

    # ---- use
    def to_steps(self, sim_mm):
        if self.A is None:
            raise RuntimeError("not fitted")
        return np.asarray(sim_mm, float) @ self.A.T + self.b

    def to_sim(self, steps):
        if self.A is None:
            raise RuntimeError("not fitted")
        return np.linalg.solve(self.A, (np.asarray(steps, float) - self.b).T).T

    def steps_per_mm(self):
        """|scale| and sign per axis (the diagonal; for an affine map, its column norms)."""
        k = np.diag(self.A) if self.kind == "axis" else np.linalg.norm(self.A, axis=0) * np.sign(np.diag(self.A))
        return np.abs(k), np.sign(k).astype(int)

    def rig_setup(self, contact_z_mm=-10.0, home_buttons=(3, 2, 0)):
        """RealRig keyword arguments (rig_setup.json) implied by the map: steps/mm, axis
        signs, and home_offset = the steps at the block centre at contact height (the
        simulated flat block's top: contact_z_mm below the reset height)."""
        spm, sign = self.steps_per_mm()
        off = np.rint(self.to_steps([0.0, 0.0, contact_z_mm])).astype(int)
        return dict(steps_per_mm=spm.tolist(), axis_sign=sign.tolist(), home_offset=off.tolist(),
                    home_buttons=list(home_buttons), start_height_mm=-contact_z_mm)

    def summary(self):
        S, _ = self.arrays()
        if self.A is None:
            return f"{len(S)} poses recorded, not fitted yet"
        spm, sign = self.steps_per_mm()
        lines = [f"{len(S)} poses, {self.kind} fit: steps/mm x {spm[0]:.2f}, y {spm[1]:.2f}, "
                 f"z {spm[2]:.2f} (signs {sign.tolist()})"]
        if self.residual_mm is not None and len(self.residual_mm):
            lines.append(f"  pose mismatch: rms {np.sqrt(np.mean(self.residual_mm ** 2)):.2f} mm, "
                         f"worst {self.residual_mm.max():.2f} mm (pose {int(self.residual_mm.argmax()) + 1})")
        return "\n".join(lines)

    # ---- save / load
    def to_dict(self):
        return dict(version=VERSION, created=self.created, kind=self.kind,
                    A=None if self.A is None else self.A.tolist(),
                    b=None if self.b is None else self.b.tolist(), points=self.points,
                    known_spm=None if self.known_spm is None else self.known_spm.tolist(),
                    known_sign=None if self.known_sign is None else self.known_sign.tolist(),
                    residual_mm=None if self.residual_mm is None else self.residual_mm.tolist())

    def save(self, path):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        pm = cls(A=d.get("A"), b=d.get("b"), kind=d.get("kind"), points=d.get("points"),
                 known_spm=d.get("known_spm"), known_sign=d.get("known_sign"))
        pm.created = d.get("created", pm.created)
        if d.get("residual_mm") is not None:
            pm.residual_mm = np.asarray(d["residual_mm"], float)
        return pm
