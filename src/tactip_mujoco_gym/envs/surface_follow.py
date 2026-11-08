"""
TacArm-surface: slide the sensor along a wavy strip, keeping it pressed in by the same depth
and (by default) square to the surface.

    env = gym.make("TacArm-surface", obs_modes=["tactile"])
    env = gym.make("TacArm-surface", wave="chirp", texture="patches")

The strip (STRIP_LENGTH x STRIP_WIDTH, a heightfield) is redrawn every episode:

    wave     "uniform"  one sine: the same amplitude and wavelength all along
             "chirp"    wavelength and amplitude change steadily along the strip
             "multi"    two or three sines added (an irregular profile)
             "flat", "random" (default: one of the three waves), or a list to draw from
    texture  None (default, smooth), a dataset texture 1..20 all along (a uniform texture),
             "random" (one of TEXTURES all along), "patches" (a different one every
             TEXTURE_PATCH metres: a non-uniform texture), or a list to draw one from.
             Textures are the 3D Printable Tactile Dataset's at the printed scale
             (see textures.py; TEXTURE_SCALE scales their relief).

The waves vary along the strip only, so square to the surface means pitching the sensor.
The depth and the surface normal are measured from the wave (the surface the task asks to
follow); a texture is relief on it that the sensor feels but doesn't have to trace.

Control: the tip advances along the strip by itself (SPEED per step). Actions in [-1, 1]:
    follow_normal=True  (2) depth (tip up / down), pitch rate of the sensor
    follow_normal=False (1) depth only; the sensor stays vertical
The depth to keep is set at the start: the sensor is lowered onto the flat start of the strip
and pressed in by PRESS.

Reward per step: closeness to that depth (W_DEPTH), sensor axis along the surface normal
(W_NORMAL, follow_normal only), and progress along the strip while touching it (W_PROG).
Success: reach the end of the strip. Fail: lift off by LOST_UP or press in LOST_DOWN deeper.

privileged input: depth error (cm), sensor pitch, the pitch square to the surface, their
difference, angle to the normal, the surface height 1, 2 and 4 cm ahead (cm, relative to
under the tip), progress along the strip, touching.
"""
import mujoco
import numpy as np

from tactip_mujoco_gym import textures as tx

from .base import HeightfieldMixin
from .robot_arm import RobotArmEnv, _tol


class SurfaceFollow(HeightfieldMixin, RobotArmEnv):
    # ---- the strip ----
    STRIP_X0      = 0.22          # m: where the strip starts (it runs along +x)
    STRIP_LENGTH  = 0.30
    STRIP_Y       = 0.0
    STRIP_WIDTH   = 0.06
    STRIP_TOP     = 0.08          # m: mean height of the surface
    SURFACE_RES   = 0.0005        # m between heightfield nodes
    HF_RANGE      = 0.04          # m: the surface stays within STRIP_TOP +- HF_RANGE / 2
    HF_BASE       = 0.005
    FLAT_IN       = 0.04          # m of flat strip at the start (the sensor starts in it) ...
    RAMP_IN, RAMP_OUT = 0.03, 0.03     # ... then the waves ease in; they ease out at the end
    WAVE_AMP      = (0.002, 0.006)     # m: amplitude range (curvature stays within what the
    WAVE_LENGTH   = (0.08, 0.16)       # m: wavelength range   TacTip's dome can follow)
    TEXTURES      = tx.DATASET         # what texture="random" / "patches" draw from
    TEXTURE_SCALE = 1.0                # x the printed relief
    TEXTURE_PATCH = 0.05               # m per texture with texture="patches"
    SURFACE_RGBA  = (0.55, 0.6, 0.65, 1.0)

    # ---- control ----
    SPEED       = 0.002           # m per step along the strip (0.1 m/s)
    Z_SCALE     = 0.0015          # m per step at |action| = 1
    PITCH_RATE  = 0.03            # rad per step at |action| = 1
    PITCH_MAX   = 0.6             # rad
    PRESS       = 0.0015          # m pressed in at the start: the depth to keep
    START_AT    = 0.02            # m along the strip where the sensor starts
    END_MARGIN  = 0.03            # m before the end of the strip that counts as the end

    # ---- reward / end ----
    SIG_DEPTH = 0.002             # m
    SIG_ANGLE = 0.15              # rad
    W_DEPTH, W_NORMAL, W_PROG = 0.4, 0.3, 0.3
    LOST_UP   = 0.008             # m further out than the depth to keep: lost contact
    LOST_DOWN = 0.008             # m deeper: pressed too deep
    MAX_STEPS = 250

    WS_LO = np.array([0.15, -0.10, 0.04])
    WS_HI = np.array([0.60,  0.10, 0.25])

    def __init__(self, wave="random", texture=None, follow_normal=True, **kwargs):
        self.wave = wave
        self.texture = texture
        self.follow_normal = bool(follow_normal)
        self._check_options(wave, texture)
        super().__init__(xml_subpath=("assets", "tactip_arm.xml"),
                         action_dim=2 if self.follow_normal else 1, **kwargs)

    @staticmethod
    def _check_options(wave, texture):
        waves = ("flat", "random") + tx.WAVES
        for w in (wave if isinstance(wave, (list, tuple)) else [wave]):
            if w not in waves:
                raise ValueError(f"wave {w!r}: choose from {waves} (or a list of them)")
        for t in (texture if isinstance(texture, (list, tuple)) else [texture]):
            if t not in (None, "random", "patches") and int(t) not in tx.TEXTURE_NAMES:
                raise ValueError(f"texture {t!r}: None, 'random', 'patches', 1..20 or a list")

    # ------------------------------------------------------------------ scene
    def _edit_spec(self, spec):
        L, W = self.STRIP_LENGTH, self.STRIP_WIDTH
        ncol = int(round(L / self.SURFACE_RES)) + 1
        nrow = int(round(W / self.SURFACE_RES)) + 1
        spec.add_hfield(name="surface", size=[L / 2, W / 2, self.HF_RANGE, self.HF_BASE],
                        nrow=nrow, ncol=ncol, userdata=[0.5] * (nrow * ncol))
        bottom = self.STRIP_TOP - self.HF_RANGE / 2                # heightfield's zero level
        body = spec.worldbody.add_body(name="surface", pos=[self.STRIP_X0 + L / 2, self.STRIP_Y, 0])
        body.add_geom(name="surface_base", type=mujoco.mjtGeom.mjGEOM_BOX,
                      size=[L / 2, W / 2, bottom / 2], pos=[0, 0, bottom / 2],
                      rgba=[0.35, 0.35, 0.4, 1.0])
        body.add_geom(name="surface_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="surface",
                      pos=[0, 0, bottom], rgba=list(self.SURFACE_RGBA), condim=3,
                      friction=[0.6, 0.005, 0.0001])

    def _setup_task(self):
        m = self.model
        self._hf = m.hfield("surface").id
        self._surface_body = m.body("surface").id
        self._x0, self._x1 = self.STRIP_X0, self.STRIP_X0 + self.STRIP_LENGTH
        y0, y1 = self.STRIP_Y - self.STRIP_WIDTH / 2, self.STRIP_Y + self.STRIP_WIDTH / 2
        self._xs = np.linspace(self._x0, self._x1, int(m.hfield_ncol[self._hf]))
        self._ys = np.linspace(y0, y1, int(m.hfield_nrow[self._hf]))
        self._axis_local = np.array([0.0, 0.0, -1.0])    # sensor axis in the ee_site frame
        self._pitch_cmd = 0.0
        self._z_ref = 0.012
        self._x_prev = self._x0
        self._make_surface(np.random.default_rng(0), "flat", None)
        self._st = None

    def _hfield_ids(self):
        return (self._hf,)

    def _choose(self, value, rng, pool):
        if isinstance(value, (list, tuple)):
            value = value[int(rng.integers(len(value)))]
        if value == "random":
            value = pool[int(rng.integers(len(pool)))]
        return value

    def _make_surface(self, rng, wave, texture):
        xs, ys = self._xs, self._ys
        s = xs - xs[0]
        h, params = tx.wave_profile(wave, s, rng, self.WAVE_AMP, self.WAVE_LENGTH)
        win = tx.smooth_window(s, self.RAMP_IN, self.RAMP_OUT, flat_in=self.FLAT_IN)
        H = np.repeat((h * win)[None, :], len(ys), axis=0)
        self._base = tx.Bilinear(H + self.STRIP_TOP, xs[0], xs[-1], ys[0], ys[-1])  # the wave
        ids = []
        if texture is not None:
            X, Y = np.meshgrid(s, ys - ys[0])
            if texture == "patches":
                patch = np.minimum((s // self.TEXTURE_PATCH).astype(int),
                                   int(np.ceil(s[-1] / self.TEXTURE_PATCH)) - 1)
                T = np.zeros_like(H)
                for p in range(int(patch.max()) + 1):
                    ids.append(int(self.TEXTURES[int(rng.integers(len(self.TEXTURES)))]))
                    cols = patch == p
                    T[:, cols] = tx.texture_patch(ids[-1], X[:, cols], Y[:, cols], self.TEXTURE_SCALE)
            else:
                ids.append(int(self._choose(texture, rng, self.TEXTURES)))
                T = tx.texture_patch(ids[0], X, Y, self.TEXTURE_SCALE)
            H = H + T * win[None, :]
        H = H + self.STRIP_TOP
        self._surf = tx.Bilinear(H, xs[0], xs[-1], ys[0], ys[-1])
        tx.write_hfield(self.model, self._hf, H, self.HF_RANGE,
                        offset=self.STRIP_TOP - self.HF_RANGE / 2)
        self._hfields_changed()
        self.surface_info = dict(wave=wave, textures=tuple(ids), **params)

    # ------------------------------------------------------------------ geometry
    def _normal_gap(self, p):
        """Distance of point p from the wave along its normal (the tangent plane under p)."""
        n = self._base.normal(p[0], p[1])
        return float((p - np.array([p[0], p[1], float(self._base(p[0], p[1]))])) @ n)

    def _state(self):
        tip = self._tip_pos()
        n = self._base.normal(tip[0], tip[1])
        axis = self.data.site_xmat[self._ee_site].reshape(3, 3) @ self._axis_local
        dx = float(tip[0] - self._x_prev)
        self._x_prev = float(tip[0])
        return dict(tip=tip, err=self._normal_gap(tip) - self._z_ref, normal=n,
                    angle=float(np.arccos(np.clip(-axis @ n, -1.0, 1.0))),
                    pitch=float(np.arctan2(-axis[0], -axis[2])),
                    pitch_goal=float(np.arctan2(n[0], n[2])),
                    dx=dx, touch=bool(self._touching(self._surface_body)))

    def _pitch_quat(self, pitch):
        q = np.array([np.cos(pitch / 2), 0.0, np.sin(pitch / 2), 0.0])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, q, np.asarray(self.TARGET_QUAT, float))
        return out

    # ------------------------------------------------------------------ episode
    def _reset_task(self, options):
        rng = self.np_random
        wave = self._choose(options.get("wave", self.wave), rng, tx.WAVES)
        texture = options.get("texture", self.texture)
        self._check_options(wave, texture)
        self._make_surface(rng, wave, texture)
        x, y = self._x0 + self.START_AT, self.STRIP_Y
        self._pitch_cmd = 0.0
        self._place_arm([x, y, float(self._base(x, y)) + 0.03])
        self._approach([0, 0, -1], self._surface_body, step=0.001, max_steps=40, press=self.PRESS)
        tip = self._tip_pos()
        self._z_ref = self._normal_gap(tip)
        self._x_prev = float(tip[0])
        self._st = self._state()

    def _apply_action(self, a):
        t = self._target.copy()
        t[0] += self.SPEED
        t[1] = self.STRIP_Y
        t[2] += a[0] * self.Z_SCALE
        if self.follow_normal:
            self._pitch_cmd = float(np.clip(self._pitch_cmd + a[1] * self.PITCH_RATE,
                                            -self.PITCH_MAX, self.PITCH_MAX))
            self._quat_cmd = self._pitch_quat(self._pitch_cmd)
        self._set_target(t)

    def _post_physics(self):
        self._st = self._state()

    def _status(self):
        st = self._st
        if st["tip"][0] >= self._x1 - self.END_MARGIN:
            return "success"
        if st["err"] > self.LOST_UP:
            return self._fail("lost contact")
        if st["err"] < -self.LOST_DOWN:
            return self._fail("pressed too deep")
        return None

    def _reward(self):
        st = self._st
        depth = _tol(st["err"], self.SIG_DEPTH)
        prog = float(np.clip(st["dx"] / self.SPEED, -1.0, 1.0)) * float(st["touch"])
        if not self.follow_normal:
            return (self.W_DEPTH + self.W_NORMAL) * depth + self.W_PROG * prog
        return (self.W_DEPTH * depth + self.W_NORMAL * _tol(st["angle"], self.SIG_ANGLE)
                + self.W_PROG * prog)

    # ------------------------------------------------------------------ observation
    def _obs_parts(self):
        parts = super()._obs_parts()
        parts["proprio"] = np.concatenate([parts["proprio"], [self._pitch_cmd]])
        return parts

    def _current(self):
        if self._st is None:                  # (observation space built before the first reset)
            self._st = self._state()
        return self._st

    def _privileged_obs(self):
        st = self._current()
        x, y = st["tip"][0], st["tip"][1]
        here = float(self._base(x, y))
        ahead = [float(self._base(min(x + d, self._x1), y)) - here for d in (0.01, 0.02, 0.04)]
        return np.array([st["err"] * 100, st["pitch"], st["pitch_goal"],
                         st["pitch_goal"] - st["pitch"], st["angle"],
                         *(np.array(ahead) * 100),
                         (x - self._x0) / self.STRIP_LENGTH, float(st["touch"])], dtype=np.float32)

    def _task_info(self):
        st = self._current()
        return {"depth_err": float(st["err"]), "normal_angle": st["angle"],
                "progress": float((st["tip"][0] - self._x0) / self.STRIP_LENGTH),
                "touching": st["touch"], "wave": self.surface_info["wave"],
                "textures": self.surface_info["textures"]}

    def get_camera(self, *args, **kwargs):
        self.sync_renderer(self._cam_renderer)
        return super().get_camera(*args, **kwargs)
