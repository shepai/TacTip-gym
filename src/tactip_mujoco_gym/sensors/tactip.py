"""
TacTip readouts: the marker (pin) displacements, and the camera inside the sensor.

Both work on every robot, on every mounted copy of the sensor (the arm's one TacTip, the
quadruped's four feet, the humanoid's two hands): they find their sites / cameras by each
mount's name prefix, so any sensor hardware that uses the same names works too.
"""
import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, frame_pose, register_sensor


@register_sensor("tactip_markers")
class TacTipMarkers(Sensor):
    """TacTip marker displacement from rest and its change since the last step (mm), all mounts.

    marker_prefix  sites read as markers: every site named <mount prefix><marker_prefix>*
                   (default: env.MARKER_PREFIX, "s_c")
    frame_site     readings are in this site's frame on each mount, <prefix><frame_site>
                   (default: env.SENSOR_SITE, "ee_site"); the mount's own frame if missing
    scale          metres per output unit (default: env.MARKER_SCALE, 1e-3 = mm)
    clip           outputs are clipped to +-clip (default: env.MARKER_CLIP, 20)
    velocity       also output the change since the previous step (default True)

    Output: [displacement (n_markers x 3), change (n_markers x 3)], markers grouped by mount.
    """
    key = "tactile"
    encoder = "mlp"

    def __init__(self, marker_prefix=None, frame_site=None, scale=None, clip=None, velocity=True):
        self.marker_prefix = marker_prefix
        self.frame_site = frame_site
        self.scale = scale
        self.clip = clip
        self.velocity = bool(velocity)

    # ------------------------------------------------------------------ setup
    def bind(self, env):
        m = env.model
        prefix = self.marker_prefix or getattr(env, "MARKER_PREFIX", "s_c")
        frame_site = self.frame_site or getattr(env, "SENSOR_SITE", "ee_site")
        self._scale = float(getattr(env, "MARKER_SCALE", 1e-3) if self.scale is None else self.scale)
        self._clip = float(getattr(env, "MARKER_CLIP", 20.0) if self.clip is None else self.clip)
        mounts = list(getattr(env, "sensor_mounts", []))
        if not mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")
        names = [m.site(i).name for i in range(m.nsite)]
        ids, owner, frames, counts = [], [], [], []
        for k, mt in enumerate(mounts):
            pre = mt.prefix + prefix
            mk = [i for i, n in enumerate(names) if n.startswith(pre)]
            ids += mk
            owner += [k] * len(mk)
            counts.append(len(mk))
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, mt.prefix + frame_site)
            frames.append(("site", sid) if sid >= 0 else tuple(mt.frame))
        if not ids:
            raise SensorUnavailable(
                f"no '{prefix}*' marker sites on the mounted sensor "
                f"(looked for {', '.join(repr(mt.prefix + prefix + '*') for mt in mounts)})")
        self._ids = np.array(ids, dtype=int)
        self._owner = np.array(owner, dtype=int)
        self._frames = frames
        self.counts = counts                     # markers per mount
        self.n_markers = len(ids)
        rest = np.zeros((self.n_markers, 3), dtype=np.float32)
        self._rest, self._last, self._delta = rest, rest.copy(), rest.copy()

    def nodes(self, data):
        """Marker positions in their mount's frame, (n_markers, 3), metres."""
        p = data.site_xpos[self._ids]
        if len(self._frames) == 1:
            c, R = frame_pose(data, self._frames[0])
            return ((p - c) @ R).astype(np.float32)          # row-wise R.T @ (p - c)
        pos, mat = zip(*(frame_pose(data, f) for f in self._frames))
        c = np.array(pos)[self._owner]
        R = np.array(mat)[self._owner]
        return np.einsum("ni,nij->nj", p - c, R).astype(np.float32)

    # ------------------------------------------------------------------ life cycle
    def calibrate(self, env, data):
        self._rest = self.nodes(data)
        self._last = self._rest.copy()
        self._delta = np.zeros_like(self._rest)

    def reset(self, env):
        self._last = self.nodes(env.data)
        self._delta = np.zeros_like(self._last)

    def on_select(self, env, active):
        if active:                  # switched on mid-episode: start the history from now
            self.reset(env)

    def env_step(self, env):
        nodes = self.nodes(env.data)
        self._delta = nodes - self._last
        self._last = nodes

    def observe(self, env):
        disp = (self._last - self._rest) / self._scale
        parts = [disp.ravel()]
        if self.velocity:
            parts.append((self._delta / self._scale).ravel())
        return np.clip(np.concatenate(parts), -self._clip, self._clip).astype(np.float32)

    # ------------------------------------------------------------------ extras
    @property
    def rest(self):
        return self._rest

    def displacement(self, env):
        """Current marker displacement from rest in metres: (n_mounts, markers per mount, 3)
        when every mount has the same marker count, else a list with one array per mount."""
        disp = self.nodes(env.data) - self._rest
        if len(set(self.counts)) == 1:
            return disp.reshape(len(self.counts), -1, 3)
        return np.split(disp, np.cumsum(self.counts)[:-1])


@register_sensor("tactip_camera")
class TacTipCamera(Sensor):
    """Image from the camera inside each TacTip (looking at the pins), uint8 (H, W, 3 x mounts).

    camera   camera name on each mount, <prefix><camera> (default: env.SENSOR_CAM, "sensor_cam")
    width, height   image size (default: env.TACTILE_IMAGE_WIDTH / _HEIGHT, 64 x 64)
    mode     "rgb" or "gray"
    groups   geom / site groups drawn (default (2,): the pins, not the scene outside)

    With several mounts (feet, hands) the images are stacked along the channel axis.
    """
    key = "tactile_image"
    encoder = "cnn"

    def __init__(self, camera=None, width=None, height=None, mode="rgb", groups=(2,)):
        self.camera = camera
        self.width = width
        self.height = height
        self.mode = mode
        self.groups = tuple(groups)

    def bind(self, env):
        if self.mode not in ("rgb", "gray"):
            raise ValueError(f"tactip_camera mode must be 'rgb' or 'gray', not {self.mode!r}")
        m = env.model
        cam = self.camera or getattr(env, "SENSOR_CAM", "sensor_cam")
        mounts = list(getattr(env, "sensor_mounts", []))
        if not mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")
        self._cams = []
        for mt in mounts:
            name = mt.prefix + cam
            if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, name) < 0:
                raise SensorUnavailable(f"camera {name!r} not found on the mounted sensor")
            self._cams.append(name)
        self._w = int(self.width or getattr(env, "TACTILE_IMAGE_WIDTH", 64))
        self._h = int(self.height or getattr(env, "TACTILE_IMAGE_HEIGHT", 64))
        self._renderer = None
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 0
        opt.sitegroup[:] = 0
        for g in self.groups:
            opt.geomgroup[g] = 1
            opt.sitegroup[g] = 1
        self._opt = opt

    def observe(self, env):
        if self._renderer is None:
            r = mujoco.Renderer(env.model, height=self._h, width=self._w)
            flags = r.scene.flags
            flags[mujoco.mjtRndFlag.mjRND_SKYBOX] = 0
            flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
            self._renderer = r
        r = self._renderer
        imgs = []
        for name in self._cams:
            r.update_scene(env.data, camera=name, scene_option=self._opt)
            img = r.render()
            if self.mode == "gray":
                gray = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
                img = np.clip(gray, 0, 255).astype(np.uint8)[..., None]
            imgs.append(img.copy())
        return imgs[0] if len(imgs) == 1 else np.concatenate(imgs, axis=-1)

    def flatten(self, value):
        return (np.asarray(value, dtype=np.float32) / 255.0).ravel()

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
