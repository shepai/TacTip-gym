"""
Contact depth readouts: a camera inside the sensor renders how far objects come into the
space its sensing surface fills at rest. This is how Tactile Gym (Church et al., 2021) and
Tactile Gym 2.0 (Lin et al., 2022) simulate the TacTip, and with a flat gel it is a GelSight's
depth map; "gelsight" mode shades that map with three coloured lights, as the real sensor's
image does.

Works with any sensor hardware that has, on each mount,
    a camera  <prefix>depth_cam   inside the sensor, looking out through the surface
    a geom    <prefix>surface     the sensing surface (sphere, ellipsoid, box or cylinder)
like the "gelsight" and "tactip_depth" models (sensors/hardware.py). The surface may move
(it is on springs there); the camera must not, and the rest shape the depth is measured from
is the surface with its joints at zero.

Only geoms that can collide are drawn, and none of the sensor's own, so the image shows what
touches it, not the decoration around it.
"""
import math

import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, register_sensor, subtree_geoms

MODES = ("depth", "gelsight")
_G = mujoco.mjtGeom


# ----------------------------------------------------------------------------- ray geometry
def pixel_rays(fovy_deg, width, height):
    """Unit ray directions through each pixel centre, in the camera frame (it looks along
    -z, y up; row 0 is the top of the image): shape (height, width, 3)."""
    t = math.tan(math.radians(fovy_deg) / 2)
    x = (2 * (np.arange(width) + 0.5) / width - 1) * t * width / height
    y = (1 - 2 * (np.arange(height) + 0.5) / height) * t
    X, Y = np.meshgrid(x, y)
    v = np.stack([X, Y, -np.ones_like(X)], axis=-1)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def _slab(o, d, h):
    """Entry and exit distances of rays (origin coordinate o, direction components d) through
    the slab -h <= x <= h, and where they never cross it (parallel and outside)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-h - o) / d, (h - o) / d
    flat = d == 0
    inside = abs(o) <= h
    lo = np.where(flat, -np.inf, np.minimum(t1, t2))
    hi = np.where(flat, np.inf, np.maximum(t1, t2))
    return lo, hi, flat & ~inside


def exit_distance(gtype, size, o, D):
    """Distance along each ray (origin o (3,), unit directions D (n, 3), geom frame) to
    where it leaves the geom: the far side of a box or slab seen from behind, the inside of
    a dome seen from its centre. NaN where the ray misses it or it is behind the origin."""
    n = len(D)
    lo, hi, miss = np.full(n, -np.inf), np.full(n, np.inf), np.zeros(n, bool)
    if gtype == _G.mjGEOM_BOX:
        for a in range(3):
            enter, leave, never = _slab(o[a], D[:, a], size[a])
            lo, hi, miss = np.maximum(lo, enter), np.minimum(hi, leave), miss | never
    elif gtype in (_G.mjGEOM_SPHERE, _G.mjGEOM_ELLIPSOID, _G.mjGEOM_CYLINDER):
        cyl = gtype == _G.mjGEOM_CYLINDER
        if gtype == _G.mjGEOM_ELLIPSOID:
            r = np.asarray(size[:3], float)
        else:
            r = np.array([size[0], size[0], size[0]])
        axes = [0, 1] if cyl else [0, 1, 2]                 # a cylinder: round in x, y
        oo, DD = o[axes] / r[axes], D[:, axes] / r[axes]
        a2 = (DD * DD).sum(1)
        b = DD @ oo
        c = oo @ oo - 1.0
        disc = b * b - a2 * c
        along = a2 < 1e-12                                  # (a cylinder: along its axis)
        miss |= (disc < 0) | (along & (c > 0))
        root = np.sqrt(np.maximum(disc, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            lo = np.where(along, -np.inf, (-b - root) / a2)
            hi = np.where(along, np.inf, (-b + root) / a2)
        if cyl:
            enter, leave, never = _slab(o[2], D[:, 2], size[1])
            lo, hi, miss = np.maximum(lo, enter), np.minimum(hi, leave), miss | never
    else:
        raise SensorUnavailable(
            "the sensing surface must be a sphere, ellipsoid, box or cylinder geom")
    ok = ~miss & np.isfinite(hi) & (hi >= np.maximum(lo, 0.0))
    return np.where(ok, hi, np.nan)


def gaussian_blur(img, sigma):
    """Separable Gaussian blur of a 2-D array (sigma in pixels), edges repeated."""
    if sigma <= 0:
        return img
    r = max(1, int(math.ceil(2.5 * sigma)))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / sigma) ** 2)
    k /= k.sum()
    out = np.pad(img, r, mode="edge")
    out = np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 0, out)
    return np.apply_along_axis(lambda v: np.convolve(v, k, mode="valid"), 1, out)


# directions towards the three lights (z: towards the camera) and their colours
_LIGHTS = np.array([[math.cos(a) * math.cos(math.radians(30)), math.sin(a) * math.cos(math.radians(30)),
                     math.sin(math.radians(30))] for a in np.radians([90.0, 210.0, 330.0])])
_COLOURS = np.eye(3)                                       # red, green, blue
_BACKGROUND = np.array([0.42, 0.44, 0.47])


def shade_gelsight(depth, pitch, gain=0.9, background=_BACKGROUND):
    """A GelSight-like RGB image (float, 0..1) of a contact depth map (m) whose pixels are
    `pitch` metres apart: three coloured lights from around the gel light its slopes, flat
    gel shows the background colour."""
    gy, gx = np.gradient(depth)                         # along image rows / columns
    gx, gy = gx / pitch, gy / pitch
    # normal of the gel's inner face, towards the camera (z), which `depth` pushes it along
    n = np.stack([-gx, -gy, np.ones_like(depth)], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True)
    lit = n @ _LIGHTS.T - _LIGHTS[:, 2]                    # change from flat gel, per light
    img = background + gain * lit @ _COLOURS
    return np.clip(img, 0.0, 1.0)


# ----------------------------------------------------------------------------- readouts
@register_sensor("contact_depth")
class ContactDepth(Sensor):
    """Contact depth image from the camera inside each sensor, uint8 (H, W, C x mounts).

    camera, surface  per-mount names of the camera and the sensing surface geom
                     (default "depth_cam", "surface")
    width, height    image size (64 x 64)
    mode             "depth": (H, W, 1), contact depth 0 .. max_depth -> 0 .. 255
                     "gelsight": (H, W, 3), the depth map lit by three coloured lights
    max_depth        contact depth (m) that reads 255
    blur             Gaussian blur of the depth map in pixels (the elastomer smooths edges)
    gain             strength of the gelsight shading

    Several mounts (gripper fingers, feet) are stacked along the channel axis. `depths`
    holds the last depth maps (m), one per mount.
    """
    key = "tactile_image"
    encoder = "cnn"

    def __init__(self, camera="depth_cam", surface="surface", width=64, height=64, mode="depth",
                 max_depth=0.004, blur=0.7, gain=0.9):
        self.camera = camera
        self.surface = surface
        self.width = width
        self.height = height
        self.mode = mode
        self.max_depth = max_depth
        self.blur = blur
        self.gain = gain

    # ------------------------------------------------------------------ setup
    def bind(self, env):
        if self.mode not in MODES:
            raise ValueError(f"contact_depth mode must be one of {MODES}, not {self.mode!r}")
        m = env.model
        mounts = list(getattr(env, "sensor_mounts", []))
        if not mounts:
            raise SensorUnavailable("no sensor is mounted on this robot")
        self._w, self._h = int(self.width), int(self.height)
        cams, surfs = [], []
        for mt in mounts:
            c = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, mt.prefix + self.camera)
            g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, mt.prefix + self.surface)
            if c < 0 or g < 0:
                raise SensorUnavailable(
                    f"no camera {mt.prefix + self.camera!r} and surface geom "
                    f"{mt.prefix + self.surface!r} on the mounted sensor (a contact depth sensor "
                    "needs both, e.g. sensor_xml='gelsight' or 'tactip_depth')")
            cams.append(c)
            surfs.append(g)
        self._cams, self._surfs = cams, surfs
        # geoms the camera doesn't draw: each sensor's own, and everything that can't touch
        own = [set(subtree_geoms(m, mt.body, colliding=False).tolist()) for mt in mounts]
        idle = (m.geom_contype == 0) & (m.geom_conaffinity == 0)
        self._hide = []
        for k in range(len(mounts)):
            mask = idle.copy()
            mask[list(own[k])] = True
            self._hide.append(mask)
        # the rest surface, measured with every joint at zero
        d = mujoco.MjData(m)
        # Camera poses are computed in MuJoCo's camera/light stage (mj_camlight),
        # which is not run by mj_kinematics alone. A full forward pass is required
        # before reading data.cam_xpos / data.cam_xmat for the pixel-ray geometry.
        mujoco.mj_forward(m, d)
        self._ref = [self._reference(m, d, c, g) for c, g in zip(cams, surfs, strict=True)]
        for k, ref in enumerate(self._ref):
            if not np.isfinite(ref).any():
                raise SensorUnavailable(f"the camera on mount {mounts[k].name!r} doesn't see "
                                        "its sensing surface")
        # pixel pitch on the surface (for the gelsight shading), and a near plane that keeps
        # the surface (and anything pushed into it) in view
        t = math.tan(math.radians(float(m.cam_fovy[cams[0]])) / 2)
        self._pitch = [np.where(np.isfinite(r), r, np.nanmedian(r)) * 2 * t / self._h for r in self._ref]
        nearest = min(float(np.nanmin(r)) for r in self._ref)
        extent = float(m.stat.extent)
        m.vis.map.znear = min(float(m.vis.map.znear), 0.2 * nearest / extent)
        opt = mujoco.MjvOption()
        opt.geomgroup[:] = 1
        opt.sitegroup[:] = 0
        opt.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = 0
        self._opt = opt
        self._renderer = None
        self.depths = [np.zeros((self._h, self._w), np.float32) for _ in mounts]

    def _reference(self, m, d, cam, geom):
        """z-depth (m) of the rest surface through each pixel; NaN off the surface."""
        rays = pixel_rays(float(m.cam_fovy[cam]), self._w, self._h).reshape(-1, 3)
        Rc = d.cam_xmat[cam].reshape(3, 3)
        Rg = d.geom_xmat[geom].reshape(3, 3)
        D = rays @ Rc.T @ Rg                                 # rays in the geom frame
        o = (d.cam_xpos[cam] - d.geom_xpos[geom]) @ Rg
        t = exit_distance(int(m.geom_type[geom]), m.geom_size[geom], o, D)
        return (t * -rays[:, 2]).reshape(self._h, self._w).astype(np.float64)   # z-depth

    # ------------------------------------------------------------------ reading
    def _render(self, env):
        new = self._renderer is None
        if new:
            q = env.model.vis.quality
            samples = q.offsamples
            q.offsamples = 0                         # depth: no multisampling
            try:
                self._renderer = mujoco.Renderer(env.model, height=self._h, width=self._w)
            finally:
                q.offsamples = samples
            self._renderer.enable_depth_rendering()
        r = self._renderer
        sync = getattr(env, "_sync_renderer", None)
        if sync is not None:                         # heightfields changed this episode
            sync(r, new)
        geom_type = int(mujoco.mjtObj.mjOBJ_GEOM)
        out = []
        for k, cam in enumerate(self._cams):
            r.update_scene(env.data, camera=cam, scene_option=self._opt)
            hide = self._hide[k]
            scn = r.scene
            away = env.data.cam_xpos[cam] + 1e3 * env.data.cam_xmat[cam].reshape(3, 3)[:, 2]
            for i in range(scn.ngeom):               # behind the camera: not drawn
                g = scn.geoms[i]
                if g.objtype == geom_type and 0 <= g.objid < len(hide) and hide[g.objid]:
                    g.pos[:] = away
            z = r.render()
            pen = np.nan_to_num(self._ref[k] - z, nan=0.0)
            pen = np.clip(pen, 0.0, None)
            if self.blur:
                pen = gaussian_blur(pen, float(self.blur))
            out.append(np.clip(pen, 0.0, self.max_depth).astype(np.float32))
        self.depths = out
        return out

    def observe(self, env):
        imgs = []
        for k, pen in enumerate(self._render(env)):
            if self.mode == "gelsight":
                rgb = shade_gelsight(pen, self._pitch[k], self.gain)
                imgs.append((rgb * 255).astype(np.uint8))
            else:
                imgs.append((pen / self.max_depth * 255).astype(np.uint8)[..., None])
        return imgs[0] if len(imgs) == 1 else np.concatenate(imgs, axis=-1)

    def flatten(self, value):
        return (np.asarray(value, dtype=np.float32) / 255.0).ravel()

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None


@register_sensor("contact_map")
class ContactMap(ContactDepth):
    """Contact depth on a coarse grid (mm), float32, for an MLP: the contact_depth image
    averaged over cell x cell pixel blocks, every mount in turn (16 x 16 per mount at the
    default 64 x 64 and cell 4). Arguments as contact_depth, plus `cell`."""
    key = "tactile"
    encoder = "mlp"

    def __init__(self, camera="depth_cam", surface="surface", width=64, height=64, cell=4,
                 max_depth=0.004, blur=0.7):
        super().__init__(camera=camera, surface=surface, width=width, height=height,
                         mode="depth", max_depth=max_depth, blur=blur)
        self.cell = cell

    def bind(self, env):
        super().bind(env)
        c = int(self.cell)
        if self._h % c or self._w % c:
            raise ValueError(f"contact_map: width and height must be multiples of cell ({c})")
        self.grid = (self._h // c, self._w // c)

    def observe(self, env):
        c = int(self.cell)
        h, w = self.grid
        cells = [p.reshape(h, c, w, c).mean(axis=(1, 3)) for p in self._render(env)]
        return (np.concatenate([x.ravel() for x in cells]) * 1000.0).astype(np.float32)

    def flatten(self, value):
        return np.asarray(value, dtype=np.float32).ravel()
