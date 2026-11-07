"""
Camera readout: an image from any MuJoCo camera in the model (the arm's hand camera, the
quadruped's and the humanoid's head cameras, or one you add).
"""
import mujoco
import numpy as np

from .base import Sensor, SensorUnavailable, register_sensor

MODES = ("rgb", "gray", "depth")


@register_sensor("camera")
class Camera(Sensor):
    """Image from a MuJoCo camera: uint8 (H, W, 3) rgb / (H, W, 1) gray, or float32 depth (m).

    camera      camera name (default: env.VISION_CAMERA, e.g. "handcam" / "headcam")
    width, height   image size (default: env.VISION_WIDTH / VISION_HEIGHT, 64 x 64). Keep it
                small: in the flat format every pixel is an input (SB3's default CNN in the
                dict format needs at least 36)
    mode        "rgb", "gray" or "depth" (default: env.VISION_MODE)
    max_depth   depth is divided by this in the flat format (default: env.VISION_MAX_DEPTH)
    shadows, antialias   render shadows / reflections and multisampling (default off: most
                of the render time for little information at 64 px)

    Every argument left as None takes the env's class constant, so tasks that override
    VISION_CAMERA etc. keep working. Uses its own renderer, separate from any viewing one.
    """
    key = "vision"
    encoder = "cnn"

    def __init__(self, camera=None, width=None, height=None, mode=None, max_depth=None,
                 shadows=None, antialias=None):
        self.camera = camera
        self.width = width
        self.height = height
        self.mode = mode
        self.max_depth = max_depth
        self.shadows = shadows
        self.antialias = antialias

    def _opt(self, env, name, attr, default):
        v = getattr(self, name)
        return getattr(env, attr, default) if v is None else v

    def bind(self, env):
        self._camera = self._opt(env, "camera", "VISION_CAMERA", "handcam")
        self._w = int(self._opt(env, "width", "VISION_WIDTH", 64))
        self._h = int(self._opt(env, "height", "VISION_HEIGHT", 64))
        self._mode = self._opt(env, "mode", "VISION_MODE", "rgb")
        self._max_depth = float(self._opt(env, "max_depth", "VISION_MAX_DEPTH", 1.0))
        self._shadows = bool(self._opt(env, "shadows", "VISION_SHADOWS", False))
        self._antialias = bool(self._opt(env, "antialias", "VISION_ANTIALIAS", False))
        self._renderer = None
        if self._mode not in MODES:
            raise ValueError(f"camera mode must be one of {MODES}, not {self._mode!r}")
        if mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, self._camera) < 0:
            raise SensorUnavailable(f"Camera '{self._camera}' not found in the model")

    def _make_renderer(self, model):
        q = model.vis.quality
        samples = q.offsamples                      # read when the GL context is made
        if not self._antialias:
            q.offsamples = 0
        try:
            r = mujoco.Renderer(model, height=self._h, width=self._w)
        finally:
            q.offsamples = samples                  # leave other renderers (videos) unchanged
        if not self._shadows:
            r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
            r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
        if self._mode == "depth":
            r.enable_depth_rendering()
        return r

    def observe(self, env):
        new = self._renderer is None
        if new:
            self._renderer = self._make_renderer(env.model)
        r = self._renderer
        sync = getattr(env, "_sync_renderer", None)
        if sync is not None:                        # e.g. TacQuad re-uploads its terrain
            sync(r, new)
        r.update_scene(env.data, camera=self._camera)
        img = r.render()
        if self._mode == "depth":
            return img.astype(np.float32)[..., None]
        if self._mode == "gray":
            gray = img.astype(np.float32) @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
            return np.clip(gray, 0, 255).astype(np.uint8)[..., None]
        return img.copy()

    def flatten(self, value):
        v = np.asarray(value)
        if v.dtype == np.uint8:
            return (v.astype(np.float32) / 255.0).ravel()
        return np.clip(v / self._max_depth, 0.0, 1.0).astype(np.float32).ravel()

    def close(self):
        if getattr(self, "_renderer", None) is not None:
            self._renderer.close()
            self._renderer = None
