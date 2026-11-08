"""
TacRig: the sensor on a Cartesian test rig (a desktop-CNC frame with the sensor where the
spindle would be, after github.com/shepai/Rig-controller), over a texture block of the 3D
Printable Tactile Dataset in its 3D-printed holder.

    env = gym.make("TacRig-texture", obs_modes=["tactile"])           # identify the texture
    env = gym.make("TacRig-v0", textures=[1, 2, 3])                   # the rig, no task
    data = press_grid(gym.make("TacRig-v0"))                          # the paper's protocol

The rig: the sensor rides on three stepper-driven axes, x (front-back), y (left-right, the
gantry carriage) and z (up-down), each a stiff position servo with a force limit (a stepper
stalls). The block sits in the holder's pocket on the plate; the plate can be tilted (the
rig's a axis) by a set angle per episode: reset option "tilt" (rad) or TILT_RANGE. On the
real rig the plate carries the x travel; here the sensor does, which is the same motion
relative to the block (MuJoCo heightfields have to be static, and the block is one).

Blocks (textures.py): the six dataset textures by default (textures=[1..20] or STL files),
29.3 x 30 mm (BLOCK_MM), 7 mm tall, from the generator's equations as a heightfield
(TEXTURE_RES_MM between nodes). PRINT_BLUR_MM / PRINT_NOISE_MM imitate a printer's
smoothing and roughness (drawn per episode). The holder is the dataset's
Holder_textures.stl, measured: a 31.1 x 30 mm pocket 5.8 mm deep, so a block stands about
1.2 mm proud of the rim.

Actions in [-1, 1] (low-pass filtered, ACTION_SMOOTHING): sensor velocity along x, y, z
(AXIS_SPEED at full command). TacRig-texture adds a decision: see TacRigTexture.

Observation inputs (obs_modes); proprioception is always included:
    proprio        axis positions (scaled to their travel), command lead over each axis,
                   the filtered action
    force          the plate's force sensor: normal force between sensor and block / holder
                   (N / F_REF)
    privileged     which texture (one-hot), the tip over the block (cm from its centre), the
                   tip's height above the surface under it (cm, negative = pressed in), tilt
    tactile, tactile_image, vision ("rigcam"), + readouts added with sensors={...}
reset(options=...): "texture" (an entry of env.textures), "start" (x, y in mm from the
block centre), "tilt" (rad).
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from tactip_mujoco_gym import textures as tx
from tactip_mujoco_gym.sensor_tools import attach_sensor, compile_spec, sensor_info
from tactip_mujoco_gym.sensors import Mount, resolve_sensor_model

from .base import HeightfieldMixin, TactileGymEnv

# Holder_textures.stl (github.com/shepai/3D-textures, assets/Print aparatus), measured, in mm,
# in a frame centred on the pocket with the holder's underside at z = 0: (centre, half size)
HOLDER_MM = (
    ((-2.836, -13.000, 0.8235), (22.696, 32.505, 0.8235)),   # floor
    ((-20.541, -13.000, 4.5575), (4.991, 32.505, 2.9105)),   # walls round the pocket
    ((17.705, -13.000, 4.5575), (2.155, 32.505, 2.9105)),
    ((0.0, -30.2525, 4.5575), (15.55, 15.2525, 2.9105)),
    ((0.0, 17.2525, 4.5575), (15.55, 2.2525, 2.9105)),
)
POCKET_FLOOR_MM = 1.647
AXES = ("rig_x", "rig_y", "rig_z")


class TacRig(HeightfieldMixin, TactileGymEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}
    TASK_INPUTS = ("privileged", "force")
    VISION_CAMERA = "rigcam"
    TRACK_BODY, TRACK_DISTANCE, TRACK_ELEVATION = "holder", 0.35, -30   # training videos
    SENSOR_XML    = ("assets", "sensor", "tactip.xml")
    SENSOR_SOURCE = ("assets", "tactip_arm.xml")
    SENSOR_ROOT   = "auto"
    SENSOR_PREFIX = "sensor_"

    TIMESTEP    = 0.001
    FRAME_SKIP  = 20              # 50 Hz
    CTRL_CHUNKS = 4               # axis targets are stepped in this many chunks per env step
    MAX_STEPS   = 500

    # ---- the rig ----
    PLATE_TOP   = 0.05            # m: top of the plate the holder stands on
    X_TRAVEL    = 0.05            # m either side of the block centre
    Y_TRAVEL    = 0.08
    Z_RANGE     = (-0.015, 0.04)  # m of spindle travel; 0 = tip HOVER above the block's top
    HOVER       = 0.010
    SPINDLE_DROP = 0.10           # m from the spindle (z carriage) down to the sensor tip
    AXIS_SPEED  = (0.02, 0.02, 0.01)       # m/s at |action| = 1
    LEASH       = (0.003, 0.003, 0.002)    # m an axis command may lead the axis
    KP          = (2e4, 2e4, 5e4)          # N/m: servo stiffness (steppers: stiff)
    KV          = (400.0, 300.0, 400.0)    # N s/m
    FORCE_LIMIT = (100.0, 100.0, 40.0)     # N: an axis stalls beyond this
    ACTION_SMOOTHING = 0.5
    TILT_AXIS   = (0.0, 1.0, 0.0)          # the plate's a axis (through the pocket centre)
    TILT_RANGE  = (0.0, 0.0)               # rad, drawn per episode (reset option "tilt")

    # ---- blocks ----
    TEXTURES       = tx.DATASET
    BLOCK_MM       = tx.BLOCK_MM
    TEXTURE_RES_MM = 0.25         # mm between heightfield nodes
    PRINT_BLUR_MM  = 0.0          # sigma of a print's smoothing (per episode: 0..this)
    PRINT_NOISE_MM = 0.0          # sd of a print's height noise
    BLOCK_FRICTION = 0.5
    START_XY_MM    = 5.0          # random start within +- this of the block centre
    F_REF          = 5.0          # N: "force" input scale

    def __init__(self, textures=None, obs_modes=None, privileged=None, obs_format="flat",
                 render_mode=None, max_steps=None, sensors=None, sensor_xml=None,
                 sensor_stiffness=1.0, overrides=None):
        """
        textures          the blocks to draw from: dataset texture numbers (1..20) and / or
                          paths of STL blocks (default: TEXTURES, the six of the dataset)
        sensor_xml        the sensor hardware on the rig (default: the TacTip)
        sensor_stiffness  scales the sensor's skin (see sensor_tools.attach_sensor)
        sensors, obs_modes, obs_format, overrides, max_steps: as on every robot
        """
        gym.Env.__init__(self)
        self._apply_overrides(overrides)
        self._init_sensor_state()
        self.textures = tuple(textures) if textures is not None else tuple(self.TEXTURES)
        if not self.textures:
            raise ValueError("textures is empty")
        self.render_mode = render_mode
        self.xml_path = None
        self.sensor_xml = sensor_xml
        self.sensor_stiffness = float(sensor_stiffness)

        spec = self._build_spec(sensor_xml)
        self.model = compile_spec(spec)
        self.data = mujoco.MjData(self.model)
        self._physics = self._renderer = self._view_renderer = self._vision_renderer = None
        self._index_model()
        self.sensor_mounts = [Mount("tip", self.SENSOR_PREFIX, self._sensor_root_id,
                                    ("site", self._tip_site))]
        self._setup_sensors(sensors)

        self.dt = self.TIMESTEP * self.FRAME_SKIP
        self.max_steps = int(max_steps or self.MAX_STEPS)
        self.step_count = 0
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self._n_actions(),), dtype=np.float32)
        self._decide_index = None
        self._ideal = [tx.texture_heights(t, self.TEXTURE_RES_MM, self.BLOCK_MM) for t in self.textures]

        # rest shape of the sensor: raised clear of the block
        mujoco.mj_resetData(self.model, self.data)
        self._set_block(0, np.random.default_rng(0))
        self._set_tilt(0.0)
        self._put_axes(np.array([0.0, 0.0, self.Z_RANGE[1]]), settle=300)
        self._calibrate_sensors()

        self.obs_format = "flat"
        self.obs_modes = {k: False for k in self.input_names}
        self.reset(seed=0)
        self.set_obs_format(obs_format)
        self.set_obs_modes(self.resolve_obs_modes(obs_modes, privileged))

    # ================================================================ model
    def _block_top_mm(self):
        return POCKET_FLOOR_MM + self.BLOCK_MM[2]

    def _scene_xml(self):
        P = self.PLATE_TOP
        top = P + self._block_top_mm() / 1000
        z_spindle = top + self.HOVER + self.SPINDLE_DROP
        gz = z_spindle + 0.04                                   # carriage on the gantry beam
        span = self.Y_TRAVEL + 0.07
        f = lambda v: " ".join(f"{float(x):.6g}" for x in v)   # noqa: E731
        holder = "".join(
            f'<geom name="holder{i}" type="box" pos="{f(np.array(c) / 1000)}" '
            f'size="{f(np.array(h) / 1000)}" material="holder" friction="{self.BLOCK_FRICTION} 0.005 0.0001"/>'
            for i, (c, h) in enumerate(HOLDER_MM))
        servos = "".join(
            f'<general name="{n}" joint="{n}" biastype="affine" gainprm="{self.KP[i]}" '
            f'biasprm="0 {-self.KP[i]} {-self.KV[i]}" ctrllimited="true" ctrlrange="{f(r)}" '
            f'forcelimited="true" forcerange="{-self.FORCE_LIMIT[i]} {self.FORCE_LIMIT[i]}"/>'
            for i, (n, r) in enumerate(zip(AXES, self._ranges(), strict=True)))
        lo, hi = self._ranges()[0]
        return f"""
<mujoco model="tacrig">
  <compiler angle="radian"/>
  <option timestep="{self.TIMESTEP}" integrator="implicitfast" tolerance="1e-6"/>
  <visual><global offwidth="640" offheight="480"/>
    <headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="4 4" texuniform="true"/>
    <material name="frame" rgba="0.15 0.15 0.17 1"/>
    <material name="rail" rgba="0.62 0.62 0.66 1"/>
    <material name="plate" rgba="0.25 0.25 0.28 1"/>
    <material name="holder" rgba="0.88 0.88 0.82 1"/>
    <material name="block" rgba="0.95 0.55 0.2 1"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="0" conaffinity="0"/>
    <camera name="rigcam" mode="targetbody" target="holder" pos="-0.20 -0.15 {P + 0.13:.4f}"/>
    <body name="frame">
      <geom type="box" pos="0 0 0.01" size="0.16 {span:.4f} 0.01" material="frame" contype="0" conaffinity="0"/>
      <geom type="box" pos="0 0 {P / 2 + 0.005:.4f}" size="0.10 0.08 {P / 2 - 0.0105:.4f}" material="frame" contype="0" conaffinity="0"/>
      <geom name="plate" type="box" pos="0 0 {P - 0.006:.4f}" size="0.10 0.08 0.006" material="plate"/>
      <geom type="cylinder" fromto="{lo - 0.03:.4f} {-span:.4f} 0.03 {hi + 0.03:.4f} {-span:.4f} 0.03" size="0.006" material="rail" contype="0" conaffinity="0"/>
      <geom type="cylinder" fromto="{lo - 0.03:.4f} {span:.4f} 0.03 {hi + 0.03:.4f} {span:.4f} 0.03" size="0.006" material="rail" contype="0" conaffinity="0"/>
    </body>
    <body name="holder" pos="0 0 {P:.4f}">
      {holder}
    </body>
    <body name="gantry">
      <joint name="rig_x" type="slide" axis="1 0 0" limited="true" range="{f(self._ranges()[0])}" damping="5" armature="0.05"/>
      <geom type="box" pos="0.05 {-span:.4f} {gz / 2:.4f}" size="0.015 0.015 {gz / 2:.4f}" material="frame" contype="0" conaffinity="0" mass="0.5"/>
      <geom type="box" pos="0.05 {span:.4f} {gz / 2:.4f}" size="0.015 0.015 {gz / 2:.4f}" material="frame" contype="0" conaffinity="0" mass="0.5"/>
      <geom type="box" pos="0.05 0 {gz:.4f}" size="0.02 {span + 0.015:.4f} 0.02" material="frame" contype="0" conaffinity="0" mass="0.5"/>
      <body name="carriage" pos="0 0 {gz:.4f}">
        <joint name="rig_y" type="slide" axis="0 1 0" limited="true" range="{f(self._ranges()[1])}" damping="5" armature="0.05"/>
        <geom type="box" pos="0.01 0 0" size="0.025 0.035 0.045" material="rail" contype="0" conaffinity="0" mass="0.4"/>
        <body name="spindle" pos="0 0 {z_spindle - gz:.4f}" gravcomp="1">
          <joint name="rig_z" type="slide" axis="0 0 1" limited="true" range="{f(self._ranges()[2])}" damping="5" armature="0.05"/>
          <geom type="cylinder" pos="0 0 0.03" size="0.022 0.03" material="frame" contype="0" conaffinity="0" mass="0.3"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>{servos}</actuator>
</mujoco>"""

    def _ranges(self):
        return ((-self.X_TRAVEL, self.X_TRAVEL), (-self.Y_TRAVEL, self.Y_TRAVEL), tuple(self.Z_RANGE))

    def _build_spec(self, sensor_xml=None):
        spec = mujoco.MjSpec.from_string(self._scene_xml())
        # the block: a heightfield in the holder's pocket (zero level on the pocket floor)
        xs, ys = tx.node_grid(self.TEXTURE_RES_MM, self.BLOCK_MM[:2])
        bx, by, bz = (v / 1000 for v in self.BLOCK_MM)
        spec.add_hfield(name="block", size=[bx / 2, by / 2, bz, 0.0005], nrow=len(ys), ncol=len(xs),
                        userdata=[0.5] * (len(xs) * len(ys)))
        spec.body("holder").add_geom(
            name="block", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="block", material="block",
            pos=[0, 0, POCKET_FLOOR_MM / 1000], condim=3, mass=0,
            friction=[self.BLOCK_FRICTION, 0.005, 0.0001])

        # the sensor, under the spindle, pointing down
        sensor = resolve_sensor_model(sensor_xml, source=self.SENSOR_SOURCE, root=self.SENSOR_ROOT,
                                      standalone=self.SENSOR_XML)
        self.sensor_info = sensor_info(sensor)
        spindle = spec.body("spindle")
        tip = [0.0, 0.0, -self.SPINDLE_DROP]
        root = attach_sensor(spec, spindle, sensor, prefix=self.SENSOR_PREFIX, tip_at=tip,
                             axis_to=[0, 0, -1], stiffness_scale=self.sensor_stiffness)
        self._sensor_root = root.name
        spindle.add_site(name="tip_site", pos=tip, size=[0.002, 0, 0], rgba=[1, 0, 0, 0.4])
        rod = self.SPINDLE_DROP - self.sensor_info["reach"]
        if rod > 0.005:
            spindle.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, fromto=[0, 0, 0, 0, 0, -rod],
                             size=[0.008, 0, 0], material="rail", contype=0, conaffinity=0, mass=0.05)
        return spec

    def _index_model(self):
        m = self.model
        jids = [m.joint(n).id for n in AXES]
        self._qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self._dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self._act = np.array([m.actuator(n).id for n in AXES])
        self._lo = np.array([r[0] for r in self._ranges()])
        self._hi = np.array([r[1] for r in self._ranges()])
        self._tip_site = m.site("tip_site").id
        self._ee_site = self._tip_site                    # (base-class marker helpers)
        self._sensor_root_id = m.body(self._sensor_root).id
        self._holder = m.body("holder").id
        self._hf = m.hfield("block").id
        self._holder_pos0 = m.body_pos[self._holder].copy()

        def under(b, root):
            while b != 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self._sensor_geom = np.array([bool(collides[g]) and under(int(m.geom_bodyid[g]), self._sensor_root_id)
                                      for g in range(m.ngeom)])
        self._holder_geom = m.geom_bodyid == self._holder
        xs, ys = tx.node_grid(self.TEXTURE_RES_MM, self.BLOCK_MM[:2])
        self._xs_m = (xs - self.BLOCK_MM[0] / 2) / 1000             # nodes, holder frame
        self._ys_m = (ys - self.BLOCK_MM[1] / 2) / 1000

    def _hfield_ids(self):
        return (self._hf,)

    def _n_actions(self):
        return 3

    # ================================================================ block, tilt, axes
    def _set_block(self, k, rng):
        """Put block k (index into self.textures) in the holder."""
        H = self._ideal[k]
        dx = self._xs_m[1] - self._xs_m[0]
        dy = self._ys_m[1] - self._ys_m[0]
        if self.PRINT_BLUR_MM > 0:
            s = rng.uniform(0, self.PRINT_BLUR_MM) / 1000
            H = tx.gaussian_blur(H, s / dx, s / dy)
        if self.PRINT_NOISE_MM > 0:
            H = H + rng.normal(0.0, self.PRINT_NOISE_MM, H.shape)
        H = np.clip(H, 0.0, self.BLOCK_MM[2])
        tx.write_hfield(self.model, self._hf, H / 1000, self.BLOCK_MM[2] / 1000)
        self._hfields_changed()
        floor = POCKET_FLOOR_MM / 1000
        self._surf = tx.Bilinear(floor + H / 1000, self._xs_m[0], self._xs_m[-1],
                                 self._ys_m[0], self._ys_m[-1])
        self._top_local = float(floor + H.max() / 1000)
        self.texture_index = int(k)
        self.texture = self.textures[k]

    def _set_tilt(self, angle):
        """Tilt the plate (holder and block) by `angle` rad about TILT_AXIS."""
        self.tilt = float(angle)
        u = np.asarray(self.TILT_AXIS, float)
        u = u / np.linalg.norm(u)
        q = np.array([np.cos(self.tilt / 2), *(np.sin(self.tilt / 2) * u)])
        self.model.body_quat[self._holder] = q
        self.model.body_pos[self._holder] = self._holder_pos0

    def _holder_frame(self):
        d = self.data
        return d.xpos[self._holder].copy(), d.xmat[self._holder].reshape(3, 3).copy()

    def _tip_local(self):
        """Sensor tip in the holder's frame."""
        p, R = self._holder_frame()
        return R.T @ (self.data.site_xpos[self._tip_site] - p)

    def _gap(self):
        """Tip height above the surface under it, along the holder's z (m)."""
        t = self._tip_local()
        return float(t[2] - self._surf(t[0], t[1]))

    def _start_z(self):
        """Spindle position that puts the tip HOVER above the (tilted) block's highest point."""
        bx, by = self.BLOCK_MM[0] / 2000, self.BLOCK_MM[1] / 2000
        corners = np.array([[x, y, self._top_local] for x in (-bx, bx) for y in (-by, by)])
        q = self.model.body_quat[self._holder]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, q)
        rise = float((corners @ R.reshape(3, 3).T)[:, 2].max() - self._top_local)
        return float(np.clip(max(0.0, rise), *self.Z_RANGE))

    def _put_axes(self, q, settle=50):
        """Teleport the axes to q (and command them there), then let the sensor settle."""
        m, d = self.model, self.data
        q = np.clip(q, self._lo, self._hi)
        d.qpos[self._qadr] = q
        d.qvel[self._dadr] = 0.0
        d.ctrl[self._act] = q
        self._cmd = q.copy()
        mujoco.mj_forward(m, d)
        if settle:
            mujoco.mj_step(m, d, settle)

    def _axes(self):
        return self.data.qpos[self._qadr].copy()

    def _plate_force(self):
        """Normal force (N) between the sensor and the holder / block: the plate's reading."""
        d, m = self.data, self.model
        n = d.ncon
        if n == 0:
            return 0.0
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        hit = ((self._sensor_geom[g1] & self._holder_geom[g2])
               | (self._sensor_geom[g2] & self._holder_geom[g1]))
        f = np.zeros(6)
        total = 0.0
        for i in np.flatnonzero(hit):
            mujoco.mj_contactForce(m, d, int(i), f)
            total += abs(f[0])
        return float(total)

    @property
    def plate_force(self):
        return self._force

    # ================================================================ gym API
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)
        if options.get("texture") is not None:
            t = options["texture"]
            if t not in self.textures:
                raise ValueError(f"texture {t!r} is not one of this env's textures {self.textures}")
            k = self.textures.index(t)
        else:
            k = int(rng.integers(len(self.textures)))
        self._set_block(k, rng)
        self._set_tilt(float(options.get("tilt", rng.uniform(*self.TILT_RANGE))))
        xy = options.get("start")
        xy = (rng.uniform(-self.START_XY_MM, self.START_XY_MM, 2) if xy is None
              else np.asarray(xy, float))
        self._put_axes(np.array([xy[0] / 1000, xy[1] / 1000, self._start_z()]))
        self.step_count = 0
        self._vel_cmd = np.zeros(3)
        self._force = self._plate_force()
        self._fail_reason = ""
        self._reset_task(options)
        self._reset_sensors()
        return self._get_obs(), self._info()

    def _move(self, v):
        """Filtered velocity command -> axis targets (leashed), stepped over the frame."""
        k = self.ACTION_SMOOTHING
        self._vel_cmd = k * self._vel_cmd + (1 - k) * np.clip(v, -1.0, 1.0)
        q = self._axes()
        cmd = np.clip(self._cmd + self._vel_cmd * np.asarray(self.AXIS_SPEED) * self.dt, self._lo, self._hi)
        leash = np.asarray(self.LEASH, float)
        cmd = np.clip(cmd, q - np.maximum(leash, np.abs(self._cmd - q)),
                      q + np.maximum(leash, np.abs(self._cmd - q)))
        old = self._cmd
        chunks = max(1, min(self.CTRL_CHUNKS, self.FRAME_SKIP))
        per = self.FRAME_SKIP // chunks
        done = 0
        for c in range(chunks):
            n = per if c < chunks - 1 else self.FRAME_SKIP - done
            self.data.ctrl[self._act] = old + (c + 1) / chunks * (cmd - old)
            self._mj_step(n)
            done += n
        self._cmd = cmd

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        self._move(a[:3])
        self.step_count += 1
        self._step_sensors()
        self._force = self._plate_force()
        reward, terminated, status = self._task_step(a)
        truncated = (not terminated) and self.step_count >= self.max_steps
        if truncated:
            reward += self._on_timeout()
        if not np.all(np.isfinite(self.data.qpos)):
            terminated, truncated, status = True, False, "fail"
            self._fail_reason = "simulation unstable"
            mujoco.mj_resetData(self.model, self.data)
        return self._get_obs(), float(reward), terminated, truncated, self._info(status)

    # ---------------------------------------------------------------- task hooks
    def _reset_task(self, options):
        pass

    def _task_step(self, a):
        """(reward, terminated, status 'success' / 'fail' / None) after a step."""
        return 0.0, False, None

    def _on_timeout(self):
        return 0.0

    def _info(self, status=None):
        return {"is_success": status == "success",
                "fail_reason": self._fail_reason if status == "fail" else "",
                "texture": self.texture if isinstance(self.texture, (int, np.integer)) else str(self.texture),
                "texture_index": self.texture_index, "plate_force": self._force, "tilt": self.tilt}

    # ---------------------------------------------------------------- observation
    def _obs_parts(self):
        q = self._axes()
        mid, half = (self._hi + self._lo) / 2, (self._hi - self._lo) / 2
        parts = {"proprio": np.concatenate([(q - mid) / half, (self._cmd - q) / np.asarray(self.LEASH),
                                            self._vel_cmd])}
        if self.obs_modes.get("force"):
            parts["force"] = np.array([self._force / self.F_REF])
        if self.obs_modes.get("privileged"):
            onehot = np.zeros(len(self.textures))
            onehot[self.texture_index] = 1.0
            t = self._tip_local()
            parts["privileged"] = np.concatenate([onehot, t[:2] * 100, [self._gap() * 100, self.tilt]])
        parts.update(self._sensor_obs())
        return parts

    def _get_obs(self):
        return self._format_obs(self._obs_parts())

    # ---------------------------------------------------------------- helpers
    def press_to(self, force, max_travel=0.02, settle_steps=10, slow=0.25):
        """Lower the sensor until the plate reads `force` newtons (or it has gone max_travel
        down), slowing to `slow` x speed once touching, then hold still for settle_steps.
        Returns the last observation."""
        a = np.zeros(self.action_space.shape)
        if self._decide_index is not None:
            a[self._decide_index] = -1.0                      # (TacRig-texture: don't answer)
        z0 = self._cmd[2]
        obs = self._get_obs()
        limit = int(max_travel / (self.AXIS_SPEED[2] * self.dt * slow)) + 50
        for _ in range(limit):
            if self._force >= force or z0 - self._cmd[2] >= max_travel:
                break
            a[2] = -1.0 if self._force < 0.05 else -slow
            obs, *_ = self.step(a)
        a[2] = 0.0
        for _ in range(settle_steps):
            obs, *_ = self.step(a)
        return obs

    def render(self):
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._view_cam = mujoco.MjvCamera()
            mujoco.mjv_defaultFreeCamera(self.model, self._view_cam)
            self._view_cam.lookat[:] = [0.0, 0.0, self.PLATE_TOP + 0.02]
            self._view_cam.distance, self._view_cam.azimuth, self._view_cam.elevation = 0.45, 135, -25
        self.sync_renderer(self._view_renderer)
        self._view_renderer.update_scene(self.data, camera=self._view_cam)
        return self._view_renderer.render().copy()


class TacRigTexture(TacRig):
    """Identify the block by touch. Actions: sensor velocity x, y, z; then "decide" (> 0
    answers, once DECIDE_AFTER steps have passed, and ends the episode); then one score per
    texture in env.textures (the answer is the highest).

    Reward per step: -STEP_COST, +W_TOUCH while the plate reads between F_TOUCH and F_MAX,
    -W_FORCE per newton above F_MAX. Answering: R_CORRECT or R_WRONG; no answer by the end
    of the episode counts as wrong. info["guess"]: the index answered."""
    STEP_COST = 0.01
    W_TOUCH   = 0.02
    W_FORCE   = 0.05
    F_TOUCH, F_MAX = 0.3, 5.0     # N
    R_CORRECT, R_WRONG = 10.0, -5.0
    DECIDE_AFTER = 10
    MAX_STEPS = 200

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._decide_index = 3

    def _n_actions(self):
        return 3 + 1 + len(self.textures)

    def _reset_task(self, options):
        self.guess = None

    def _task_step(self, a):
        F = self._force
        r = (-self.STEP_COST + self.W_TOUCH * float(self.F_TOUCH < F < self.F_MAX)
             - self.W_FORCE * max(0.0, F - self.F_MAX))
        if a[3] > 0 and self.step_count >= self.DECIDE_AFTER:
            self.guess = int(np.argmax(a[4:]))
            if self.guess == self.texture_index:
                return r + self.R_CORRECT, True, "success"
            self._fail_reason = "wrong texture"
            return r + self.R_WRONG, True, "fail"
        return r, False, None

    def _on_timeout(self):
        return self.R_WRONG

    def _info(self, status=None):
        info = super()._info(status)
        info["guess"] = self.guess
        return info


# ==================================================================== the paper's protocol
def press_grid(env, textures=None, points=5, span_mm=(5.0, 5.0), centre_mm=(0.0, 0.0),
               repeats=2, force=1.9, readout="tactile", tilt=0.0, max_travel=0.02,
               settle_steps=10, verbose=True):
    """The dataset paper's data collection on the simulated rig: for every texture, press the
    sensor onto a points x points grid (span_mm wide, centred centre_mm from the block
    centre) until the plate reads `force` newtons (the paper: 1.9 N, a 5 x 5 grid over
    5 x 5 mm, two presses per point), and record the `readout` input there.

    Switches the env to obs_format="dict" with only `readout` selected. Returns a dict:
    X (samples, ...), y (index into textures), textures, pos (mm), force (N, as recorded)."""
    e = env.unwrapped
    e.set_obs_format("dict")
    e.set_obs_modes([readout])
    textures = tuple(e.textures if textures is None else textures)
    xs = centre_mm[0] + np.linspace(-span_mm[0] / 2, span_mm[0] / 2, points)
    ys = centre_mm[1] + np.linspace(-span_mm[1] / 2, span_mm[1] / 2, points)
    X, y, pos, F = [], [], [], []
    for i, t in enumerate(textures):
        for _ in range(repeats):
            for px in xs:
                for py in ys:
                    e.reset(options={"texture": t, "start": (px, py), "tilt": tilt})
                    obs = e.press_to(force, max_travel=max_travel, settle_steps=settle_steps)
                    X.append(np.asarray(obs[readout]).copy())
                    y.append(i)
                    pos.append((px, py))
                    F.append(e.plate_force)
        if verbose:
            print(f"[press_grid] texture {t}: {repeats * points * points} presses, "
                  f"force {np.mean(F[-repeats * points * points:]):.2f} N on average")
    return dict(X=np.stack(X), y=np.array(y), textures=textures, pos=np.array(pos), force=np.array(F))
