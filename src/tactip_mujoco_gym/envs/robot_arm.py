"""
Tactip arm tasks.

    TacArm-hill   Incline         intercept a ball rolling down a ramp, then push it to the top
    TacArm-peg    Peg             start touching a cylinder, push it into the hole
    TacArm-shape  ShapeInsertion  start touching a shape, identify it, push it into its own hole
    TacArm-edge   Edge            start touching the top edge of a block, follow it round (CCW)

Every task accepts:
    obs_modes={"vision": bool, "tactile": bool, "privileged": bool, ...} or a list of names
              (see base.py; default: privileged only; missing keys count as False). Inputs:
              privileged, tactile, tactile_image, vision, events, plus any added by `sensors`
    obs_format="flat" | "dict"     one long vector (default) or one key per input
    sensors={key: spec}            change / add / remove readouts (see base.py, sensors/)
    sensor_xml=None                sensor hardware on the arm: None = the TacTip written into
                                   the scene; a registered name ("tactip"), a standalone MJCF
                                   file or an MjSpec swaps it (see sensors/models.py)
    sensor_stiffness=1.0           scales the sensor's joint / tendon stiffness and damping
    overrides={"TIP_R": 0.03}      set class constants on this instance
    events=True/False              append tx_* taxel events (= "events" in obs_modes)
    privileged=True/False          earlier shortcut: privileged only / tactile only

Swapping the sensor: the built-in TacTip (every body under SENSOR_PARENT, with its tendons,
equalities and contact excludes) is cut out of the scene and the chosen sensor is attached to
SENSOR_LINK, turned and moved so its tip is where the TacTip's tip was. The arm's ee_site
(the point the controller moves) and hand camera stay where they are. Constants that depend
on the sensor's shape (TIP_R, PUSH_Z, Incline.TIP_H) may need `overrides`; check_envs.py
reports them for a new sensor.

Control (shared by all tasks)
-----------------------------
action in [-1, 1]  ->  low-pass filtered velocity command (ACTION_SMOOTHING)
    -> Cartesian tip displacement (ACTION_SCALE metres per step at full command)
    -> commanded tip target, which may not run more than LEASH ahead of the measured tip
       (no wind-up when blocked) but is never dragged back to it (no gravity-sag ratchet)
    -> differential IK (damped least squares) warm-started from the previous *command*,
       never from the measured joints, so the arm can't drift or flip IK branches
    -> joint targets with velocity + gravity feed-forward, interpolated over FRAME_SKIP steps.

Rewards
-------
Per step, roughly in [-1, 1]: mostly *progress* made this step (object toward its hole, ball up
the slope, distance along the edge) plus small closeness terms. All terms depend only on the
state (no hidden stage counters). Paying for progress rather than proximity matters: with a
static closeness reward, doing nothing out-scored noisy attempts and PPO learned to sit still.
Success adds R_SUCCESS and ends the episode; unrecoverable failure adds R_FAIL and ends it.
"""
from importlib.resources import files

from .base import OBS_FORMATS, TactileGymEnv, TaxelMixin  # noqa: F401
import gymnasium as gym
import numpy as np
import mujoco

from tactip_mujoco_gym.sensors import Mount

_BUILTIN_TIP = {}             # scene file -> (tip, axis) of its TacTip, in SENSOR_LINK's frame


def _tol(d, sigma):
    """Smooth closeness reward: 1 at d = 0, 0.24 at d = sigma, ~0 beyond 3 sigma."""
    return float(1.0 - np.tanh(abs(d) / sigma))


# =============================================================================
# Base arm environment
# =============================================================================
class RobotArmEnv(TaxelMixin):
    ARM_JOINTS = [f"joint{i}" for i in range(1, 7)]
    ACTION_BOUND = 1.0        # actions in [-1, 1]; scaled to metres below

    # ---- control ----
    FRAME_SKIP   = 20         # physics steps per env step (timestep 1 ms -> 50 Hz policy)
    CTRL_CHUNKS  = 4          # joint targets are interpolated in this many chunks per env step
    ACTION_SCALE = 0.005      # metres per env step at |action| = 1  (0.25 m/s)
    LEASH        = 0.01       # max |commanded tip - measured tip| (m)
    MAX_STEPS    = 250        # 5 s episodes
    PLANAR       = False      # True: actions are (dx, dy) and the tip height is held at PUSH_Z
    PUSH_Z       = 0.10
    ACTION_SMOOTHING = 0.6    # velocity command = a*prev + (1-a)*action: exploration noise
                              # becomes gentle motion instead of jerks that knock objects over
    VEL_FF       = True       # servo velocity feed-forward
    GRAVITY_COMP = True       # servo gravity feed-forward

    # ---- IK ----
    TARGET_QUAT    = np.array([1.0, 0.0, 0.0, 0.0])   # sensor orientation (same as before)
    ROT_WEIGHT     = 0.3      # rad -> m trade-off of the orientation error
    IK_ITERS       = 4        # per env step (warm-started: 1-2 normally suffice)
    IK_RESET_ITERS = 300
    IK_DAMPING     = 5e-4
    IK_MAX_DQ      = 0.2      # rad per IK iteration
    HOME_TIP       = np.array([0.35, 0.0, 0.20])
    SETTLE_STEPS   = 100      # physics steps after teleporting the arm at reset

    # reachable box for commanded tip positions; override per task
    WS_LO = np.array([0.10, -0.40, 0.03])
    WS_HI = np.array([0.65,  0.40, 0.40])

    # ---- reward ----
    R_SUCCESS = 100.0
    R_FAIL    = -10.0

    # ---- hand camera ----
    CAMERA_NAME = "handcam"          # get_camera() default (viewing / recording)
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    VISION_CAMERA = CAMERA_NAME      # camera used for the "vision" observation (size: see base.py)

    # ---- sensor hardware ----
    SENSOR_PARENT = "sensor_root"    # body in the scene XML that holds the built-in TacTip
    SENSOR_LINK   = "link06"         # arm link a swapped-in sensor is attached to
    SENSOR_PREFIX = "sensor_"        # element-name prefix of a swapped-in sensor
    SENSOR_SOURCE = ("assets", "tactip_arm.xml")   # scene the built-in TacTip's tip is read from

    def __init__(self, xml_subpath=("assets", "tactip_arm.xml"), action_dim=3,
                 obs_modes=None, privileged=None, obs_format="flat", events=False,
                 render_mode=None, max_steps=None, sensors=None, sensor_xml=None,
                 sensor_stiffness=1.0, overrides=None):
        self._apply_overrides(overrides)
        self._quat_cmd = None     # sensor orientation the IK holds (None: TARGET_QUAT)
        model = self._build_model(xml_subpath, sensor_xml, sensor_stiffness)
        super().__init__(list(xml_subpath), 1, action_dim, model=model)   # obs space: below
        self.render_mode = render_mode
        # get_camera() keeps its own renderer: self._renderer belongs to the base class
        # (in-sensor camera), and sharing it mixed up image sizes between the two
        self._cam_renderer = None
        self._camera_width = self.CAMERA_WIDTH
        self._camera_height = self.CAMERA_HEIGHT
        m = self.model
        self.max_steps = int(max_steps or self.MAX_STEPS)
        self._action_dim = action_dim

        # arm joints / actuators
        jids = [m.joint(n).id for n in self.ARM_JOINTS]
        self._arm_qadr = np.array([m.jnt_qposadr[j] for j in jids])
        self._arm_dadr = np.array([m.jnt_dofadr[j] for j in jids])
        self._q_lo = m.jnt_range[jids, 0].copy()
        self._q_hi = m.jnt_range[jids, 1].copy()
        unlimited = ~m.jnt_limited[jids].astype(bool)
        self._q_lo[unlimited], self._q_hi[unlimited] = -np.pi, np.pi
        self._arm_act = []
        for j in jids:
            hits = np.flatnonzero((m.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT)
                                  & (m.actuator_trnid[:, 0] == j))
            if hits.size == 0:
                raise RuntimeError(f"No actuator drives joint {m.joint(j).name}")
            self._arm_act.append(int(hits[0]))
        self._arm_act = np.array(self._arm_act)
        # affine servo gains: force = gain*ctrl - kp*q - kd*qdot  (gain = 0 -> plain targets)
        self._servo_gain = m.actuator_gainprm[self._arm_act, 0].copy()
        self._servo_kp = -m.actuator_biasprm[self._arm_act, 1].copy()
        self._servo_kd = -m.actuator_biasprm[self._arm_act, 2].copy()
        affine = m.actuator_biastype[self._arm_act] == mujoco.mjtBias.mjBIAS_AFFINE
        self._servo_gain[~affine] = 0.0

        # geoms belonging to the arm (anything in the same kinematic tree as the sensor)
        self._arm_root = int(m.body_rootid[m.site_bodyid[self._ee_site]])
        self._arm_geom = m.body_rootid[m.geom_bodyid] == self._arm_root

        # IK scratch data (never touches the simulation state)
        self._ikd = mujoco.MjData(m)
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))
        self._qbuf = np.zeros((4, 4))           # quaternion scratch
        self._wbuf = np.zeros(3)

        # home configuration, solved once from the model's default pose
        q0 = np.clip(m.qpos0[self._arm_qadr].copy(), self._q_lo, self._q_hi)
        self._q_home = q0
        self._q_home = self._ik(self.HOME_TIP, q0, iters=600)
        self._q_cmd = self._q_home.copy()
        self._target = self.HOME_TIP.copy()
        self._cmd_vel = np.zeros(action_dim)
        self._potentials = {}
        self._ik_err = 0.0

        self._setup_task()

        # readouts, bound to the mounted sensor
        self.sensor_mounts = [self._arm_mount()]
        self._setup_sensors(sensors)

        # marker rest shape: arm at home, sensor in free space
        mujoco.mj_resetData(m, self.data)
        self._place_arm(self.HOME_TIP, settle=300)
        self._calibrate_sensors()

        modes = self.resolve_obs_modes(obs_modes, privileged)
        if events:
            modes["events"] = True
        if obs_format not in OBS_FORMATS:
            raise ValueError(f"obs_format must be one of {OBS_FORMATS}, not {obs_format!r}")
        self.obs_format = obs_format
        self.set_obs_modes(modes)           # validates the selection, builds the space

    # ------------------------------------------------------------------ sensor hardware
    def _build_model(self, xml_subpath, sensor_xml, sensor_stiffness):
        """The scene as written (sensor_xml None, stiffness 1), or with its TacTip swapped
        for `sensor_xml` / rescaled (see the module docstring)."""
        path = str(files("tactip_mujoco_gym").joinpath(*xml_subpath))
        self.sensor_xml = sensor_xml
        self.sensor_stiffness = float(sensor_stiffness)
        self.sensor_info = None
        self._sensor_root = None
        edits = type(self)._edit_spec is not RobotArmEnv._edit_spec
        if sensor_xml is None and self.sensor_stiffness == 1.0 and not edits:
            return mujoco.MjModel.from_xml_path(path)
        from tactip_mujoco_gym.sensor_tools import (attach_sensor, child_bodies, compile_spec,
                                                    scene_without_bodies, sensor_info)
        if sensor_xml is None and self.sensor_stiffness == 1.0:
            spec = mujoco.MjSpec.from_file(path)
            self._edit_spec(spec)
            return compile_spec(spec)
        from tactip_mujoco_gym.sensors import resolve_sensor_model
        xml, _ = scene_without_bodies(path, child_bodies(path, self.SENSOR_PARENT))
        spec = mujoco.MjSpec.from_string(xml)
        tip, axis = self._builtin_tip()
        sensor = resolve_sensor_model(sensor_xml)
        self.sensor_info = sensor_info(sensor)
        root = attach_sensor(spec, spec.body(self.SENSOR_LINK), sensor, prefix=self.SENSOR_PREFIX,
                             tip_at=tip, axis_to=axis, stiffness_scale=self.sensor_stiffness)
        self._sensor_root = root.name
        self._edit_spec(spec)
        return compile_spec(spec)

    def _edit_spec(self, spec):
        """Task hook: add to the scene (an MjSpec of the scene XML, with the sensor already
        swapped if sensor_xml was given) before it is compiled."""

    def _builtin_tip(self):
        """Tip point and direction of the scene's own TacTip, in SENSOR_LINK's frame,
        measured in the scene as written (as extract_sensor would see it, without the
        extraction and its MuJoCo attach notices)."""
        src = str(files("tactip_mujoco_gym").joinpath(*self.SENSOR_SOURCE))
        if src not in _BUILTIN_TIP:
            from tactip_mujoco_gym.sensor_tools import model_sensor_info
            m = mujoco.MjModel.from_xml_path(src)
            d = mujoco.MjData(m)
            mujoco.mj_forward(m, d)
            info = model_sensor_info(m, d, m.body(self.SENSOR_LINK).id, self.MARKER_PREFIX,
                                     within=True, skip_root_geoms=True)
            _BUILTIN_TIP[src] = (info["tip"].copy(), info["axis"].copy())
        tip, axis = _BUILTIN_TIP[src]
        return tip.copy(), axis.copy()

    def _arm_mount(self):
        m = self.model
        if self._sensor_root is not None:
            body, prefix = m.body(self._sensor_root).id, self.SENSOR_PREFIX
        else:
            body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, self.SENSOR_PARENT)
            body = int(m.site_bodyid[self._ee_site]) if body < 0 else body
            prefix = ""
        return Mount("tip", prefix, int(body), ("site", int(self._ee_site)))

    # ------------------------------------------------------------------ task hooks
    def _setup_task(self):
        """Look up ids, set per-task state. Called once from __init__."""

    # ------------------------------------------------------------------ hand camera
    def get_camera(
        self,
        mode="rgb",
        camera=None,
        width=None,
        height=None,
    ):
        """
        Render an image from a MuJoCo camera attached to the robot.

        Args:
            mode: "rgb" or "depth"
            camera: Camera name (defaults to CAMERA_NAME)
            width: Image width (defaults to CAMERA_WIDTH)
            height: Image height (defaults to CAMERA_HEIGHT)

        Returns:
            rgb:   uint8 array, shape (H, W, 3)
            depth: float32 array, shape (H, W), depth in metres
        """

        camera = camera or self.CAMERA_NAME
        width = width or self.CAMERA_WIDTH
        height = height or self.CAMERA_HEIGHT

        if mode not in ("rgb", "depth"):
            raise ValueError(
                f"Unknown camera mode '{mode}'. Use 'rgb' or 'depth'."
            )

        # Check camera exists
        try:
            self.model.camera(camera)
        except KeyError:
            raise ValueError(
                f"Camera '{camera}' not found in MuJoCo model."
            ) from None

        # Create renderer on first use
        if (
            self._cam_renderer is None
            or self._camera_width != width
            or self._camera_height != height
        ):
            if self._cam_renderer is not None:
                self._cam_renderer.close()

            self._cam_renderer = mujoco.Renderer(
                self.model,
                height=height,
                width=width,
            )

            self._camera_width = width
            self._camera_height = height

        # Switch between RGB and depth rendering
        if mode == "depth":
            self._cam_renderer.enable_depth_rendering()
        else:
            self._cam_renderer.disable_depth_rendering()

        # Render from the selected camera
        self._cam_renderer.update_scene(
            self.data,
            camera=camera,
        )

        image = self._cam_renderer.render()

        return image.copy()

    def close(self):
        """Release the camera renderer, then the base class renderers."""
        if self._cam_renderer is not None:
            self._cam_renderer.close()
            self._cam_renderer = None
        super().close()

    def _reset_task(self, options):
        """Place objects and the arm. Must leave self._q_cmd / self._target consistent."""
        self._place_arm(self.HOME_TIP)

    def _privileged_obs(self):
        return np.zeros(0, dtype=np.float32)

    def _reward(self):
        return 0.0

    def _status(self):
        """'success', 'fail' or None."""
        return None

    def _task_info(self):
        return {}

    def _post_physics(self):
        """Called after every env step's physics (e.g. to re-park hidden objects)."""

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)          # seeds self.np_random (skip the legacy reset)
        mujoco.mj_resetData(self.model, self.data)
        self.step_count = 0
        self._cmd_vel = np.zeros(self._action_dim)
        self._potentials = {}
        self._fail_reason = ""
        self._quat_cmd = None
        self._reset_task(options or {})
        self._reset_sensors()
        return self._get_obs(), self._info()

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel()[:self._action_dim], -1.0, 1.0)
        k = self.ACTION_SMOOTHING
        self._cmd_vel = k * self._cmd_vel + (1.0 - k) * a
        self._apply_action(self._cmd_vel)
        self._drive_arm(self.FRAME_SKIP)
        self._post_physics()
        self._step_sensors()
        self.step_count += 1

        status = self._status()
        reward = self._reward()
        if status == "success":
            reward += self.R_SUCCESS
        elif status == "fail":
            reward += self.R_FAIL
        terminated = status is not None
        truncated = (not terminated) and self.step_count >= self.max_steps
        info = self._info(status)
        return self._get_obs(), float(reward), terminated, truncated, info

    def _info(self, status=None):
        info = {"is_success": status == "success", "ik_err": self._ik_err,
                "fail_reason": self._fail_reason if status == "fail" else ""}
        info.update(self._task_info())
        return info

    def _fail(self, reason):
        """Use in _status: `return self._fail("tipped")`; the reason appears in info["fail_reason"]."""
        self._fail_reason = reason
        return "fail"

    def _obs_parts(self):
        """The selected inputs, in a fixed order: proprioception (always), the task's
        privileged state, then the sensor readouts (tactile, tactile_image, vision, events,
        then any added ones)."""
        tip = self._tip_pos()
        parts = {"proprio": np.concatenate([
            tip,
            (self._target - tip) / self.LEASH,      # commanded vs measured: a crude force sense
            self._cmd_vel,                          # filtered velocity command (part of the state)
        ])}
        if self.obs_modes["privileged"]:
            parts["privileged"] = self._privileged_obs()
        parts.update(self._sensor_obs())
        return parts

    def _get_obs(self):
        return self._format_obs(self._obs_parts())

    # ------------------------------------------------------------------ control
    def _apply_action(self, a):
        """Default: Cartesian deltas in world xyz (or xy with fixed height when PLANAR)."""
        delta = np.zeros(3)
        if self.PLANAR:
            delta[:2] = a[:2] * self.ACTION_SCALE
            self._target[2] = self.PUSH_Z
        else:
            delta[:] = a[:3] * self.ACTION_SCALE
        self._set_target(self._target + delta)

    def _set_target(self, t):
        self._target = self._leash(np.clip(np.asarray(t, dtype=float), self.WS_LO, self.WS_HI),
                                   self._target, self._tip_pos())

    def _leash(self, new, old, tip):
        """Anti wind-up: the command may not run further than LEASH ahead of the measured tip
        (e.g. when pushing against something). It is never pulled back toward the tip, so a
        sagging or lagging arm can't ratchet the command away (the old resync bug)."""
        off = new - tip
        n = np.linalg.norm(off)
        lim = max(self.LEASH, np.linalg.norm(old - tip))
        if n > lim:
            new = tip + off * (lim / n)
        return new

    def _servo_ctrl(self, q_ref, qd_ref):
        """Joint position targets for the arm's affine (PD) servos, with velocity and gravity
        feed-forward. Without it the Z1 servos (kp=1000, kd=100) lag 0.1 s behind any motion
        and sag under gravity."""
        u = q_ref.copy()
        ok = self._servo_gain > 0
        if np.any(ok):
            tau = self._servo_kd * qd_ref
            if self.GRAVITY_COMP:
                tau = tau + self.data.qfrc_bias[self._arm_dadr]
            u[ok] = (self._servo_kp[ok] * q_ref[ok] + tau[ok]) / self._servo_gain[ok]
        return u

    def _drive_arm(self, n_steps):
        """Solve IK for the current target and run n_steps physics steps toward it."""
        q_prev = self._q_cmd.copy()
        self._q_cmd = self._ik(self._target, self._q_cmd, self.IK_ITERS, quat=self._quat_cmd)
        m, d = self.model, self.data
        qd_ref = (self._q_cmd - q_prev) / (n_steps * m.opt.timestep) if self.VEL_FF else 0 * q_prev
        chunks = max(1, min(self.CTRL_CHUNKS, n_steps))
        per = n_steps // chunks
        done = 0
        for c in range(chunks):
            k = per if c < chunks - 1 else n_steps - done
            q_ref = q_prev + (c + 1) / chunks * (self._q_cmd - q_prev)
            d.ctrl[self._arm_act] = self._servo_ctrl(q_ref, qd_ref)
            self._mj_step(k)                # (per physics step while taxel events are on)
            done += k

    def _ik(self, target, q_init, iters, quat=None):
        """Damped-least-squares IK on a scratch MjData. Returns arm joint angles."""
        m, d = self.model, self._ikd
        quat = self.TARGET_QUAT if quat is None else quat
        q = np.array(q_init, dtype=float)
        q_cur, q_neg, q_err = self._qbuf[0], self._qbuf[1], self._qbuf[2]
        w = self.ROT_WEIGHT
        err_p = np.zeros(3)
        for _ in range(iters):
            d.qpos[self._arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            err_p = target - d.site_xpos[self._ee_site]
            mujoco.mju_mat2Quat(q_cur, d.site_xmat[self._ee_site])
            mujoco.mju_negQuat(q_neg, q_cur)
            mujoco.mju_mulQuat(q_err, quat, q_neg)
            if q_err[0] < 0:
                q_err *= -1
            mujoco.mju_quat2Vel(self._wbuf, q_err, 1.0)
            err = np.concatenate([err_p, w * self._wbuf])
            if np.linalg.norm(err_p) < 2e-5 and w * np.linalg.norm(self._wbuf) < 2e-4:
                break
            mujoco.mj_jacSite(m, d, self._jacp, self._jacr, self._ee_site)
            J = np.vstack([self._jacp[:, self._arm_dadr], w * self._jacr[:, self._arm_dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + self.IK_DAMPING * np.eye(6), err)
            s = np.max(np.abs(dq))
            if s > self.IK_MAX_DQ:
                dq *= self.IK_MAX_DQ / s
            q = np.clip(q + dq, self._q_lo, self._q_hi)
        self._ik_err = float(np.linalg.norm(err_p))
        return q

    def _place_arm(self, tip_target, settle=None, q_seed=None):
        """Reset helper: teleport the arm to an IK solution for tip_target and let it settle."""
        q = self._ik(np.asarray(tip_target, float),
                     self._q_home if q_seed is None else q_seed, self.IK_RESET_ITERS)
        m, d = self.model, self.data
        d.qpos[self._arm_qadr] = q
        d.qvel[self._arm_dadr] = 0.0
        self._q_cmd = q.copy()
        self._target = np.clip(np.asarray(tip_target, float), self.WS_LO, self.WS_HI)
        mujoco.mj_forward(m, d)
        d.ctrl[self._arm_act] = self._servo_ctrl(q, np.zeros_like(q))
        n = self.SETTLE_STEPS if settle is None else settle
        if n:
            mujoco.mj_step(m, d, n)

    def _approach(self, direction, body_id, step=0.0008, max_steps=60, press=0.0, retries=3):
        """Reset helper: creep along `direction` until the sensor touches `body_id`, stop the
        command at the tip (+ `press` metres), let things settle, and retry with finer steps if
        the contact didn't survive (a light object can be nudged away). Returns True on contact."""
        u = np.asarray(direction, float)
        u = u / (np.linalg.norm(u) + 1e-12)
        for _ in range(retries + 1):
            for _ in range(max_steps):
                if self._touching(body_id):
                    break
                self._target = self._target + u * step
                self._drive_arm(10)
            self._target = self._tip_pos() + u * press   # from the tip, not the (leading) command
            self._drive_arm(40)
            if self._touching(body_id):
                return True
            step *= 0.5
        return False

    # ------------------------------------------------------------------ helpers
    def _tip_pos(self):
        return self.data.site_xpos[self._ee_site].copy()

    def _delta(self, key, value):
        """Per-step decrease of `value` (e.g. distance to goal) in units of the max tip step,
        clipped to [-1, 1]. Pays for moving toward the goal rather than for being near it,
        so idling earns little (a static 'closeness' reward made doing nothing a local optimum)."""
        prev = self._potentials.get(key)
        self._potentials[key] = value
        if prev is None:
            return 0.0
        step = float(np.atleast_1d(self.ACTION_SCALE)[0])
        return float(np.clip((prev - value) / step, -1.0, 1.0))

    def _contacts_with(self, body_id):
        """Indices of contacts between the arm/sensor and body_id."""
        d, m = self.data, self.model
        n = d.ncon
        if n == 0:
            return np.zeros(0, dtype=int)
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        b1, b2 = m.geom_bodyid[g1], m.geom_bodyid[g2]
        hit = ((b1 == body_id) & self._arm_geom[g2]) | ((b2 == body_id) & self._arm_geom[g1])
        return np.flatnonzero(hit)

    def _touching(self, body_id):
        return self._contacts_with(body_id).size > 0

    def _contact_force(self, body_id):
        """Total normal force (N) between the sensor/arm and body_id."""
        f = np.zeros(6)
        total = 0.0
        for i in self._contacts_with(body_id):
            mujoco.mj_contactForce(self.model, self.data, int(i), f)
            total += abs(f[0])
        return total

    def _set_free_body(self, qadr, dadr, pos, quat=(1, 0, 0, 0)):
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = quat
        self.data.qvel[dadr:dadr + 6] = 0.0


# =============================================================================
# Shared pushing logic (Peg and ShapeInsertion)
# =============================================================================
class _PushTask(RobotArmEnv):
    PLANAR = True
    TIP_R = 0.012             # effective radius of the sensor tip (stand-in: 12 mm)
    W_REACH, W_GOAL, W_PROG = 0.1, 0.2, 0.7
    SIG_REACH = 0.02          # m
    SIG_GOAL = 0.08           # m

    def _push_point(self, obj_xy, goal_xy, obj_r):
        d = goal_xy - obj_xy
        d = d / (np.linalg.norm(d) + 1e-9)
        return obj_xy - d * (obj_r + self.TIP_R)

    def _start_touching(self, obj_body, obj_xy, goal_xy, obj_r, gap=0.006):
        """Teleport the tip just behind the object (w.r.t. the goal) and creep into contact."""
        d = goal_xy - obj_xy
        d = d / (np.linalg.norm(d) + 1e-9)
        start = np.array([*(obj_xy - d * (obj_r + self.TIP_R + gap)), self.PUSH_Z])
        self._place_arm(start)
        self._approach([d[0], d[1], 0.0], obj_body)

    def _push_reward(self, tip, obj_xy, goal_xy, obj_r):
        """reach: tip behind the object w.r.t. the goal; goal: object near the goal;
        prog: object moved toward the goal this step (dominant term)."""
        push_pt = self._push_point(obj_xy, goal_xy, obj_r)
        dist = float(np.linalg.norm(obj_xy - goal_xy))
        reach = _tol(np.linalg.norm(tip[:2] - push_pt), self.SIG_REACH)
        goal = _tol(dist, self.SIG_GOAL)
        prog = self._delta("goal", dist)
        return self.W_REACH * reach + self.W_GOAL * goal + self.W_PROG * prog


# =============================================================================
# Peg: push a cylinder into a hole
# =============================================================================
class Peg(_PushTask):
    HOLE_XY      = np.array([0.45, 0.0])
    HOLE_HALF    = 0.03
    PLATFORM_TOP = 0.05
    CYL_R        = 0.02
    CYL_HALF_H   = 0.03
    PUSH_Z       = 0.05 + 0.015          # push low on the cylinder so it slides instead of tipping
    TILT_LIMIT   = 0.87                  # cos(~30 deg)

    WS_LO = np.array([0.20, -0.20, 0.055])
    WS_HI = np.array([0.55,  0.20, 0.25])

    def __init__(self, start_touching=True, **kwargs):
        self.start_touching = start_touching
        super().__init__(xml_subpath=("assets", "peg_worldbody_tx.xml"), action_dim=2, **kwargs)

    def _setup_task(self):
        m = self.model
        self._cyl_body = m.body("peg_cyl").id
        self._cyl_qadr = m.joint("peg_free").qposadr[0]
        self._cyl_dadr = m.joint("peg_free").dofadr[0]

    # ---------- helpers ----------
    def _cyl_pos(self):
        return self.data.xpos[self._cyl_body].copy()

    def _upright(self):
        return self.data.xmat[self._cyl_body][8]

    def _in_hole(self, cyl):
        return bool(np.all(np.abs(cyl[:2] - self.HOLE_XY) < self.HOLE_HALF)
                    and cyl[2] < self.PLATFORM_TOP + 0.5 * self.CYL_HALF_H)

    def _status(self):
        cyl = self._cyl_pos()
        if self._in_hole(cyl):
            return "success"
        if self._upright() < 0.3 and cyl[2] > self.PLATFORM_TOP:
            return self._fail("lying on platform")          # unrecoverable
        over_hole = np.all(np.abs(cyl[:2] - self.HOLE_XY) < self.HOLE_HALF + self.CYL_R)
        if over_hole:
            return None                                     # tilting while it drops in is fine
        if self._upright() < self.TILT_LIMIT:
            return self._fail("tipped")
        if cyl[2] < self.PLATFORM_TOP:
            return self._fail("below platform top")        # pushed off the platform
        return None

    # ---------- episode ----------
    def _reset_task(self, options):
        rng = self.np_random
        x, y = rng.uniform(0.33, 0.37), rng.uniform(-0.02, 0.02)
        z = self.PLATFORM_TOP + self.CYL_HALF_H + 1e-3
        self._set_free_body(self._cyl_qadr, self._cyl_dadr, [x, y, z])
        mujoco.mj_forward(self.model, self.data)
        if self.start_touching:
            self._start_touching(self._cyl_body, np.array([x, y]), self.HOLE_XY, self.CYL_R)
        else:
            self._place_arm([x - 0.06 + rng.uniform(-0.02, 0.0), y + rng.uniform(-0.04, 0.04), self.PUSH_Z])

    def _privileged_obs(self):
        tip, cyl = self._tip_pos(), self._cyl_pos()
        R = self.data.xmat[self._cyl_body]
        vel = self.data.qvel[self._cyl_dadr:self._cyl_dadr + 2]
        return np.concatenate([
            cyl,                                  # object marker
            self.HOLE_XY - cyl[:2],
            tip[:2] - cyl[:2],
            [R[2], R[5]],                         # tilt of the cylinder axis
            vel,
            [float(self._touching(self._cyl_body))],
        ]).astype(np.float32)

    def _reward(self):
        tip, cyl = self._tip_pos(), self._cyl_pos()
        return self._push_reward(tip, cyl[:2], self.HOLE_XY, self.CYL_R)

    def _task_info(self):
        cyl = self._cyl_pos()
        return {"goal_dist": float(np.linalg.norm(cyl[:2] - self.HOLE_XY)),
                "touching": self._touching(self._cyl_body)}


# =============================================================================
# Shape insertion: identify the shape by touch, push it into its own hole
# =============================================================================
PLATE_TOP    = 0.05
PLATE_HALF_T = 0.01            # plate spans z = 0.03 .. 0.05
OBJ_HALF_H   = 0.03
CLEARANCE    = 0.004           # hole is this much bigger than the object, per side
PLATE_X      = (0.28, 0.62)
HOLE_X       = 0.50
HOLE_Y       = [-0.15, -0.05, 0.05, 0.15]   # one hole per shape, in SHAPES order
CELL_HALF    = 0.05            # half-size of the square patch around each hole
STRIP        = 0.0025          # resolution used to cut non-box holes

# sym = rotational symmetry order (0 = circle, any yaw fits)
SHAPES = {
    "circle":    dict(kind="cylinder", r=0.02,           sym=0, bound_r=0.020, rgba="0.9 0.2 0.2 1"),
    "square":    dict(kind="box", hx=0.02, hy=0.02,      sym=4, bound_r=0.029, rgba="0.2 0.7 0.3 1"),
    "rectangle": dict(kind="box", hx=0.03, hy=0.015,     sym=2, bound_r=0.034, rgba="0.2 0.4 0.9 1"),
    "triangle":  dict(kind="prism", R=0.025,             sym=3, bound_r=0.025, rgba="0.9 0.7 0.1 1"),
}
PLATE_ATTR = 'rgba="0.6 0.6 0.6 1" friction="0.8 0.005 0.0001"'


class ShapeInsertion(_PushTask):
    SHAPE_NAMES = list(SHAPES.keys())
    N_SHAPES    = len(SHAPE_NAMES)

    SPAWN_X  = (0.34, 0.38)
    SPAWN_Y  = (-0.04, 0.04)
    PUSH_Z   = PLATE_TOP + 0.015
    TILT_LIMIT = 0.87
    HOLE_TOL   = 0.03            # xy distance for "went into the correct hole"
    INSERT_DEPTH = 0.008         # inserted once the shape's lowest point is this far below the plate
    YAW_FADE   = 0.10            # yaw starts to matter within 10 cm of the hole
    SIG_YAW    = 0.15            # rad
    SIG_GOAL   = 0.12
    W_REACH, W_GOAL, W_PROG, W_YAW = 0.1, 0.15, 0.6, 0.15
    MAX_STEPS  = 300

    WS_LO = np.array([0.15, -0.25, 0.055])
    WS_HI = np.array([0.62,  0.25, 0.25])

    def __init__(self, spawn_yaw=0.1, spawn_toward_hole=0.0, start_touching=True, **kwargs):
        # Curriculum knobs:
        #   spawn_yaw          0.1 rad fits every hole without re-orienting; np.pi means the policy
        #                      has to rotate shapes by pushing (much harder).
        #   spawn_toward_hole  0 = spawn in the middle (the policy must route to the right hole),
        #                      1 = spawn in line with the shape's own hole (straight push).
        self.spawn_yaw = float(spawn_yaw)
        self.spawn_toward_hole = float(spawn_toward_hole)
        self.start_touching = start_touching
        super().__init__(xml_subpath=("assets", "tactip_arm_shapes.xml"), action_dim=2, **kwargs)

    def _setup_task(self):
        m = self.model
        self._body, self._geom, self._qadr, self._dadr = {}, {}, {}, {}
        self._rgba0, self._contype0, self._conaff0 = {}, {}, {}
        for n in self.SHAPE_NAMES:
            self._body[n] = m.body(f"shape_{n}").id
            g = m.geom(f"shape_{n}_geom").id
            self._geom[n] = g
            jid = m.joint(f"shape_{n}_free").id
            self._qadr[n] = m.jnt_qposadr[jid]
            self._dadr[n] = m.jnt_dofadr[jid]
            self._rgba0[n] = m.geom_rgba[g].copy()
            self._contype0[n] = int(m.geom_contype[g])
            self._conaff0[n] = int(m.geom_conaffinity[g])
        self.shape = self.SHAPE_NAMES[0]
        self._target_xy = np.array([HOLE_X, HOLE_Y[0]])

    # ---------- helpers ----------
    def _park_pos(self, i):
        return [1.5, -0.6 + 0.4 * i, 0.5]

    def _obj_pos(self):
        return self.data.xpos[self._body[self.shape]].copy()

    def _upright(self):
        return self.data.xmat[self._body[self.shape]][8]

    def _yaw(self):
        R = self.data.xmat[self._body[self.shape]]
        return float(np.arctan2(R[3], R[0]))

    def _yaw_error(self):
        """Signed yaw error to the hole (hole yaw = 0), wrapped by the shape's symmetry."""
        k = SHAPES[self.shape]["sym"]
        if k == 0:
            return 0.0
        period = 2 * np.pi / k
        return (self._yaw() + period / 2) % period - period / 2

    def _show_only(self, name):
        """Hide every shape except `name`: transparent and no collisions."""
        m = self.model
        for n in self.SHAPE_NAMES:
            g, b = self._geom[n], self._body[n]
            on = (n == name)
            m.geom_rgba[g] = self._rgba0[n] if on else np.append(self._rgba0[n][:3], 0.0)
            m.geom_contype[g] = self._contype0[n] if on else 0
            m.geom_conaffinity[g] = self._conaff0[n] if on else 0
            m.body_gravcomp[b] = 0.0 if on else 1.0      # only effective if the XML uses gravcomp

    def _post_physics(self):
        # hidden shapes don't collide; keep them parked so they don't free-fall forever
        for i, n in enumerate(self.SHAPE_NAMES):
            if n != self.shape:
                self._set_free_body(self._qadr[n], self._dadr[n], self._park_pos(i))

    def _lowest_z(self):
        """Lowest point of the shape (from its local bounding box), world z."""
        g = self._geom[self.shape]
        R = self.data.geom_xmat[g].reshape(3, 3)
        aabb = self.model.geom_aabb[g]                      # [cx, cy, cz, hx, hy, hz] local
        return float(self.data.geom_xpos[g][2] + R[2] @ aabb[:3] - np.abs(R[2]) @ aabb[3:])

    def _status(self):
        obj = self._obj_pos()
        dist = np.linalg.norm(obj[:2] - self._target_xy)
        if self._lowest_z() < PLATE_TOP - self.INSERT_DEPTH:
            # part of the shape is below the plate surface -> it is in a hole (or off the plate);
            # a shape that tips and wedges in its own hole still counts as inserted
            return "success" if dist < self.HOLE_TOL else self._fail("wrong hole / off plate")
        if self._upright() < self.TILT_LIMIT and dist > 0.05:
            return self._fail("tipped")                    # away from its hole
        if self._upright() < 0.3 and obj[2] > PLATE_TOP:
            return self._fail("lying on plate")            # unrecoverable
        return None

    # ---------- episode ----------
    def _reset_task(self, options):
        rng = self.np_random
        name = options.get("shape") or str(rng.choice(self.SHAPE_NAMES))
        self.shape = name
        idx = self.SHAPE_NAMES.index(name)
        self._target_xy = np.array([HOLE_X, HOLE_Y[idx]])
        self._show_only(name)
        for i, n in enumerate(self.SHAPE_NAMES):
            if n != name:
                self._set_free_body(self._qadr[n], self._dadr[n], self._park_pos(i))

        x, y = rng.uniform(*self.SPAWN_X), rng.uniform(*self.SPAWN_Y)
        y += self.spawn_toward_hole * self._target_xy[1]
        yaw = rng.uniform(-self.spawn_yaw, self.spawn_yaw)
        self._set_free_body(self._qadr[name], self._dadr[name],
                            [x, y, PLATE_TOP + OBJ_HALF_H + 1e-3],
                            [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        mujoco.mj_forward(self.model, self.data)
        r = SHAPES[name]["bound_r"]
        if self.start_touching:
            self._start_touching(self._body[name], np.array([x, y]), self._target_xy, r)
        else:
            self._place_arm([x - r - 0.04, y, self.PUSH_Z])

    def _privileged_obs(self):
        tip, obj = self._tip_pos(), self._obj_pos()
        k = max(SHAPES[self.shape]["sym"], 1)
        yaw = self._yaw()
        onehot = np.zeros(self.N_SHAPES)
        onehot[self.SHAPE_NAMES.index(self.shape)] = 1.0
        vel = self.data.qvel[self._dadr[self.shape]:self._dadr[self.shape] + 2]
        return np.concatenate([
            obj,
            [np.sin(k * yaw), np.cos(k * yaw), self._yaw_error()],
            self._target_xy - obj[:2],
            tip[:2] - obj[:2],
            vel,
            [float(self._touching(self._body[self.shape]))],
            onehot,
        ]).astype(np.float32)

    def _reward(self):
        tip, obj = self._tip_pos(), self._obj_pos()
        r = self._push_reward(tip, obj[:2], self._target_xy, SHAPES[self.shape]["bound_r"])
        goal_d = np.linalg.norm(obj[:2] - self._target_xy)
        near = np.clip(1.0 - goal_d / self.YAW_FADE, 0.0, 1.0)      # yaw only matters near the hole
        return r + self.W_YAW * near * _tol(self._yaw_error(), self.SIG_YAW)

    def _task_info(self):
        obj = self._obj_pos()
        return {"shape": self.shape,
                "goal_dist": float(np.linalg.norm(obj[:2] - self._target_xy)),
                "yaw_err": float(self._yaw_error())}


# =============================================================================
# Incline: stop a ball rolling down a ramp, then push it up to the top
# =============================================================================
class Incline(RobotArmEnv):
    BALL_R = 0.02
    TIP_R  = 0.012
    TIP_H  = 0.02            # sensor height above the surface along the normal (= ball centre)

    F_SLIDE = (0.05, 1.0)    # sliding friction, log-interpolated
    F_ROLL  = (0.0, 0.008)   # rolling friction, linear (needs condim=6)

    S_MIN, Y_LIM = -0.06, 0.06
    SUCCESS_MARGIN = 0.04    # success when the ball centre is this close to the top edge
    W_REACH, W_HEIGHT, W_PROG = 0.15, 0.25, 0.6
    SIG_REACH = 0.03
    MAX_STEPS = 300

    WS_LO = np.array([0.15, -0.15, 0.0])
    WS_HI = np.array([0.65,  0.15, 0.30])

    def __init__(self, observe_friction=False, **kwargs):
        self.observe_friction = observe_friction
        super().__init__(xml_subpath=("assets", "hill_worldbody_tx.xml"), action_dim=2, **kwargs)

    def _setup_task(self):
        m, d = self.model, self.data
        self._ball_body = m.body("ball").id
        jid = m.joint("ball_free").id
        self._ball_qadr = m.jnt_qposadr[jid]
        self._ball_dadr = m.jnt_dofadr[jid]
        self._ramp_geom = m.geom("ramp_surface").id
        mujoco.mj_forward(m, d)
        ramp = m.body("ramp").id
        R = d.xmat[ramp].reshape(3, 3)
        self._S = d.xpos[ramp].copy()          # bottom edge of the surface
        self._D = R[:, 0].copy()               # up-slope direction
        self._N = R[:, 2].copy()               # surface normal
        self._L = 2.0 * m.geom_size[self._ramp_geom][0]
        self.S_MAX = self._L - 0.03
        self.friction_level = 0.5
        self._cmd_s = self._cmd_y = 0.0

    # ---------- helpers ----------
    def _ball_pos(self):
        return self.data.xpos[self._ball_body].copy()

    def _slope_coords(self, p):
        """(s, h, lat): distance up the slope, height above the surface, sideways offset."""
        rel = p - self._S
        return float(rel @ self._D), float(rel @ self._N), float(rel[1])

    def _tip_target(self, s, y):
        t = self._S + s * self._D + self.TIP_H * self._N
        t[1] += y
        return t

    def _set_friction(self, f):
        self.friction_level = float(f)
        slide = self.F_SLIDE[0] * (self.F_SLIDE[1] / self.F_SLIDE[0]) ** f
        roll = self.F_ROLL[0] + f * (self.F_ROLL[1] - self.F_ROLL[0])
        self.model.geom_friction[self._ramp_geom] = [slide, 0.005, roll]
        g = 0.8 - 0.5 * f
        self.model.geom_rgba[self._ramp_geom] = [g, g, g, 1.0]

    def _push_dist(self, tip, ball):
        """Distance (in the slope plane) from the tip to the point just below the ball."""
        s_b, _, lat_b = self._slope_coords(ball)
        s_t, _, lat_t = self._slope_coords(tip)
        s_goal = s_b - (self.BALL_R + self.TIP_R)
        return float(np.hypot(s_t - s_goal, lat_t - lat_b))

    def _status(self):
        s, h, lat = self._slope_coords(self._ball_pos())
        if s >= self._L - self.SUCCESS_MARGIN:
            return "success"
        if s < -0.04:
            return self._fail("ball rolled past the bottom")
        if abs(lat) > 0.10:
            return self._fail("ball off the side")
        if h > 0.08 or h < -0.02:
            return self._fail("ball left the slope surface")
        return None

    # ---------- control: actions are (along slope, sideways) ----------
    def _apply_action(self, a):
        s_t, _, y_t = self._slope_coords(self._tip_pos())
        s = float(np.clip(self._cmd_s + a[0] * self.ACTION_SCALE, self.S_MIN, self.S_MAX))
        y = float(np.clip(self._cmd_y + a[1] * self.ACTION_SCALE, -self.Y_LIM, self.Y_LIM))
        s, y = self._leash(np.array([s, y]), np.array([self._cmd_s, self._cmd_y]), np.array([s_t, y_t]))
        self._cmd_s, self._cmd_y = float(s), float(y)
        self._target = self._tip_target(self._cmd_s, self._cmd_y)

    # ---------- episode ----------
    def _reset_task(self, options):
        rng = self.np_random
        f = options["friction"] if "friction" in options else rng.uniform(0.0, 1.0)
        self._set_friction(f)
        a, d = self._ball_qadr, self._ball_dadr

        # 1. park the ball out of the way while the arm is placed
        self._set_free_body(a, d, [1.0, 1.0, 0.5])
        mujoco.mj_forward(self.model, self.data)

        # 2. put the tip somewhere on the lower half of the ramp
        s_tip = rng.uniform(0.05, 0.5) * self._L
        y0 = rng.uniform(-0.02, 0.02)
        self._place_arm(self._tip_target(s_tip, y0))
        s_t, _, lat_t = self._slope_coords(self._tip_pos())
        self._cmd_s, self._cmd_y = s_t, lat_t

        # 3. drop the ball uphill of the tip so it rolls toward it
        s_ball = max(rng.uniform(0.5, 0.85) * self._L, s_t + 0.06)
        centre = self._S + s_ball * self._D + (self.BALL_R + 1e-3) * self._N
        centre[1] += rng.uniform(-0.02, 0.02)
        self._set_free_body(a, d, centre)
        mujoco.mj_forward(self.model, self.data)

    def _privileged_obs(self):
        tip, ball = self._tip_pos(), self._ball_pos()
        s_b, h_b, lat_b = self._slope_coords(ball)
        s_t, h_t, lat_t = self._slope_coords(tip)
        v = self.data.qvel[self._ball_dadr:self._ball_dadr + 3]
        w = self.data.qvel[self._ball_dadr + 3:self._ball_dadr + 6]
        obs = [[s_t / self._L, lat_t, h_t],
               [s_b / self._L, lat_b, h_b],                   # object marker (slope frame)
               [s_t - s_b, lat_t - lat_b],
               [v @ self._D, v[1], v @ self._N],             # ball velocity (slope frame)
               w * self.BALL_R,
               [float(self._touching(self._ball_body))]]
        if self.observe_friction:
            obs.append([self.friction_level])
        return np.concatenate(obs).astype(np.float32)

    def _reward(self):
        tip, ball = self._tip_pos(), self._ball_pos()
        s, _, _ = self._slope_coords(ball)
        reach = _tol(self._push_dist(tip, ball), self.SIG_REACH)     # tip just below the ball
        height = float(np.clip(s / self._L, 0.0, 1.0))               # ball high up the slope
        prog = self._delta("down", -s)        # + pushing it up, - letting it roll down
        return self.W_REACH * reach + self.W_HEIGHT * height + self.W_PROG * prog

    def _task_info(self):
        return {"friction": self.friction_level,
                "ball_s": self._slope_coords(self._ball_pos())[0] / self._L,
                "touching": self._touching(self._ball_body)}


# =============================================================================
# Edge: follow the top edge of a randomly rotated block, counter-clockwise
# =============================================================================
class Edge(RobotArmEnv):
    ACTION_SCALE = np.array([0.004, 0.004, 0.0015])   # slower in z: depth control
    MAX_STEPS = 400              # a lap (0.6 m) takes ~150 steps at full speed
    LAP_FRACTION = 1.0           # success after this many laps
    CORNER_CLEAR = 0.015         # don't start within this distance of a corner
    LOST_LAT = 0.025             # |sideways error| beyond this = lost the edge
    LOST_UP, LOST_DOWN = 0.02, 0.015
    SIG_LAT, SIG_DEPTH = 0.005, 0.003
    W_EDGE, W_TOUCH, W_DEPTH, W_PROG = 0.10, 0.05, 0.05, 0.80

    WS_LO = np.array([0.12, 0.12, 0.08])
    WS_HI = np.array([0.48, 0.48, 0.20])

    def __init__(self, **kwargs):
        super().__init__(xml_subpath=("assets", "edge_worldbody_tx.xml"), action_dim=3, **kwargs)

    def _setup_task(self):
        m = self.model
        self._block_body = m.body("block").id
        self._block_geom = m.geom("block_geom").id
        self._block_quat0 = m.body_quat[self._block_body].copy()
        self._travelled = 0.0
        self._u_prev = 0.0
        self._z_ref = 0.012
        self._compute_perimeter()
        self._progress = (0.0, 0.0, self._corners[0], np.array([1.0, 0.0]))

    # ---------- geometry ----------
    def _compute_perimeter(self):
        mujoco.mj_forward(self.model, self.data)
        g = self._block_geom
        pos = self.data.geom_xpos[g]
        R = self.data.geom_xmat[g].reshape(3, 3)
        hx, hy, hz = self.model.geom_size[g]
        local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])   # CCW
        self._corners = local @ R[:2, :2].T + pos[:2]
        self._seg = np.roll(self._corners, -1, axis=0) - self._corners
        self._seg_len = np.linalg.norm(self._seg, axis=1)
        self._cum = np.concatenate([[0.0], np.cumsum(self._seg_len)[:-1]])
        self._P = float(self._seg_len.sum())
        self._top_z = float(pos[2] + hz)
        self._block_c, self._block_R, self._half = pos[:2].copy(), R[:2, :2].copy(), (hx, hy)

    def _edge_coords(self, p_xy):
        """Arc length u along the CCW perimeter, signed distance (+ outside),
        closest edge point, and the CCW tangent there."""
        rel = p_xy - self._corners
        t = np.clip(np.einsum("ij,ij->i", rel, self._seg) / self._seg_len ** 2, 0.0, 1.0)
        q = self._corners + t[:, None] * self._seg
        dist = np.linalg.norm(p_xy - q, axis=1)
        i = int(np.argmin(dist))
        local = self._block_R.T @ (p_xy - self._block_c)
        inside = abs(local[0]) < self._half[0] and abs(local[1]) < self._half[1]
        lat = -dist[i] if inside else dist[i]
        return self._cum[i] + t[i] * self._seg_len[i], lat, q[i], self._seg[i] / self._seg_len[i]

    def _randomize_block(self):
        angle = self.np_random.uniform(-np.pi, np.pi)
        self.quat = np.array([np.cos(angle / 2), 0, 0, np.sin(angle / 2)])
        self.model.body_quat[self._block_body] = self.quat
        self._compute_perimeter()

    def _status(self):
        tip = self._tip_pos()
        _, lat, _, _ = self._edge_coords(tip[:2])
        dz = tip[2] - self._top_z - self._z_ref
        if self._travelled >= self.LAP_FRACTION * self._P:
            return "success"
        if abs(lat) > self.LOST_LAT:
            return self._fail("lost the edge")
        if dz > self.LOST_UP:
            return self._fail("lifted off")
        if dz < -self.LOST_DOWN:
            return self._fail("pressed too deep")
        return None

    # ---------- episode ----------
    def _reset_task(self, options):
        self._randomize_block()
        rng = self.np_random
        # random start point on the perimeter, away from the corners
        for _ in range(1000):
            u0 = rng.uniform(0, self._P)
            i = int(np.searchsorted(self._cum, u0, side="right") - 1)
            along = u0 - self._cum[i]
            if self.CORNER_CLEAR < along < self._seg_len[i] - self.CORNER_CLEAR:
                break
        else:                                   # block too small for CORNER_CLEAR: use a mid-point
            i = int(np.argmax(self._seg_len))
            along = 0.5 * self._seg_len[i]
        p = self._corners[i] + self._seg[i] / self._seg_len[i] * along
        self._place_arm([p[0], p[1], self._top_z + 0.025])
        self._approach([0, 0, -1], self._block_body, step=0.001, max_steps=40, press=0.0015)
        tip = self._tip_pos()
        self._z_ref = float(tip[2] - self._top_z)
        u, lat, q, tan = self._edge_coords(tip[:2])
        self._u_prev, self._travelled = u, 0.0
        self._progress = (0.0, lat, q, tan)

    def _step_progress(self):
        u, lat, q, tan = self._edge_coords(self._tip_pos()[:2])
        du = u - self._u_prev
        if du > self._P / 2:
            du -= self._P
        elif du < -self._P / 2:
            du += self._P
        self._u_prev = u
        self._travelled += du
        return du, lat, q, tan

    def _post_physics(self):
        # integrate progress exactly once per step, before _status / _reward read it
        self._progress = self._step_progress()

    def _privileged_obs(self):
        tip = self._tip_pos()
        u, lat, q, tan = self._edge_coords(tip[:2])
        yaw = 2 * np.arctan2(self.model.body_quat[self._block_body][3],
                             self.model.body_quat[self._block_body][0])
        seg = int(np.searchsorted(self._cum, u, side="right") - 1)
        to_corner = self._cum[seg] + self._seg_len[seg] - u
        return np.concatenate([
            self._block_c - tip[:2],               # object marker (block centre)
            [np.sin(yaw), np.cos(yaw)],
            q - tip[:2],                           # nearest edge point
            tan,                                   # direction to follow
            [lat, tip[2] - self._top_z - self._z_ref, to_corner,
             self._travelled / self._P, float(self._touching(self._block_body))],
        ]).astype(np.float32)

    def _reward(self):
        tip = self._tip_pos()
        du, lat, _, _ = self._progress
        dz = tip[2] - self._top_z - self._z_ref
        touch = float(self._touching(self._block_body))
        on_edge = _tol(lat, self.SIG_LAT)
        prog = float(np.clip(du / self.ACTION_SCALE[0], -1.0, 1.0))
        return (self.W_EDGE * on_edge + self.W_TOUCH * touch + self.W_DEPTH * _tol(dz, self.SIG_DEPTH)
                + self.W_PROG * on_edge * touch * prog)

    def _task_info(self):
        return {"laps": self._travelled / self._P}
