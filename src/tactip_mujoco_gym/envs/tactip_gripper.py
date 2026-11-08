"""
TacGripper: a parallel-jaw gripper whose two fingers are TacTips, on a Cartesian gantry over
a table. The task is the humanoid's (TacHumanoid-pick): squeeze-grasp a soft or rigid object,
carry it to a target on the table, put it down. Objects, randomisation, curriculum, reward,
success and failure are all shared with it (see tactip_humanoid.py); only the robot differs.

    env = gym.make("TacGripper-pick", obs_modes=["tactile", "object"])

Robot: the gripper hangs from a gantry that moves it in x, y, z and turns it about the
vertical (stiff position servos, gravity compensated). Its two fingers slide toward each
other, each a position servo with a force limit (FINGER_FORCE) so a closed command squeezes.
The TacTips face each other with their tips on the grasp line; the frame around them is
sized from the mounted sensor, so a different sensor_xml gets a gripper that fits it.

Actions (5, in [-1, 1], low-pass filtered): grasp centre x, y, z (velocity, world frame),
grasp width (velocity: negative closes), grasp yaw (turn rate, up to +-GRASP_YAW_MAX).
The width command may sit at most SQUEEZE_MAX inside the measured gap between the tips,
which bounds the grip force.

Observation inputs (obs_modes); proprioception is always included:
    proprio     both tips (world), how far each lags its command, the filtered action, the
                target, grasp yaw, whether the object has been lifted
    object, privileged, tactile, tactile_image as on TacHumanoid-pick; vision: "overhead"
reset(options=...): "object", "stiffness", "mass", "friction", "yaw", "level".
ScriptedPicker (tactip_humanoid.py) drives this gripper too.
"""
import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from tactip_mujoco_gym.sensor_tools import _collision_points, attach_sensor, compile_spec, sensor_info

from .tactip_humanoid import OBJECT_NAMES, SIDES, ScriptedPicker, TacHumanoidPick  # noqa: F401

GRIP_JOINTS = ("grip_x", "grip_y", "grip_z", "grip_yaw", "finger_left", "finger_right")
TABLE_TOP = TacHumanoidPick.TABLE_TOP


def _subtree(b):
    out = [b]
    for c in b.bodies:
        out.extend(_subtree(c))
    return out


def sensor_extent(sensor, info):
    """(depth, lateral) of a standalone sensor: how far its colliding geometry reaches back
    from the tip along the axis, and out from the axis (m)."""
    m = compile_spec(sensor.copy())
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rid = m.body(info["root"]).id
    origin, R = d.xpos[rid], d.xmat[rid].reshape(3, 3)
    rel = (_collision_points(m, d) - origin) @ R - info["tip"]
    along = rel @ info["axis"]
    lateral = rel - np.outer(along, info["axis"])
    return float(max(0.0, -along.min())), float(np.linalg.norm(lateral, axis=1).max())


class TacGripperPick(TacHumanoidPick):
    HAS_BASE      = False
    ROBOT_ROOT    = "gripper"
    VISION_CAMERA = "overhead"
    TRACK_BODY, TRACK_DISTANCE, TRACK_ELEVATION = "gripper", 0.9, -30
    EPISODE_SECONDS = 15.0

    # ---------------------------------------------------------------- gripper
    HOME         = np.array([0.70, 0.0, TABLE_TOP + 0.15])   # grasp centre at qpos 0
    OPEN_WIDTH   = 0.25           # m between the tips, fingers fully open
    MIN_WIDTH    = 0.02           # m: fully closed
    PALM_Z       = 0.10           # m from the grasp line up to the palm (raised if the
                                  # sensor is taller)
    FRAME_GAP    = 0.008          # m between the back of a sensor and its finger
    KP_XYZ, KV_XYZ = 2e4, 400.0   # gantry servos, x and y (N/m, N s/m)
    KP_Z, KV_Z   = 2500.0, 120.0  # z: compliant, so a sensor pressed onto the table (a 15 mm
                                  # command lead: ~40 N) can't pin the fingers by friction
    KP_YAW, KV_YAW = 200.0, 10.0
    FINGER_KP, FINGER_KV = 2500.0, 50.0
    FINGER_FORCE = 60.0           # N a finger can squeeze with
    GRASP_CLEARANCE = 0.002       # m: the lowest grasp keeps the sensors this far off the table
    PICKER_LOWER_TO = 0.0         # ScriptedPicker puts objects down to the grasp height
    YAW_LIMIT    = 1.0            # rad the gripper turns either way (> GRASP_YAW_MAX)

    # ---------------------------------------------------------------- task
    WS_LO = np.array([0.45, -0.35, TABLE_TOP + 0.01])
    WS_HI = np.array([0.95, 0.35, TABLE_TOP + 0.25])
    WIDTH_RANGE  = (0.02, 0.25)
    START_CENTRE = HOME.copy()
    START_WIDTH  = 0.15
    OBJECT_X     = (0.60, 0.80)
    OBJECT_Y     = (-0.10, 0.10)
    TARGET_DIST  = ((0.10, 0.18), (0.12, 0.40))   # m from the object (level 0, level 1)
    TARGET_MARGIN = 0.06          # m inside the workspace and the table's edge

    def __init__(self, objects=None, level=None, curriculum=True, sensor_stiffness=10.0,
                 obs_modes=None, privileged=None, obs_format="flat", render_mode=None,
                 max_steps=None, sensor_xml=None, sensors=None, overrides=None):
        """
        objects           names from OBJECT_NAMES to draw from (default: all ten)
        level             0..1 fixes the difficulty; None starts at 0 with a curriculum
        sensor_xml        the sensor used as each finger (default: the TacTip)
        sensor_stiffness  scales the sensors' skin (see sensor_tools.attach_sensor)
        sensors, obs_modes, obs_format, overrides, max_steps: as on every robot
        """
        gym.Env.__init__(self)
        self._apply_overrides(overrides)
        self._init_sensor_state()
        self.object_names = tuple(objects) if objects else OBJECT_NAMES
        bad = [o for o in self.object_names if o not in OBJECT_NAMES]
        if bad:
            raise ValueError(f"unknown objects {bad}; choose from {OBJECT_NAMES}")
        self.action_mode = "gripper"
        self.sensor_stiffness = float(sensor_stiffness)
        self.curriculum = curriculum and level is None
        self.level = 0.0 if level is None else float(np.clip(level, 0, 1))
        self.render_mode = render_mode
        self.xml_path = None
        self.sensor_xml = sensor_xml

        spec = self._build_spec(sensor_xml)
        self.model = compile_spec(spec)
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)          # (scratch for mj_setConst)
        self._index_model(sensors)

        self._physics = self._renderer = self._view_renderer = self._vision_renderer = None
        self.step_count = 0
        self.dt = self.TIMESTEP * self.FRAME_SKIP
        self.max_steps = max_steps or int(round(self.EPISODE_SECONDS / self.dt))
        self.action_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self._last_outcome = None
        self.obs_format = "flat"
        self.obs_modes = {k: False for k in self.input_names}
        self.reset(seed=0)
        self.set_obs_format(obs_format)
        self.set_obs_modes(self.resolve_obs_modes(obs_modes, privileged))

    # ================================================================ model
    def _travel(self):
        return (self.OPEN_WIDTH - self.MIN_WIDTH) / 2

    def _gripper_xml(self, depth, lateral):
        H, W0 = self.HOME, self.OPEN_WIDTH
        lo = self.WS_LO - H - 0.02
        hi = self.WS_HI - H + 0.02
        palm_z = max(self.PALM_Z, lateral + 0.015)
        bar_y = depth + self.FRAME_GAP + 0.006            # finger bar centre, from the tip
        palm_hy = W0 / 2 + bar_y + 0.012
        self._joint_lo = np.array([*lo, -self.YAW_LIMIT, 0.0, 0.0])
        self._joint_hi = np.array([*hi, self.YAW_LIMIT, self._travel(), self._travel()])

        def finger(side, sgn):                           # left: +y, closes toward -y
            return (f'<body name="finger_{side}" pos="0 {sgn * W0 / 2:.4f} 0" gravcomp="1">'
                    f'<joint name="finger_{side}" type="slide" axis="0 {-sgn} 0" limited="true" '
                    f'range="0 {self._travel():.4f}" damping="2" armature="0.02"/>'
                    f'<geom name="finger_{side}_bar" type="box" pos="0 {sgn * bar_y:.4f} {palm_z / 2:.4f}" '
                    f'size="0.012 0.006 {palm_z / 2:.4f}" material="gripper" mass="0.15"/>'
                    '</body>')
        slides = "".join(
            f'<joint name="grip_{a}" type="slide" axis="{ax}" limited="true" '
            f'range="{lo[i]:.4f} {hi[i]:.4f}" damping="20" armature="0.1"/>'
            for i, (a, ax) in enumerate((("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1"))))
        return (
            '<camera name="overhead" pos="0.70 -0.40 1.25" xyaxes="1 0 0 0 0.8 0.6"/>'
            f'<body name="gripper" pos="{H[0]:.4f} {H[1]:.4f} {H[2]:.4f}" gravcomp="1">{slides}'
            # (the shared scene reads angles in degrees)
            f'<joint name="grip_yaw" type="hinge" axis="0 0 1" limited="true" '
            f'range="{-np.degrees(self.YAW_LIMIT):.3f} {np.degrees(self.YAW_LIMIT):.3f}" '
            'damping="1" armature="0.01"/>'
            f'<geom name="palm" type="box" pos="0 0 {palm_z + 0.012:.4f}" size="0.03 {palm_hy:.4f} 0.012" '
            'material="gripper" mass="0.6"/>'
            f'<geom name="column" type="cylinder" fromto="0 0 {palm_z + 0.024:.4f} 0 0 {palm_z + 0.5:.4f}" '
            'size="0.02" material="gantry" contype="0" conaffinity="0" mass="0.3"/>'
            + finger("left", 1) + finger("right", -1) + '</body>')

    def _actuator_xml(self):
        lo, hi = self._joint_lo, self._joint_hi
        gains = ((self.KP_XYZ, self.KV_XYZ, 200.0),) * 2 + ((self.KP_Z, self.KV_Z, 200.0),) \
            + ((self.KP_YAW, self.KV_YAW, 20.0),) + ((self.FINGER_KP, self.FINGER_KV, self.FINGER_FORCE),) * 2
        # targets are clipped in _joint_targets; no ctrlrange on the hinge (degrees in this
        # scene would make its units ambiguous)
        acts = "".join(
            f'<general name="{n}" joint="{n}" biastype="affine" gainprm="{kp}" biasprm="0 {-kp} {-kv}" '
            + (f'ctrllimited="true" ctrlrange="{lo[i]:.4f} {hi[i]:.4f}" ' if n != "grip_yaw" else
               'ctrllimited="false" ')
            + f'forcelimited="true" forcerange="{-f} {f}"/>'
            for i, (n, (kp, kv, f)) in enumerate(zip(GRIP_JOINTS, gains, strict=True)))
        return f"<actuator>{acts}</actuator>"

    def _build_spec(self, sensor_xml=None, robot_xml=None):
        sensor = self._sensor_spec(sensor_xml)
        info = sensor_info(sensor)
        self.hand_info = info
        depth, lateral = sensor_extent(sensor, info)
        # the sensors reach `lateral` below the grasp line: grasping lower presses them into
        # the table (80 mm across for the TacTip, more than half the objects are tall)
        self.sensor_lateral = lateral
        self.grasp_z_min = self.TABLE_TOP + lateral + self.GRASP_CLEARANCE
        materials = ('<material name="gripper" rgba="0.2 0.22 0.25 1"/>'
                     '<material name="gantry" rgba="0.6 0.6 0.65 1"/>')
        spec = mujoco.MjSpec.from_string(self._scene(robot_asset=materials,
                                                     robot_body=self._gripper_xml(depth, lateral),
                                                     extra=self._actuator_xml()))
        for side, sgn in (("left", 1), ("right", -1)):
            finger = spec.body(f"finger_{side}")
            root = attach_sensor(spec, finger, sensor, prefix=f"{side}_", tip_at=[0, 0, 0],
                                 axis_to=[0, -sgn, 0], stiffness_scale=self.sensor_stiffness)
            for b in _subtree(root):
                b.gravcomp = 1.0
                spec.add_exclude(bodyname1="gripper", bodyname2=b.name)
            finger.add_site(name=f"{side}_tip", pos=[0, 0, 0], size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5])
        self.tip_local = np.zeros(3)
        return spec

    def _index_robot(self):
        m = self.model
        jids = [m.joint(n).id for n in GRIP_JOINTS]
        self._g_q = np.array([m.jnt_qposadr[j] for j in jids])
        self._g_v = np.array([m.jnt_dofadr[j] for j in jids])
        self._g_act = np.array([m.actuator(n).id for n in GRIP_JOINTS])
        self._tip = {s: m.site(f"{s}_tip").id for s in SIDES}

    # ================================================================ robot frame = world
    def _base_pose(self, d=None):
        return np.zeros(3), 0.0

    def _joint_targets(self, centre, width, yaw):
        f = (self.OPEN_WIDTH - width) / 2
        q = np.array([*(np.asarray(centre) - self.HOME), yaw, f, f])
        return np.clip(q, self._joint_lo, self._joint_hi)

    # ================================================================ reset
    def _reset_robot(self, rng):
        d = self.data
        self._centre_cmd = np.asarray(self.START_CENTRE, float).copy()
        self._width_cmd = float(self.START_WIDTH)
        self._yaw_cmd = 0.0
        self._cmd = self._hand_targets(self._centre_cmd, self._width_cmd)
        q = self._joint_targets(self._centre_cmd, self._width_cmd, self._yaw_cmd)
        d.qpos[self._g_q] = q
        d.qvel[self._g_v] = 0.0
        d.ctrl[self._g_act] = q
        self._q_cmd = q

    def _choose_target(self, xy, rng):
        lo = np.maximum(self.WS_LO[:2], [self.TABLE_X[0], -self.TABLE_HALF_WIDTH]) + self.TARGET_MARGIN
        hi = np.minimum(self.WS_HI[:2], [self.TABLE_X[1], self.TABLE_HALF_WIDTH]) - self.TARGET_MARGIN
        for _ in range(300):
            dist = rng.uniform(*self._lerp(self.TARGET_DIST))
            ang = rng.uniform(-np.pi, np.pi)
            c = np.asarray(xy) + dist * np.array([np.cos(ang), np.sin(ang)])
            if np.all(c >= lo) and np.all(c <= hi):
                return c
        return np.clip(np.asarray(xy) + [0.12, 0.0], lo, hi)

    # ================================================================ control
    def _apply_action(self, a):
        k = self.ACTION_SMOOTHING
        self._vel_cmd = k * self._vel_cmd + (1 - k) * a
        v = self._vel_cmd
        self._yaw_cmd = float(np.clip(self._yaw_cmd + v[4] * self.GRASP_YAW_SPEED,
                                      -self.GRASP_YAW_MAX, self.GRASP_YAW_MAX))
        tips = self._tips_local()
        # moving and squeezing are leashed separately, so carrying never loosens the grip
        mc = (tips["left"] + tips["right"]) / 2
        mw = float((tips["left"] - tips["right"]) @ np.array([-np.sin(self._yaw_cmd), np.cos(self._yaw_cmd), 0.0]))
        c = np.clip(self._centre_cmd + v[0:3] * self.CENTRE_SPEED, self.WS_LO, self.WS_HI)
        lead = c - mc
        n = np.linalg.norm(lead)
        if n > self.LEASH:
            c = mc + lead * (self.LEASH / n)
        w = self._width_cmd + v[3] * self.WIDTH_SPEED
        w = float(np.clip(np.clip(w, mw - self.SQUEEZE_MAX, mw + self.LEASH), *self.WIDTH_RANGE))
        self._centre_cmd, self._width_cmd = c, w
        self._cmd = self._hand_targets(c, w)
        q_new = self._joint_targets(c, w, self._yaw_cmd)
        q_old = self._q_cmd
        d = self.data
        n_sub = self.FRAME_SKIP // self.CTRL_CHUNKS
        for i in range(self.CTRL_CHUNKS):
            d.ctrl[self._g_act] = q_old + (i + 1) / self.CTRL_CHUNKS * (q_new - q_old)
            self._mj_step(n_sub)
        self._q_cmd = q_new

    # ================================================================ observation
    def _proprio(self, tips):
        lag = np.concatenate([(self._cmd[s] - tips[s]) / self.LEASH for s in SIDES])
        return np.concatenate([tips["left"], tips["right"], np.clip(lag, -3, 3), self._vel_cmd,
                               self._target, [self._yaw_cmd, float(self._was_lifted)]])
