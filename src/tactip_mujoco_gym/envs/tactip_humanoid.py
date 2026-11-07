"""
TacHumanoid: a wheeled humanoid (Rainbow Robotics RB-Y1, MuJoCo Menagerie) whose two hands
are TacTips. It picks up an object by squeezing it between the two sensors, carries it to a
target on the table and puts it down.

    env = gym.make("tactip_mujoco_gym:TacHumanoid-pick", obs_modes={"tactile": True})

Objects (one per episode, drawn from OBJECT_NAMES): five soft ones (cube, brick, ball, egg,
cylinder: a core covered in spring-loaded pads that dent where they are pressed) and five rigid
ones (box, cylinder, sphere, capsule, prism). Every episode randomises the object's stiffness
(soft: foam .. firm rubber; rigid: rigid), mass, surface friction, position and yaw, and where
the target is. Gripping too lightly lets heavy or slippery objects slide out; squeezing too
hard crushes soft ones.

Robot: the base drives like a differential-drive platform (forward speed + turn rate); the
6-joint torso and the head hold a fixed pose leaning over the table. Each TacTip is mounted
like a palm, on the side of the wrist, so the forearms come down from above while the two
sensors face each other; each tip is moved in Cartesian space by inverse kinematics.

Actions (action_mode):
    "bimanual"     (default, 7) base forward speed, turn rate, grasp centre x, y, z (robot
                   frame, velocity), grasp width (velocity: negative closes the hands),
                   grasp yaw (turn rate of the line between the hands, up to +-0.8 rad)
    "independent"  (9) base forward speed, turn rate, left tip x y z, right tip x y z, grasp
                   yaw

Observation inputs (obs_modes), as in the other tasks; proprioception is always included:
    proprio     both tips in the robot frame, how far each lags its command, the smoothed
                action, base speed and turn rate, the target relative to the robot, and
                whether the object has been lifted yet
    object      what a camera-based object tracker would give: the object's position
                (relative to the hands and to the target), velocity, orientation, size, shape
                and height above the table
    privileged  "object" plus what touch has to work out: stiffness, mass, friction, how
                squashed it is and the grip force on each hand
    tactile     both TacTips' markers: displacement from rest and change since last step (mm)
    vision      "headcam" on the head, looking down at the table

Episode: 20 s at 50 Hz. Success: the object, after being lifted, rests within 5 cm of the
target with both hands off it for 0.2 s. Fail: it falls off the table, or a soft object is
squashed flat (a pad pushed in all the way) for 0.06 s.
"""
import os
from importlib.resources import files

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from tactip_mujoco_gym.envs.base import TactileGymEnv
from tactip_mujoco_gym.sensor_tools import attach_sensor, extract_sensor, sensor_info

SIDES = ("left", "right")

# ------------------------------------------------------------------------- the objects
# Soft objects are a light core covered in spring-loaded pads: each pad slides along its
# surface normal on a sprung joint (up to PAD_TRAVEL inward), so pressing a TacTip into the
# object dents it locally, while the pad stays fixed sideways and passes the grip's friction
# to the core. (MuJoCo's flex bodies were tried first: squeeze grasps on them slip, so a
# lifted flex object slides out of the hands.)
# name: (shape, dimensions in m) - box: half sizes; ellipsoid: semi-axes; cylinder: radius,
# half height
SOFT_OBJECTS = {
    "soft_cube":     ("box", (0.03, 0.03, 0.03)),           # 6 cm
    "soft_brick":    ("box", (0.04, 0.02, 0.02)),           # 8 x 4 x 4 cm
    "soft_ball":     ("ellipsoid", (0.035, 0.035, 0.035)),  # 7 cm
    "soft_egg":      ("ellipsoid", (0.045, 0.03, 0.03)),    # 9 x 6 x 6 cm, lying
    "soft_cylinder": ("cylinder", (0.035, 0.035)),          # 7 cm, upright
}
# All objects use condim 4 (sliding + twisting friction): a dome pressing on a surface resists
# the object pivoting about the line between the two hands, as real fingertips do.
RIGID_OBJECTS = {
    "box":      '<geom type="box" size="0.03 0.03 0.03"/>',
    "cylinder": '<geom type="cylinder" size="0.032 0.04"/>',
    "sphere":   '<geom type="sphere" size="0.035"/>',
    "capsule":  '<geom type="capsule" size="0.028 0.035" euler="0 90 0"/>',    # (degrees here)
    "prism":    '<geom type="mesh" mesh="prism"/>',
}
OBJECT_NAMES = tuple(SOFT_OBJECTS) + tuple(RIGID_OBJECTS)
# triangular prism lying like a roof: triangle in x-z, extruded 6 cm along y
PRISM = "-0.035 -0.03 -0.03  0.035 -0.03 -0.03  0 -0.03 0.0306  -0.035 0.03 -0.03  0.035 0.03 -0.03  0 0.03 0.0306"
# rotational symmetry about z (for aligning a grasp): order n -> faces repeat every 2pi/n
SYMMETRY = {"soft_cube": 4, "box": 4, "soft_brick": 2, "prism": 2, "soft_egg": 2, "capsule": 2,
            "soft_ball": 0, "sphere": 0, "soft_cylinder": 0, "cylinder": 0}       # 0 = round

PAD_PITCH  = 0.02                 # m between pad centres
PAD_T      = 0.004                # m: pad half thickness
PAD_TRAVEL = 0.010                # m: how far a pad can be pushed in
PAD_ARMATURE = 0.015              # kg along the pad's axis: keeps stiff pads stable at 2 ms
PAD_DAMPING_RATIO = 0.3
PAD_MASS_FRACTION = 0.1           # of the object's mass, shared by the pads
RIGID_LOG_STIFFNESS = 4.0         # "log10 pad stiffness (N/m)" reported for rigid objects


def _tangents(n):
    n = np.asarray(n, float) / np.linalg.norm(n)
    t1 = np.cross([0, 0, 1.0], n) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
    t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)


def _fibonacci(n):
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    r = np.sqrt(1 - z * z)
    phi = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def soft_pads(shape, dims):
    """Pads covering a soft object: (centre, normal, tangent, geom type, geom size) each, in
    the object's frame. The pad's outer surface sits on the object's surface."""
    pads, gap = [], 0.0006
    if shape == "box":
        h = np.asarray(dims, float)
        for ax in range(3):
            o = [i for i in range(3) if i != ax]
            nu, nv = (max(2, int(round(2 * h[k] / PAD_PITCH))) for k in o)
            du, dv = h[o[0]] / nu, h[o[1]] / nv
            for sgn in (-1, 1):
                n = np.zeros(3)
                n[ax] = sgn
                t1 = np.zeros(3)
                t1[o[0]] = 1.0
                for i in range(nu):
                    for j in range(nv):
                        p = np.zeros(3)
                        p[ax] = sgn * (h[ax] - PAD_T)
                        p[o[0]] = -h[o[0]] + (2 * i + 1) * du
                        p[o[1]] = -h[o[1]] + (2 * j + 1) * dv
                        pads.append((p, n, t1, "box", (du - gap / 2, dv - gap / 2, PAD_T)))
    elif shape == "ellipsoid":
        D = np.asarray(dims, float)
        area = 4 * np.pi * (((D[0] * D[1]) ** 1.6 + (D[0] * D[2]) ** 1.6 + (D[1] * D[2]) ** 1.6) / 3) ** (1 / 1.6)
        count = int(np.ceil(area / PAD_PITCH ** 2 * 1.1))
        for u in _fibonacci(count):
            s = D * u
            n = u / D
            n /= np.linalg.norm(n)
            local_area = 4 * np.pi / count * np.prod(D) * np.linalg.norm(u / D)
            a = 1.15 * np.sqrt(local_area / np.pi)           # discs overlap a little
            pads.append((s - PAD_T * n, n, _tangents(n)[0], "ellipsoid", (a, a, PAD_T)))
    elif shape == "cylinder":
        r, hh = dims
        n_ring = max(2, int(round(2 * hh / PAD_PITCH)))
        n_around = max(6, int(round(2 * np.pi * r / PAD_PITCH)))
        dz, half_arc = hh / n_ring, np.pi * r / n_around
        for i in range(n_ring):
            z = -hh + (2 * i + 1) * dz
            for k in range(n_around):
                ang = 2 * np.pi * (k + 0.5 * (i % 2)) / n_around
                n = np.array([np.cos(ang), np.sin(ang), 0.0])
                t1 = np.array([-np.sin(ang), np.cos(ang), 0.0])
                pads.append(((r - PAD_T) * n + [0, 0, z], n, t1, "box",
                             (half_arc - gap / 2, dz - gap / 2, PAD_T)))
        ring = max(5, int(round(2 * np.pi * (r * 0.62) / PAD_PITCH)))
        for sgn in (-1, 1):                                   # end caps: a disc and a ring
            n = np.array([0, 0, float(sgn)])
            z = sgn * (hh - PAD_T)
            pads.append((np.array([0, 0, z]), n, np.array([1.0, 0, 0]), "cylinder", (0.38 * r, PAD_T)))
            for k in range(ring):
                ang = 2 * np.pi * k / ring
                pads.append((np.array([0.69 * r * np.cos(ang), 0.69 * r * np.sin(ang), z]), n,
                             np.array([1.0, 0, 0]), "cylinder", (0.3 * r, PAD_T)))
    else:
        raise ValueError(shape)
    return pads


def soft_core(shape, dims):
    """The core's geom: the object shrunk by a pad's thickness and travel (a backstop for a
    fully squashed pad)."""
    inset = 2 * PAD_T + PAD_TRAVEL
    if shape == "cylinder":
        size = (max(0.004, dims[0] - inset), max(0.004, dims[1] - inset))
    else:
        size = tuple(max(0.004, d - inset) for d in dims)
    return shape, size


_GEOM_TYPE = {"box": mujoco.mjtGeom.mjGEOM_BOX, "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
              "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER}
N_PADS = max(len(soft_pads(*v)) for v in SOFT_OBJECTS.values())


def _geom_bounds(gtype, size):
    """(bounding-sphere radius, aabb half sizes) of a primitive geom, as MuJoCo computes them."""
    size = np.asarray(size, float)
    if gtype == "box":
        return float(np.linalg.norm(size)), size
    if gtype == "ellipsoid":
        return float(size.max()), size
    r, h = size[0], size[1]                                    # cylinder
    return float(np.hypot(r, h)), np.array([r, r, h])


def soft_layout(name):
    """Everything that turns the shared soft body into this object: per pad its position,
    orientation, geom type, size and bounds, and the core's geom (in the object frame)."""
    shape, dims = SOFT_OBJECTS[name]
    pads = soft_pads(shape, dims)
    quat = np.zeros((len(pads), 4))
    for i, (_, n, t1, _, _) in enumerate(pads):
        mujoco.mju_mat2Quat(quat[i], np.column_stack([t1, np.cross(n, t1), n]).ravel())
    size = np.array([[*sz, 0.0][:3] for *_, sz in pads])
    bounds = [_geom_bounds(t, sz) for *_, t, sz in pads]
    ctype, csize = soft_core(shape, dims)
    return dict(pos=np.array([p for p, *_ in pads]), quat=quat,
                type=np.array([_GEOM_TYPE[t] for *_, t, _ in pads]), pad_size=size,
                rbound=np.array([b[0] for b in bounds]), aabb=np.array([b[1] for b in bounds]),
                core_type=_GEOM_TYPE[ctype], core_size=np.array([*csize, 0.0][:3]),
                core_bounds=_geom_bounds(ctype, csize), outer=(shape, dims))


def _soft_body_xml():
    """One soft body for every soft object: a core and N_PADS sprung pads, rearranged (pads
    moved, resized, the unused ones switched off) for each episode's shape."""
    body = ['<body name="soft" pos="-3 3 0.5"><freejoint name="soft"/>'
            '<geom name="soft_core" type="box" size="0.01 0.01 0.01" mass="0.18" condim="4" priority="2" '
            'friction="0.8 0.005 0.0001" contype="3" conaffinity="3" rgba="0.5 0.3 0.15 1"/>']
    for i in range(N_PADS):
        body.append(
            f'<body name="soft_pad{i}" pos="0 0 0">'
            f'<joint name="soft_pad{i}" type="slide" axis="0 0 1" range="{-PAD_TRAVEL} 0" limited="true" '
            f'stiffness="500" damping="1" armature="{PAD_ARMATURE}" frictionloss="0"/>'
            f'<geom name="soft_pad{i}" type="box" size="0.005 0.005 {PAD_T}" mass="0.0004" condim="4" priority="2" '
            f'friction="0.8 0.005 0.0001" contype="4" conaffinity="3" rgba="0.9 0.6 0.3 1"/></body>')
    body.append("</body>")
    return "".join(body)


def _rigid_body_xml():
    """One rigid body carrying every rigid object's geom; only the episode's one collides."""
    geoms = "".join(g.replace("<geom ", f'<geom name="{name}" mass="0.2" priority="2" condim="4" '
                                        f'friction="0.8 0.005 0.0001" contype="3" conaffinity="3" '
                                        f'rgba="0.3 0.6 0.9 1" ')
                    for name, g in RIGID_OBJECTS.items())
    return f'<body name="rigid" pos="-3 4 0.5"><freejoint name="rigid"/>{geoms}</body>'


def _unit_inertia(geom):
    """(principal inertia, inertial pos, inertial quat) of a 1 kg body made of this geom."""
    geom = geom.replace("<geom ", '<geom mass="1" ')
    m = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><mesh name="prism" vertex="{PRISM}"/></asset><worldbody><body>'
        f'<freejoint/>{geom}</body></worldbody></mujoco>')
    return m.body_inertia[1].copy(), m.body_ipos[1].copy(), m.body_iquat[1].copy()


def _asset(*parts):
    return str(files("tactip_mujoco_gym").joinpath(*parts))


def _rot_z(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def _close(d, sigma):
    return 1.0 - np.tanh(np.abs(d) / sigma)


class TacHumanoidPick(TactileGymEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # ---------------------------------------------------------------- robot + sensors
    ROBOT_FILE    = "rby1a_1.2_no_gripper.xml"          # found anywhere under assets/
    SENSOR_XML    = ("assets", "sensor", "tactip.xml")
    SENSOR_SOURCE = ("assets", "tactip_arm.xml")
    SENSOR_ROOT   = "auto"
    FLANGE        = 0.134         # m: end of the wrist link (link_*_arm_6) along its -z
    PALM_DROP     = 0.02          # m: the TacTip's axis sits this far beyond the wrist's end
    TORSO_POSE    = np.array([0.0, 0.4, 0.0, 0.0, 0.0, 0.0])   # leans over the table, chest
                                  # far enough back that the forearms clear it
    HEAD_POSE     = np.array([0.0, 0.0])
    ARM_KP, ARM_KV = 800.0, 40.0  # arm joint servos (Nm/rad, Nm s/rad): compliant enough to
                                  # squeeze gently, stiff enough to carry 1 kg
    TRACK_BODY, TRACK_DISTANCE, TRACK_ELEVATION = "link_torso_5", 1.3, -35
    VISION_CAMERA = "headcam"
    OBS_MODE_NAMES = ("vision", "tactile", "privileged", "object")

    TIMESTEP   = 0.002
    FRAME_SKIP = 10               # 50 Hz policy
    CTRL_CHUNKS = 5               # arm targets are blended in over the physics steps
    EPISODE_SECONDS = 20.0

    # ---------------------------------------------------------------- scene
    TABLE_TOP  = 0.72             # m
    TABLE_X    = (0.40, 1.00)     # near and far edge, robot frame at the start
    TABLE_HALF_WIDTH = 0.80
    OBJECT_X   = (0.50, 0.65)     # spawn region (robot frame at the start)
    OBJECT_Y   = (-0.08, 0.08)

    # ---------------------------------------------------------------- control
    ACTION_SMOOTHING = 0.6
    CENTRE_SPEED = 0.005          # m per step (0.25 m/s)
    WIDTH_SPEED  = 0.004          # m per step
    GRASP_YAW_SPEED = 0.03        # rad per step
    GRASP_YAW_MAX   = 0.8         # rad: both palms can turn this far (about z) and still
                                  # face each other; covers every face of a cube
    BASE_SPEED   = 0.30           # m/s
    TURN_RATE    = 0.6            # rad/s
    BASE_ACCEL   = 1.0            # m/s^2 and ...
    TURN_ACCEL   = 2.0            # ... rad/s^2: the base ramps up to the commanded speeds
    LEASH        = 0.015          # m: a tip command may lead the measured tip by this much
    SQUEEZE_MAX  = 0.03           # m: the width command may sit this far inside the measured
                                  # gap between the tips (bounds the grip force)
    # grasp workspace (robot frame): where both palms reach without the arms touching the
    # body; further to the side, the base turns
    WS_LO = np.array([0.45, -0.10, TABLE_TOP + 0.01])
    WS_HI = np.array([0.72, 0.10, TABLE_TOP + 0.20])
    WIDTH_RANGE = (0.0, 0.25)
    START_CENTRE = np.array([0.55, 0.0, TABLE_TOP + 0.12])
    START_WIDTH = 0.15
    REACH_R = (0.50, 0.70)        # m from the robot's spot (x = 0): where the hands can put
                                  # an object down once the base faces it. The base can back
                                  # off and turn, but not drive closer than its spot: the
                                  # torso would hit the table.
    IK_ITERS = 4
    IK_MAX_STEP = 0.15            # rad: largest joint change per IK iteration
    ROLL_WEIGHT = 0.2             # the TacTip axis is held exactly; turning about it (where
                                  # the forearm comes from) is only a weak preference
    NULL_GAIN = 0.05
    ARM_SEEDS = (                                      # start-pose IK tries these
        {"left": np.array([0.0, 0.3, 0.0, -1.5, 0.0, 0.0, 0.0]),
         "right": np.array([0.0, -0.3, 0.0, -1.5, 0.0, 0.0, 0.0])},
        {"left": np.array([0.5, 0.6, 0.0, -1.8, 0.0, 1.2, 0.0]),
         "right": np.array([0.5, -0.6, 0.0, -1.8, 0.0, 1.2, 0.0])},
        {"left": np.array([-0.5, 0.8, 0.0, -1.2, 0.5, 1.0, 0.0]),
         "right": np.array([-0.5, -0.8, 0.0, -1.2, -0.5, 1.0, 0.0])},
    )

    # ---------------------------------------------------------------- task
    LIFT_MIN     = 0.03           # m above the table: counts as lifted
    LIFT_HEIGHT  = 0.08           # m: full lift reward
    PLACE_RADIUS = 0.05
    RELEASE_STEPS = 10            # hands off and object still for this long = placed
    GRIP_FORCE   = 0.5            # N per hand to count as gripping
    CRUSH_STRAIN = 0.6            # a pad pushed in further than this (fraction of its
                                  # travel) is penalised ...
    CRUSH_FAIL, CRUSH_STEPS = 0.95, 3   # ... and pushed in all the way for 3 steps ends it
    R_SUCCESS, R_FAIL = 100.0, -10.0
    # (level 0 range, level 1 range)
    MASS       = ((0.05, 0.3), (0.05, 1.0))      # kg, log-uniform
    FRICTION   = ((0.7, 1.2), (0.25, 1.2))
    STIFFNESS  = ((800.0, 4000.0), (200.0, 4000.0))   # N/m per pad, log-uniform (soft objects)
    STIFF_PER_KG = 800.0          # soft objects are at least this x mass(kg) / friction stiff:
                                  # a firm enough grip to hold a heavy, slippery one must not
                                  # squash it flat
    TARGET_DIST = ((0.10, 0.18), (0.12, 0.60))   # m from the object, on the table
    TARGET_BEARING = (0.15, 0.75)                # rad: how far to the side of the robot's
                                                 # heading the target may be (level 0, 1);
                                                 # beyond ~0.15 the base has to turn
    YAW        = ((-0.3, 0.3), (-0.8, 0.8))      # object yaw away from square to the hands
                                                 # (plus turns by its symmetry); round
                                                 # objects: any
    BASE_BACK  = ((0.0, 0.0), (0.0, 0.25))       # m the robot starts back from its spot
    LEVEL_UP, LEVEL_DOWN = 0.02, 0.04   # per success / per miss (failure or timeout): the
                                        # level settles where about 2 in 3 episodes succeed

    def __init__(self, objects=None, level=None, curriculum=True, action_mode="bimanual",
                 sensor_stiffness=10.0, obs_modes=None, privileged=None, obs_format="flat",
                 render_mode=None, max_steps=None, sensor_xml=None, robot_xml=None):
        """
        objects           names from OBJECT_NAMES to draw from (default: all ten)
        level             0..1 fixes the difficulty (mass, friction, softness, target
                          distance, start distance); None starts at 0 with a curriculum
        action_mode       "bimanual" (7) or "independent" (9)
        sensor_stiffness  scales the TacTips' skin (see sensor_tools.attach_sensor)

        reset(options=...) accepts "object" (a name), "stiffness" (N/m per pad), "mass",
        "friction", "yaw", "level".
        """
        gym.Env.__init__(self)
        if action_mode not in ("bimanual", "independent"):
            raise ValueError("action_mode must be 'bimanual' or 'independent'")
        self.object_names = tuple(objects) if objects else OBJECT_NAMES
        bad = [o for o in self.object_names if o not in OBJECT_NAMES]
        if bad:
            raise ValueError(f"unknown objects {bad}; choose from {OBJECT_NAMES}")
        self.action_mode = action_mode
        self.sensor_stiffness = float(sensor_stiffness)
        self.curriculum = curriculum and level is None
        self.level = 0.0 if level is None else float(np.clip(level, 0, 1))
        self.render_mode = render_mode
        self.xml_path = None

        spec = self._build_spec(sensor_xml, robot_xml)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._ik_data = mujoco.MjData(self.model)
        self._index_model()

        self._physics = self._renderer = self._view_renderer = self._vision_renderer = None
        self.step_count = 0
        self.dt = self.TIMESTEP * self.FRAME_SKIP
        self.max_steps = max_steps or int(round(self.EPISODE_SECONDS / self.dt))
        self.events = False
        n_act = 7 if action_mode == "bimanual" else 9
        self.action_space = spaces.Box(-1.0, 1.0, shape=(n_act,), dtype=np.float32)
        self._last_outcome = None
        self.obs_format = "flat"
        self.obs_modes = {k: False for k in self.OBS_MODE_NAMES}
        self.reset(seed=0)
        self.set_obs_format(obs_format)
        self.set_obs_modes(self.resolve_obs_modes(obs_modes, privileged))

    # ================================================================ model building
    @classmethod
    def find_robot(cls, robot_xml=None):
        if robot_xml is not None:
            return str(robot_xml)
        found = sorted(files("tactip_mujoco_gym").joinpath("assets").rglob(cls.ROBOT_FILE))
        if not found:
            raise FileNotFoundError(
                f"No {cls.ROBOT_FILE} under {_asset('assets')}. Download RB-Y1 with "
                "`python tools/fetch_menagerie.py rby1`, or pass robot_xml=...")
        return str(found[0])

    def _sensor_spec(self, sensor_xml):
        path = sensor_xml or _asset(*self.SENSOR_XML)
        if os.path.exists(path):
            return mujoco.MjSpec.from_file(path)
        src = _asset(*self.SENSOR_SOURCE)
        print(f"[TacHumanoid] {path} not found: using the sensor in {src}")
        return extract_sensor(src, self.SENSOR_ROOT)

    def _scene_xml(self, robot_path):
        """Robot, floor, table, target marker and every object (parked), as one MJCF."""
        tz = self.TABLE_TOP
        tx = (self.TABLE_X[0] + self.TABLE_X[1]) / 2
        thx = (self.TABLE_X[1] - self.TABLE_X[0]) / 2
        hl = tz / 2 - 0.02                                   # leg half length
        legs = "".join(
            f'<geom type="box" size="0.025 0.025 {hl:.4f}" pos="{x:.3f} {y:.3f} {hl:.4f}" '
            f'material="table_leg" contype="1" conaffinity="1"/>'
            for x in (-thx + 0.05, thx - 0.05)
            for y in (-self.TABLE_HALF_WIDTH + 0.05, self.TABLE_HALF_WIDTH - 0.05))
        objs = [_soft_body_xml(), _rigid_body_xml()]
        return f"""
<mujoco model="tachumanoid">
  <option timestep="{self.TIMESTEP}" integrator="implicitfast" cone="elliptic" impratio="10"
          solver="CG" iterations="50" tolerance="1e-6"/>
  <visual><global offwidth="640" offheight="480"/><headlight ambient="0.35 0.35 0.35" diffuse="0.5 0.5 0.5"/></visual>
  <asset>
    <model name="robot" file="{robot_path}"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" rgb1="0.2 0.3 0.4" rgb2="0.1 0.15 0.2" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="1 1" texuniform="true"/>
    <material name="table" rgba="0.75 0.65 0.5 1"/>
    <material name="table_leg" rgba="0.4 0.35 0.3 1"/>
    <mesh name="prism" vertex="{PRISM}"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1" directional="true" diffuse="0.6 0.6 0.6" castshadow="false"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="floor" contype="2" conaffinity="2"/>
    <body name="table" pos="{tx:.3f} 0 0">
      <geom name="table_top" type="box" size="{thx:.3f} {self.TABLE_HALF_WIDTH} 0.02" pos="0 0 {tz - 0.02}"
            material="table" contype="1" conaffinity="1" friction="0.8"/>
      {legs}
    </body>
    <site name="target" type="cylinder" size="{self.PLACE_RADIUS} 0.001" pos="0.6 0.2 {tz + 0.001}"
          rgba="0.2 0.9 0.3 0.5"/>
    <attach model="robot" body="base" prefix=""/>
    {''.join(objs)}
  </worldbody>
</mujoco>"""

    def _build_spec(self, sensor_xml=None, robot_xml=None):
        path = os.path.abspath(self.find_robot(robot_xml))
        spec = mujoco.MjSpec.from_string(self._scene_xml(path))
        base = spec.body("base")

        # The robot file's default joint (damping 5, range +-3.14) also reaches the objects'
        # free joints once the robot is attached: free them again (no limit, damping or
        # armature). The pads' joints set all of these themselves.
        for name in ("soft", "rigid"):
            for j in spec.body(name).joints:
                j.limited = mujoco.mjtLimited.mjLIMITED_FALSE
                j.damping = 0.0
                j.armature = 0.0
                j.frictionloss = 0.0

        # ---- base: free joint -> planar joints driven by velocity servos (diff-drive)
        for j in list(base.joints):
            spec.delete(j)
        # (explicitly unlimited: the robot file's default range +-3.14 would otherwise reach
        # these joints, and this scene reads it in degrees)
        free = dict(damping=0, armature=0, limited=mujoco.mjtLimited.mjLIMITED_FALSE)
        base.add_joint(name="base_x", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], **free)
        base.add_joint(name="base_y", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, 1, 0], **free)
        base.add_joint(name="base_yaw", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1], **free)
        for j, kv in (("base_x", 2e4), ("base_y", 2e4), ("base_yaw", 4e3)):
            a = spec.add_actuator(name=f"{j}_vel", target=j, trntype=mujoco.mjtTrn.mjTRN_JOINT)
            a.gaintype = mujoco.mjtGain.mjGAIN_FIXED
            a.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            a.gainprm[0] = kv
            a.biasprm[0], a.biasprm[1], a.biasprm[2] = 0.0, 0.0, -kv
            a.ctrlrange = [-2, 2]
            a.ctrllimited = True

        # ---- gravity compensation for the robot; softer arm servos
        def subtree(b):
            out = [b]
            for c in b.bodies:
                out.extend(subtree(c))
            return out
        for b in subtree(base)[1:]:
            b.gravcomp = 1.0
        for a in spec.actuators:
            if "arm" in a.target:
                a.gainprm[0] = self.ARM_KP
                a.biasprm[1], a.biasprm[2] = -self.ARM_KP, -self.ARM_KV

        # ---- TacTip palms: mounted at the wrist's end, tip along the wrist's +x
        sensor = self._sensor_spec(sensor_xml)
        info = sensor_info(sensor)
        self.hand_info = info
        tip_local = np.array([info["reach"], 0.0, -(self.FLANGE + self.PALM_DROP)])
        for side in SIDES:
            wrist = spec.body(f"link_{side}_arm_6")
            root = attach_sensor(spec, wrist, sensor, prefix=f"{side}_", tip_at=tip_local,
                                 axis_to=[1, 0, 0], stiffness_scale=self.sensor_stiffness)
            for b in subtree(root):
                b.gravcomp = 1.0
                spec.add_exclude(bodyname1=f"link_{side}_arm_5", bodyname2=b.name)
            wrist.add_site(name=f"{side}_tip", pos=tip_local, size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5])
        self.tip_local = tip_local

        # ---- head camera, looking down at the table
        spec.body("link_head_2").add_camera(name="headcam", pos=[0.08, 0, 0.0], fovy=70,
                                            xyaxes=[0, -1, 0, 0.6, 0, 0.8])
        # objects: hidden contact class for the parked ones is set at runtime
        return spec

    def _index_model(self):
        m = self.model
        jid = lambda n: m.joint(n).id                                   # noqa: E731
        self._base_q = np.array([m.jnt_qposadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")])
        self._base_v = np.array([m.jnt_dofadr[jid(n)] for n in ("base_x", "base_y", "base_yaw")])
        self._base_act = np.array([m.actuator(f"{n}_vel").id for n in ("base_x", "base_y", "base_yaw")])
        self._wheels = [m.actuator(n).id for n in ("left_wheel_act", "right_wheel_act")]
        self._torso_q = np.array([m.jnt_qposadr[jid(f"torso_{i}")] for i in range(6)])
        self._head_q = np.array([m.jnt_qposadr[jid(f"head_{i}")] for i in range(2)])
        act_of = {int(m.actuator_trnid[a, 0]): a for a in range(m.nu)
                  if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
        self._torso_act = np.array([act_of[jid(f"torso_{i}")] for i in range(6)])
        self._head_act = np.array([act_of[jid(f"head_{i}")] for i in range(2)])
        self._arm_j = {s: np.array([jid(f"{s}_arm_{i}") for i in range(7)]) for s in SIDES}
        self._arm_q = {s: m.jnt_qposadr[self._arm_j[s]] for s in SIDES}
        self._arm_v = {s: m.jnt_dofadr[self._arm_j[s]] for s in SIDES}
        self._arm_act = {s: np.array([act_of[j] for j in self._arm_j[s]]) for s in SIDES}
        self._arm_lo = {s: m.jnt_range[self._arm_j[s], 0] for s in SIDES}
        self._arm_hi = {s: m.jnt_range[self._arm_j[s], 1] for s in SIDES}
        self._tip = {s: m.site(f"{s}_tip").id for s in SIDES}
        self._wrist = {s: m.body(f"link_{s}_arm_6").id for s in SIDES}
        self._target_site = m.site("target").id
        self._table = m.geom("table_top").id

        # hand geoms (each TacTip's colliding geoms) and table geoms
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)

        def under(b, root):
            while b != 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False
        self._hand_geom = np.full(m.ngeom, -1)
        for k, s in enumerate(SIDES):
            root = m.body(f"{s}_{self.hand_info['root']}").id
            for g in np.flatnonzero(collides):
                if under(int(m.geom_bodyid[g]), root):
                    self._hand_geom[g] = k
        table_body = m.body("table").id
        self._is_table = (m.geom_bodyid == table_body)
        base_id = m.body("base").id
        self._robot_geom = np.array([collides[g] and self._hand_geom[g] < 0 and under(int(m.geom_bodyid[g]), base_id)
                                     for g in range(m.ngeom)])

        # objects: a shared soft body (core + N_PADS pads) and a shared rigid body (one geom
        # per rigid object); a record per object says how to set them up
        sb, rb = m.body("soft").id, m.body("rigid").id
        pads = np.array([m.body(f"soft_pad{i}").id for i in range(N_PADS)])
        pad_j = np.array([m.body_jntadr[p] for p in pads])
        self._soft = dict(body=sb, qadr=m.jnt_qposadr[m.joint("soft").id], vadr=m.jnt_dofadr[m.joint("soft").id],
                          park=m.body_pos[sb].copy(), core=m.geom("soft_core").id, pads=pads, pad_j=pad_j,
                          pad_q=m.jnt_qposadr[pad_j], pad_v=m.jnt_dofadr[pad_j],
                          pad_g=np.array([m.geom(f"soft_pad{i}").id for i in range(N_PADS)]))
        self._rigid = dict(body=rb, qadr=m.jnt_qposadr[m.joint("rigid").id], vadr=m.jnt_dofadr[m.joint("rigid").id],
                           park=m.body_pos[rb].copy(), geoms=np.array([m.geom(n).id for n in RIGID_OBJECTS]))
        self._object_geoms = np.concatenate([[self._soft["core"]], self._soft["pad_g"], self._rigid["geoms"]])
        self._geom_contype0 = m.geom_contype.copy()
        self._geom_conaffinity0 = m.geom_conaffinity.copy()
        self._objects = {}
        for k, name in enumerate(OBJECT_NAMES):
            if name in SOFT_OBJECTS:
                lay = soft_layout(name)
                shape, dims = SOFT_OBJECTS[name]
                outer = f'<geom type="{shape}" size="{" ".join(str(v) for v in dims)}"/>'
                full = (2 * np.array(dims) if shape != "cylinder"
                        else 2 * np.array([dims[0], dims[0], dims[1]]))
                n = len(lay["pos"])
                rec = dict(lay, soft=True, n=n, size=full, bottom=full[2] / 2,
                           geoms=np.concatenate([[self._soft["core"]], self._soft["pad_g"][:n]]),
                           **dict(zip(("inertia1", "ipos", "iquat"), _unit_inertia(outer), strict=True)))
            else:
                g = m.geom(name).id
                corners = self._geom_corners_local(g)
                rec = dict(soft=False, geom=g, geoms=np.array([g]),
                           size=corners.max(axis=0) - corners.min(axis=0), bottom=-corners[:, 2].min(),
                           **dict(zip(("inertia1", "ipos", "iquat"), _unit_inertia(RIGID_OBJECTS[name]),
                                     strict=True)))
            rec["index"] = k
            self._objects[name] = rec
        self._geom_obj = np.full(m.ngeom, -1)                 # set per episode
        self._shape_onehot = {n: np.eye(len(OBJECT_NAMES))[i] for i, n in enumerate(OBJECT_NAMES)}

        # TacTip markers, one frame per hand
        ids, owner, frames = [], [], []
        for k, s in enumerate(SIDES):
            mk = [i for i in range(m.nsite) if m.site(i).name.startswith(f"{s}_{self.MARKER_PREFIX}")]
            ids += mk
            owner += [k] * len(mk)
            site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{s}_{self.SENSOR_SITE}")
            frames.append(("site", site) if site >= 0 else ("body", m.body(f"{s}_{self.hand_info['root']}").id))
        self._marker_ids = np.array(ids, dtype=int)
        self._marker_owner = np.array(owner, dtype=int)
        self._hand_frames = frames
        self.n_markers = len(ids)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self._nodes_rest = self._nodes(d)
        self._nodes_last = self._nodes_rest.copy()
        self._nodes_delta = np.zeros_like(self._nodes_rest)

    def _geom_corners_local(self, g):
        m = self.model
        c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.geom_quat[g])
        return m.geom_pos[g] + (c + signs * h) @ R.reshape(3, 3).T

    # ================================================================ tactip read-out
    def _nodes(self, d=None):
        d = self.data if d is None else d
        pos, mat = [], []
        for kind, i in self._hand_frames:
            if kind == "site":
                pos.append(d.site_xpos[i])
                mat.append(d.site_xmat[i].reshape(3, 3))
            else:
                pos.append(d.xpos[i])
                mat.append(d.xmat[i].reshape(3, 3))
        c = np.array(pos)[self._marker_owner]
        R = np.array(mat)[self._marker_owner]
        return np.einsum("ni,nij->nj", d.site_xpos[self._marker_ids] - c, R).astype(np.float32)

    def get_nodes(self):
        return self._nodes()

    def get_hand_nodes(self):
        """Marker displacement from rest per hand (left, right), (2, markers, 3) in metres."""
        return (self._nodes() - self._nodes_rest).reshape(2, -1, 3)

    # ================================================================ robot frame
    def _base_pose(self, d=None):
        d = self.data if d is None else d
        x, y, yaw = d.qpos[self._base_q]
        return np.array([x, y, 0.0]), yaw

    def to_world(self, p_local):
        pos, yaw = self._base_pose()
        return pos + _rot_z(yaw) @ np.asarray(p_local)

    def to_local(self, p_world):
        pos, yaw = self._base_pose()
        return _rot_z(yaw).T @ (np.asarray(p_world) - pos)

    def _tips_local(self):
        return {s: self.to_local(self.data.site_xpos[self._tip[s]]) for s in SIDES}

    def _hand_rot(self, side, yaw):
        """Wrist orientation: wrist end (its -z) pointing down, TacTip axis (its +x) toward
        the other hand."""
        x = np.array([0, -1.0, 0]) if side == "left" else np.array([0, 1.0, 0])
        z = np.array([0, 0, 1.0])
        return _rot_z(yaw) @ np.column_stack([x, np.cross(z, x), z])

    # ================================================================ inverse kinematics
    def _ik(self, side, target_w, R_w, q_start):
        m, d = self.model, self._ik_data
        d.qpos[:] = self.data.qpos
        qa, va = self._arm_q[side], self._arm_v[side]
        d.qpos[qa] = q_start
        sid, body = self._tip[side], self._wrist[side]
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        q_nom = self._q_nominal[side]
        for _ in range(self.IK_ITERS):
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            e_p = target_w - d.site_xpos[sid]
            R = d.xmat[body].reshape(3, 3)
            e_r = np.cross(R[:, 0], R_w[:, 0]) + self.ROLL_WEIGHT * np.cross(R[:, 2], R_w[:, 2])
            mujoco.mj_jacSite(m, d, jp, jr, sid)
            J = np.vstack([jp[:, va], 0.3 * jr[:, va]])
            e = np.concatenate([e_p, 0.3 * e_r])
            ok = np.linalg.norm(e_p) < 1e-4 and np.linalg.norm(np.cross(R[:, 0], R_w[:, 0])) < 1e-3
            if ok:
                break
            JJt = J @ J.T + 1e-3 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, e)
            null = np.eye(7) - J.T @ np.linalg.solve(JJt, J)
            dq += null @ (self.NULL_GAIN * (q_nom - d.qpos[qa]))
            n = np.linalg.norm(dq)
            if n > self.IK_MAX_STEP:
                dq *= self.IK_MAX_STEP / n
            d.qpos[qa] = np.clip(d.qpos[qa] + dq, self._arm_lo[side], self._arm_hi[side])
        mujoco.mj_kinematics(m, d)
        return d.qpos[qa].copy(), float(np.linalg.norm(target_w - d.site_xpos[sid]))

    def _start_pose(self, d):
        """Arm joints for the start pose: IK from each of ARM_SEEDS, keep the best per arm
        (computed once, then reused every reset)."""
        mujoco.mj_forward(self.model, d)
        _, yaw = self._base_pose()
        best = {}
        for side in SIDES:
            tries = []
            for seed in self.ARM_SEEDS:
                self._q_nominal = {s: seed[s].copy() for s in SIDES}
                q = seed[side].copy()
                for _ in range(40):
                    q, err = self._ik(side, self.to_world(self._cmd[side]),
                                      self._hand_rot(side, yaw + self._yaw_cmd), q)
                tries.append((err, q))
            err, best[side] = min(tries, key=lambda t: t[0])
            if err > 0.01:
                print(f"[TacHumanoid] warning: start pose IK error {err * 1000:.0f} mm ({side})")
        return best

    def _solve_arms(self, targets_local):
        _, yaw = self._base_pose()
        out = {}
        for s in SIDES:
            q, err = self._ik(s, self.to_world(targets_local[s]), self._hand_rot(s, yaw + self._yaw_cmd),
                              self._q_cmd[s])
            out[s] = q
        return out

    # ================================================================ objects
    @property
    def _rec(self):
        return self._objects[self.obj]

    @property
    def _body(self):
        """The shared body (soft or rigid) playing this episode's object."""
        return self._soft if self._rec["soft"] else self._rigid

    def _park_all(self):
        """The shared body not in use this episode: far away, contact off, held still."""
        d = self.data
        other = self._rigid if self._rec["soft"] else self._soft
        d.qpos[other["qadr"]:other["qadr"] + 3] = other["park"]
        d.qpos[other["qadr"] + 3:other["qadr"] + 7] = [1, 0, 0, 0]
        d.qvel[other["vadr"]:other["vadr"] + 6] = 0.0
        if other is self._soft:
            d.qpos[other["pad_q"]] = 0.0
            d.qvel[other["pad_v"]] = 0.0
        elif self._rec["n"] < N_PADS:                          # the soft body's spare pads
            n = self._rec["n"]
            d.qpos[self._soft["pad_q"][n:]] = 0.0
            d.qvel[self._soft["pad_v"][n:]] = 0.0

    def _configure_object(self):
        """Contacts on for this object's geoms only; the soft body's pads laid out for it."""
        m, r = self.model, self._rec
        g = self._object_geoms
        m.geom_contype[g] = 0
        m.geom_conaffinity[g] = 0
        m.geom_rgba[g, 3] = 0.0
        on = r["geoms"]
        m.geom_contype[on] = self._geom_contype0[on]
        m.geom_conaffinity[on] = self._geom_conaffinity0[on]
        m.geom_rgba[on, 3] = 1.0
        self._geom_obj[:] = -1
        self._geom_obj[on] = r["index"]
        if r["soft"]:
            S, n = self._soft, r["n"]
            m.body_pos[S["pads"]] = 0.0
            m.body_quat[S["pads"]] = [1, 0, 0, 0]
            m.body_pos[S["pads"][:n]] = r["pos"]
            m.body_quat[S["pads"][:n]] = r["quat"]
            pg = S["pad_g"][:n]
            m.geom_type[pg] = r["type"]
            m.geom_size[pg] = r["pad_size"]
            m.geom_rbound[pg] = r["rbound"]
            m.geom_aabb[pg, :3] = 0.0
            m.geom_aabb[pg, 3:] = r["aabb"]
            c = S["core"]
            m.geom_type[c] = r["core_type"]
            m.geom_size[c] = r["core_size"]
            m.geom_rbound[c] = r["core_bounds"][0]
            m.geom_aabb[c, :3] = 0.0
            m.geom_aabb[c, 3:] = r["core_bounds"][1]
            # each of these bodies has one geom, centred on it: its bounding box (which the
            # broad phase uses) is the geom's
            bv = m.body_bvhadr[S["pads"][:n]]
            m.bvh_aabb[bv, :3] = 0.0
            m.bvh_aabb[bv, 3:] = r["aabb"]
            bv = m.body_bvhadr[S["body"]]
            m.bvh_aabb[bv, :3] = 0.0
            m.bvh_aabb[bv, 3:] = r["core_bounds"][1]

    def _place_object(self, xy_world, yaw):
        d, r, B = self.data, self._rec, self._body
        d.qpos[B["qadr"]:B["qadr"] + 3] = [*xy_world, self.TABLE_TOP + r["bottom"] + 0.002]
        d.qpos[B["qadr"] + 3:B["qadr"] + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[B["vadr"]:B["vadr"] + 6] = 0.0
        if r["soft"]:
            d.qpos[B["pad_q"]] = 0.0
            d.qvel[B["pad_v"]] = 0.0

    def _set_properties(self, stiffness, mass, friction):
        """stiffness: N/m per pad (soft objects; ignored for rigid ones)."""
        m, r, B = self.model, self._rec, self._body
        b = B["body"]
        core = (1 - PAD_MASS_FRACTION) * mass if r["soft"] else mass
        m.body_mass[b] = core
        m.body_inertia[b] = r["inertia1"] * core
        m.body_ipos[b] = r["ipos"]
        m.body_iquat[b] = r["iquat"]
        if r["soft"]:
            pads = B["pads"]
            pad_mass = PAD_MASS_FRACTION * mass / r["n"]
            m.body_mass[pads] = pad_mass
            m.body_inertia[pads] = pad_mass * PAD_T ** 2
            m.jnt_stiffness[B["pad_j"]] = stiffness
            m.dof_damping[B["pad_v"]] = 2 * PAD_DAMPING_RATIO * np.sqrt(stiffness * (PAD_ARMATURE + pad_mass))
            soft = np.clip(np.log(stiffness / 200.0) / np.log(20.0), 0, 1)      # colour: soft = pale
            m.geom_rgba[B["pad_g"], :3] = (1 - soft) * np.array([1.0, 0.85, 0.7]) + soft * np.array([0.85, 0.4, 0.15])
        m.geom_friction[r["geoms"], 0] = friction
        mujoco.mj_setConst(m, self._ik_data)              # (it overwrites the data it is given)

    def _object_state(self):
        """centre, velocity (world), bottom height above the table, width along the robot's
        y axis, strain (0 = undeformed, 1 = a pad pushed in all the way), rotation (3x3)."""
        d, m, r, B = self.data, self.model, self._rec, self._body
        _, yaw = self._base_pose()
        ey = _rot_z(yaw + self._yaw_cmd)[:, 1]                # along the grasp axis
        b = B["body"]
        centre = d.xpos[b].copy()
        vel = d.qvel[B["vadr"]:B["vadr"] + 3].copy()
        R = d.xmat[b].reshape(3, 3).copy()
        if r["soft"]:
            pads = B["pads"][:r["n"]]
            pts = d.xpos[pads] + PAD_T * d.xmat[pads].reshape(-1, 3, 3)[:, :, 2]
            strain = float(np.clip(-d.qpos[B["pad_q"][:r["n"]]].min() / PAD_TRAVEL, 0, 1))
        else:
            g = r["geom"]
            Rg = d.geom_xmat[g].reshape(3, 3)
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
            pts = d.geom_xpos[g] + (c + signs * h) @ Rg.T
            strain = 0.0
        return centre, vel, pts[:, 2].min() - self.TABLE_TOP, float(np.ptp(pts @ ey)), strain, R

    def _contacts(self):
        """Normal force of each hand on the object, hands' force on the table. Also sets
        self._bump: the robot's body (not its hands) is touching the table."""
        d = self.data
        n = d.ncon
        hand = np.zeros(2)
        table = 0.0
        self._bump = False
        if n == 0:
            return hand, table
        k = self._rec["index"]
        g = d.contact.geom[:n]
        k0 = np.where(g[:, 0] >= 0, self._hand_geom[np.maximum(g[:, 0], 0)], -1)
        k1 = np.where(g[:, 1] >= 0, self._hand_geom[np.maximum(g[:, 1], 0)], -1)
        obj0 = (g[:, 0] >= 0) & (self._geom_obj[np.maximum(g[:, 0], 0)] == k)
        obj1 = (g[:, 1] >= 0) & (self._geom_obj[np.maximum(g[:, 1], 0)] == k)
        side = np.where(obj1 & (k0 >= 0), k0, np.where(obj0 & (k1 >= 0), k1, -1))
        tab0 = (g[:, 0] >= 0) & self._is_table[np.maximum(g[:, 0], 0)]
        tab1 = (g[:, 1] >= 0) & self._is_table[np.maximum(g[:, 1], 0)]
        on_table = (tab0 & (k1 >= 0)) | (tab1 & (k0 >= 0))
        r0 = (g[:, 0] >= 0) & self._robot_geom[np.maximum(g[:, 0], 0)]
        r1 = (g[:, 1] >= 0) & self._robot_geom[np.maximum(g[:, 1], 0)]
        self._bump = bool(np.any((tab0 & r1) | (tab1 & r0)))
        idx = np.flatnonzero((side >= 0) | on_table)
        if idx.size == 0:
            return hand, table
        adr = d.contact.efc_address[idx]
        fn = np.where(adr >= 0, np.abs(d.efc_force[np.maximum(adr, 0)]), 0.0)
        s = side[idx]
        np.add.at(hand, s[s >= 0], fn[s >= 0])
        table = float(fn[on_table[idx]].sum())
        return hand, table

    # ================================================================ observation
    def _obs_parts(self):
        tips = self._tips_local()
        lag = np.concatenate([(self._cmd[s] - tips[s]) / self.LEASH for s in SIDES])
        R = _rot_z(self._base_pose()[1])
        vb = R.T @ np.array([*self.data.qvel[self._base_v[:2]], 0.0])
        tgt = self.to_local(self._target)
        parts = {"proprio": np.concatenate([
            tips["left"], tips["right"], np.clip(lag, -3, 3), self._vel_cmd,
            [vb[0], self.data.qvel[self._base_v[2]]], tgt, [self._yaw_cmd, float(self._was_lifted)]])}
        if self.obs_modes["object"] or self.obs_modes["privileged"]:
            # what a camera-based object tracker would give: pose, motion, size, shape
            centre, vel, h, _, strain, Ro = self._obj
            hc = (tips["left"] + tips["right"]) / 2
            parts["object"] = np.concatenate([
                self.to_local(centre) - hc, R.T @ (centre - self._target), R.T @ vel,
                (R.T @ Ro)[:, :2].T.ravel(), self._rec["size"] * 10,
                self._shape_onehot[self.obj], [h * 10]])
        if self.obs_modes["privileged"]:
            # what touch has to work out: how stiff, heavy and slippery it is, how hard it is
            # gripped and how squashed
            logk = np.log10(self._props["stiffness"]) if self._rec["soft"] else RIGID_LOG_STIFFNESS
            parts["privileged"] = np.concatenate([
                [logk - 3.0, self._props["mass"], self._props["friction"], strain],
                self._hand_force / 10.0])
        if self.obs_modes["tactile"]:
            parts["tactile"] = self._tactile_obs()
        if self.obs_modes["vision"]:
            parts["vision"] = self.get_vision()
        return parts

    def _get_obs(self):
        return self._format_obs(self._obs_parts())

    # ================================================================ reset
    def _lerp(self, pair):
        (a0, b0), (a1, b1) = pair
        lv = self.level
        return a0 + lv * (a1 - a0), b0 + lv * (b1 - b0)

    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if "level" in options:
            self.level = float(np.clip(options["level"], 0, 1))
        elif self.curriculum and self._last_outcome is not None:
            if self._last_outcome == "success":
                self.level = min(1.0, self.level + self.LEVEL_UP)
            else:                                         # failed or ran out of time
                self.level = max(0.0, self.level - self.LEVEL_DOWN)
        self._last_outcome = None
        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)

        # robot: base a little back from its spot, torso / head posed, arms at the start pose
        back = rng.uniform(*self._lerp(self.BASE_BACK))
        d.qpos[self._base_q] = [-back, 0.0, rng.uniform(-0.05, 0.05)]
        d.qpos[self._torso_q] = self.TORSO_POSE
        d.qpos[self._head_q] = self.HEAD_POSE
        d.ctrl[self._torso_act] = self.TORSO_POSE
        d.ctrl[self._head_act] = self.HEAD_POSE
        self._centre_cmd = self.START_CENTRE.copy()
        self._width_cmd = self.START_WIDTH
        self._yaw_cmd = 0.0
        self._cmd = self._hand_targets(self._centre_cmd, self._width_cmd)
        if not hasattr(self, "_q_start"):
            self._q_start = self._start_pose(d)
        self._q_nominal = self._q_start
        self._q_cmd = {s: self._q_start[s].copy() for s in SIDES}
        for s in SIDES:
            d.qpos[self._arm_q[s]] = self._q_cmd[s]
            d.ctrl[self._arm_act[s]] = self._q_cmd[s]

        # object and its properties
        self.obj = options.get("object") or self.object_names[rng.integers(len(self.object_names))]
        soft = self._rec["soft"]
        lo, hi = self._lerp(self.MASS)
        mass = float(options.get("mass", np.exp(rng.uniform(np.log(lo), np.log(hi)))))
        friction = float(options.get("friction", rng.uniform(*self._lerp(self.FRICTION))))
        lo, hi = self._lerp(self.STIFFNESS)
        lo = min(hi, max(lo, self.STIFF_PER_KG * mass / min(friction, 1.0)))
        stiffness = float(options.get("stiffness", np.exp(rng.uniform(np.log(lo), np.log(hi))))) if soft else np.inf
        self._props = dict(stiffness=stiffness, mass=mass, friction=friction)
        self._configure_object()
        self._set_properties(stiffness, mass, friction)
        start = np.array([rng.uniform(*self.OBJECT_X), rng.uniform(*self.OBJECT_Y), 0.0])
        xy = start[:2]                                  # world frame: the table doesn't move
        n = SYMMETRY[self.obj]                          # yaw: the grasp can always square up
        yaw = (rng.uniform(-np.pi, np.pi) if n == 0
               else rng.uniform(*self._lerp(self.YAW)) + 2 * np.pi / n * rng.integers(n))
        self._place_object(xy, float(options.get("yaw", yaw)))
        self._park_all()

        # target: on the table, a set distance from the object, within reach of the robot's
        # spot; at level 0 nearly straight ahead, at higher levels off to the side (the base
        # has to turn to face it)
        bearing = self.TARGET_BEARING[0] + self.level * (self.TARGET_BEARING[1] - self.TARGET_BEARING[0])
        t, best = None, np.inf
        for _ in range(300):
            dist = rng.uniform(*self._lerp(self.TARGET_DIST))
            ang = rng.uniform(-np.pi, np.pi)
            c = xy + dist * np.array([np.cos(ang), np.sin(ang)])
            on_table = (self.TABLE_X[0] + 0.08 < c[0] < self.TABLE_X[1] - 0.08
                        and abs(c[1]) < self.TABLE_HALF_WIDTH - 0.08)
            r, b = np.hypot(*c), abs(np.arctan2(c[1], c[0]))
            miss = max(0.0, self.REACH_R[0] - r, r - self.REACH_R[1]) + max(0.0, b - bearing)
            if on_table and miss < best:                # closest to reachable, if none is
                t, best = c, miss
            if on_table and miss == 0.0 and np.linalg.norm(c - xy) > 0.06:
                break
        self._target = np.array([t[0], t[1], self.TABLE_TOP])
        m.site_pos[self._target_site] = [t[0], t[1], self.TABLE_TOP + 0.001]

        mujoco.mj_forward(m, d)
        for _ in range(100):                            # settle the object on the table
            mujoco.mj_step(m, d)
            self._park_all()
        mujoco.mj_forward(m, d)

        self.step_count = 0
        self._vel_cmd = np.zeros(self.action_space.shape[0])
        self._base_cmd = np.zeros(2)
        self._last_action = np.zeros_like(self._vel_cmd)
        self._was_lifted = False
        self._release = 0
        self._crush = 0
        self._peak_strain = 0.0
        self._hand_force, self._table_force = self._contacts()
        self._obj = self._object_state()
        self._target_dist0 = max(np.linalg.norm(self._obj[0][:2] - self._target[:2]), 1e-3)
        self._last_obj_centre = self._obj[0].copy()
        self._last_hand_centre = sum(self.data.site_xpos[self._tip[s]] for s in SIDES) / 2
        self._phi = self._potential()
        self._ep_terms = {}
        self._fail_reason = ""
        self._reset_marker_history()
        return self._get_obs(), self._info()

    # ================================================================ control
    def _hand_targets(self, centre, width, yaw=None):
        yaw = self._yaw_cmd if yaw is None else yaw
        off = _rot_z(yaw) @ np.array([0.0, width / 2, 0.0])
        return {"left": centre + off, "right": centre - off}

    def _apply_action(self, a):
        k = self.ACTION_SMOOTHING
        self._vel_cmd = k * self._vel_cmd + (1 - k) * a
        v = self._vel_cmd
        self._yaw_cmd = float(np.clip(self._yaw_cmd + v[-1] * self.GRASP_YAW_SPEED,
                                      -self.GRASP_YAW_MAX, self.GRASP_YAW_MAX))
        tips = self._tips_local()
        if self.action_mode == "bimanual":
            # moving together and squeezing are leashed separately, so carrying an object
            # sideways never loosens the grip
            mc = (tips["left"] + tips["right"]) / 2
            mw = float((tips["left"] - tips["right"]) @ _rot_z(self._yaw_cmd)[:, 1])
            c = np.clip(self._centre_cmd + v[2:5] * self.CENTRE_SPEED, self.WS_LO, self.WS_HI)
            lead = c - mc
            n = np.linalg.norm(lead)
            if n > self.LEASH:
                c = mc + lead * (self.LEASH / n)
            w = self._width_cmd + v[5] * self.WIDTH_SPEED
            w = float(np.clip(np.clip(w, mw - self.SQUEEZE_MAX, mw + self.LEASH), *self.WIDTH_RANGE))
            self._centre_cmd, self._width_cmd = c, w
            cmd = self._hand_targets(c, w)
        else:
            cmd = {s: np.clip(self._cmd[s] + v[2 + 3 * i:5 + 3 * i] * self.CENTRE_SPEED, self.WS_LO, self.WS_HI)
                   for i, s in enumerate(SIDES)}
            for s in SIDES:                              # leash: never run away from the tip
                lead = cmd[s] - tips[s]
                n = np.linalg.norm(lead)
                if n > self.LEASH:
                    cmd[s] = tips[s] + lead * (self.LEASH / n)
        self._cmd = cmd
        q_new = self._solve_arms(cmd)

        # base: diff-drive -> world-frame velocities for the planar joints, accelerating at
        # most BASE_ACCEL / TURN_ACCEL (a jolt would shake a held object loose)
        step = np.array([self.BASE_ACCEL, self.TURN_ACCEL]) * self.dt
        want = np.array([v[0] * self.BASE_SPEED, v[1] * self.TURN_RATE])
        self._base_cmd = self._base_cmd + np.clip(want - self._base_cmd, -step, step)
        fwd, turn = self._base_cmd
        yaw = self._base_pose()[1]
        base_ctrl = np.array([fwd * np.cos(yaw), fwd * np.sin(yaw), turn])
        wheel_r, track = 0.1, 0.53
        wheels = np.array([(fwd - turn * track / 2) / wheel_r, (fwd + turn * track / 2) / wheel_r])

        m, d = self.model, self.data
        q_old = self._q_cmd
        n_sub = self.FRAME_SKIP // self.CTRL_CHUNKS
        for c in range(self.CTRL_CHUNKS):
            w = (c + 1) / self.CTRL_CHUNKS
            for s in SIDES:
                d.ctrl[self._arm_act[s]] = q_old[s] + w * (q_new[s] - q_old[s])
            d.ctrl[self._base_act] = base_ctrl
            d.ctrl[self._wheels] = np.clip(wheels, -3.14, 3.14)
            mujoco.mj_step(m, d, n_sub)
        self._q_cmd = q_new

    # ================================================================ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        self._apply_action(a)
        self._park_all()
        mujoco.mj_forward(self.model, self.data)
        self.step_count += 1
        self._update_marker_history()

        self._hand_force, self._table_force = self._contacts()
        self._obj = self._object_state()
        self._peak_strain = max(self._peak_strain, self._obj[4])
        reward, terms = self._reward(a)
        for k, v in terms.items():
            self._ep_terms[k] = self._ep_terms.get(k, 0.0) + v
        status = self._status()
        terminated = status is not None
        truncated = not terminated and self.step_count >= self.max_steps
        if not np.all(np.isfinite(self.data.qpos)):
            terminated, status, self._fail_reason = True, "fail", "simulation unstable"
            mujoco.mj_resetData(self.model, self.data)
        if status == "success":
            reward += self.R_SUCCESS
        elif status == "fail":
            reward += self.R_FAIL
        if terminated or truncated:
            self._last_outcome = status or "timeout"
        self._last_action = a
        return self._get_obs(), float(reward), terminated, truncated, self._info(status, truncated)

    def _info(self, status=None, truncated=False):
        info = {"is_success": status == "success",
                "fail_reason": self._fail_reason if status == "fail" else "",
                "object": self.obj, "level": self.level, **{k: float(v) for k, v in self._props.items()}}
        if status is not None or truncated:
            info["reward_terms"] = dict(self._ep_terms)
            info["peak_squash"] = self._peak_strain
        return info

    # ================================================================ reward
    def _gripped(self):
        return bool(np.all(self._hand_force > self.GRIP_FORCE))

    def _potential(self):
        centre, _, h, _, _, _ = self._obj
        hand_c = sum(self.data.site_xpos[self._tip[s]] for s in SIDES) / 2
        grip = self._gripped()
        reach = _close(np.linalg.norm(hand_c - centre), 0.10)
        lift = grip * np.clip(h / self.LIFT_HEIGHT, 0, 1)
        d_tgt = np.linalg.norm(centre[:2] - self._target[:2])
        near = d_tgt < self.PLACE_RADIUS
        carry = self._was_lifted * (grip or near) * (1 - np.clip(d_tgt / self._target_dist0, 0, 1))
        place = self._was_lifted * near * (1 - np.clip(h / self.LIFT_HEIGHT, 0, 1))
        return 0.10 * reach + 0.15 * grip + 0.25 * lift + 0.35 * carry + 0.15 * place

    def _reward(self, a):
        centre, _, h, _, strain, _ = self._obj
        grip = self._gripped()
        if grip and h > self.LIFT_MIN:
            self._was_lifted = True
        phi = self._potential()
        progress = 20.0 * (phi - self._phi)
        self._phi = phi
        hand_c = sum(self.data.site_xpos[self._tip[s]] for s in SIDES) / 2
        slip = np.linalg.norm((centre - self._last_obj_centre) - (hand_c - self._last_hand_centre)) / self.dt
        self._last_obj_centre, self._last_hand_centre = centre.copy(), hand_c
        terms = dict(
            progress=progress,
            state=0.02 * phi,
            crush=-1.0 * max(0.0, strain - self.CRUSH_STRAIN),
            slip=-0.5 * min(1.0, slip / 0.1) * (grip and h > self.LIFT_MIN),
            table=-0.1 * (self._table_force > 20.0),
            bump=-0.2 * self._bump,                 # torso / arms against the table
            action_rate=-0.01 * float(np.sum((a - self._last_action) ** 2)),
        )
        return sum(terms.values()), terms

    def _status(self):
        centre, vel, h, _, strain, _ = self._obj
        if centre[2] < self.TABLE_TOP - 0.10:
            self._fail_reason = "object fell off the table"
            return "fail"
        self._crush = self._crush + 1 if strain > self.CRUSH_FAIL else 0
        if self._crush >= self.CRUSH_STEPS:
            self._fail_reason = "object crushed"
            return "fail"
        d_tgt = np.linalg.norm(centre[:2] - self._target[:2])
        resting = (self._was_lifted and d_tgt < self.PLACE_RADIUS and h < 0.01
                   and np.all(self._hand_force < 0.1) and np.linalg.norm(vel) < 0.05)
        self._release = self._release + 1 if resting else 0
        return "success" if self._release >= self.RELEASE_STEPS else None

    # ================================================================ rendering
    def render(self):
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._view_cam = mujoco.MjvCamera()
            self._view_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._view_cam.trackbodyid = self.model.body(self.TRACK_BODY).id
            self._view_cam.distance = self.TRACK_DISTANCE
            self._view_cam.azimuth, self._view_cam.elevation = 150, self.TRACK_ELEVATION
        self._view_renderer.update_scene(self.data, camera=self._view_cam)
        return self._view_renderer.render().copy()


# ==================================================================== scripted teacher
class ScriptedPicker:
    """Bimanual squeeze grasp written by hand, using the simulator's state (any observation
    mode): drive up if the object is out of reach, square up to its faces, approach, close
    until the grip force suits its weight and friction, lift, carry (driving if the target is
    out of reach), lower, release. For checks, demos and warm-starting a learned policy.

        picker = ScriptedPicker(env)          # after env.reset(); make a new one per episode
        obs, r, term, trunc, info = env.step(picker())
    """
    LIFT = 0.08                 # m above the grasp height
    SWEET_X = 0.58              # m in front of the base: where the hands work best

    def __init__(self, env, safety=1.8):
        self.env = env.unwrapped
        self.safety = safety
        self.phase = "align"
        self.t = 0
        self.offset = None
        self.lost = 0
        self.yaw_goal = 0.0

    def _align_yaw(self):
        """Grasp yaw (relative to the base) that squares the hands to the object's faces."""
        e = self.env
        n = SYMMETRY[e.obj]
        if n == 0:
            return 0.0
        R = e._obj[5]
        period = 2 * np.pi / n
        err = (np.arctan2(R[1, 0], R[0, 0]) - e._base_pose()[1] + period / 2) % period - period / 2
        return float(np.clip(err, -e.GRASP_YAW_MAX, e.GRASP_YAW_MAX))

    def _drive(self, goal, turn=True):
        """(forward, turn) in action units bringing goal (robot frame) in front of the hands."""
        e = self.env
        fwd, rot = 0.0, 0.0
        dx = goal[0] - self.SWEET_X
        if abs(dx) > 0.06:
            fwd = float(np.clip(dx / 0.15, -1, 1))
            if fwd > 0 and e._base_pose()[0][0] > -0.01:      # at its spot: the table is close
                fwd = 0.0
        if turn and abs(goal[1]) > 0.06:
            rot = float(np.clip(np.arctan2(goal[1], max(goal[0], 0.2)) / 0.4, -1, 1))
        return fwd, rot

    def __call__(self):
        e = self.env
        centre, _, h, width, _, _ = e._obj
        c = e.to_local(centre)
        tgt = e.to_local(e._target)
        axis = _rot_z(e._yaw_cmd)[:, 1]                     # grasp axis, robot frame
        cmd_c = (e._cmd["left"] + e._cmd["right"]) / 2
        cmd_w = float((e._cmd["left"] - e._cmd["right"]) @ axis)
        f = e._hand_force
        mass, mu = e._props["mass"], e._props["friction"]
        need = self.safety * mass * 9.81 / (2 * mu) + e.GRIP_FORCE
        half_h = e._objects[e.obj]["size"][2] / 2
        grasp_z = e.TABLE_TOP + min(half_h, 0.035)
        goal_c, goal_w = cmd_c.copy(), cmd_w
        self.t += 1
        fwd = turn = 0.0
        if self.phase == "align":                          # square the hands up to its faces
            self.yaw_goal = self._align_yaw()
            fwd, _ = self._drive(c, turn=False)
            goal_c = np.array([cmd_c[0], cmd_c[1], e.TABLE_TOP + 0.10])
            if (abs(self.yaw_goal - e._yaw_cmd) < 0.02 and fwd == 0.0) or self.t > 200:
                self.phase = "approach"
        elif self.phase == "approach":                     # above, open
            fwd, _ = self._drive(c, turn=False)
            goal_c = np.array([c[0], c[1], e.TABLE_TOP + 0.10])
            goal_w = width + 0.08
            if np.linalg.norm(goal_c - cmd_c) < 0.01 and abs(goal_w - cmd_w) < 0.01 and fwd == 0.0:
                self.phase = "descend"
        elif self.phase == "descend":
            goal_c = np.array([c[0], c[1], grasp_z])
            goal_w = width + 0.08
            if np.linalg.norm(goal_c - cmd_c) < 0.008:
                self.phase = "close"
        elif self.phase == "close":
            goal_c = np.array([c[0], c[1], grasp_z])
            goal_w = cmd_w - 0.004 * np.clip((need - f.min()) / 5.0, -1, 1)
            if f.min() > need:
                self.phase, self.t = "lift", 0
        elif self.phase in ("lift", "carry", "lower"):
            squeeze = -0.002 * np.clip((need - f.min()) / 5.0, 0, 1)    # tighten if it slips
            goal_w = cmd_w + squeeze
            self.lost = self.lost + 1 if f.max() < 0.05 else 0
            if self.lost > 10:                                # dropped it: start again
                self.phase, self.t, self.offset, self.lost = "align", 0, None, 0
            elif self.phase == "lift":
                goal_c = np.array([cmd_c[0], cmd_c[1], grasp_z + self.LIFT])
                if cmd_c[2] > grasp_z + self.LIFT - 0.01:
                    self.phase = "carry"
            elif self.phase == "carry":
                if self.offset is None:                      # hands relative to the object
                    self.offset = cmd_c[:2] - c[:2]
                fwd, turn = self._drive(tgt)
                goal_c = np.array([*(tgt[:2] + self.offset), grasp_z + self.LIFT])
                if np.linalg.norm(centre[:2] - e._target[:2]) < 0.015:
                    self.phase = "lower"
            else:
                goal_c = np.array([cmd_c[0], cmd_c[1], grasp_z + 0.005])
                if h < 0.004:
                    self.phase = "release"
        elif self.phase == "release":
            goal_w = width + 0.10
            if f.max() < 0.05 and cmd_w > width + 0.06:
                self.phase = "retreat"
        else:                                                 # retreat upwards
            goal_c = np.array([cmd_c[0], cmd_c[1], e.TABLE_TOP + 0.15])
        gain = 0.5 if self.phase == "carry" else 1.0         # carry gently
        a_c = np.clip(gain * (goal_c - cmd_c) / e.CENTRE_SPEED, -gain, gain)
        a_w = float(np.clip((goal_w - cmd_w) / e.WIDTH_SPEED, -1, 1))
        a_y = float(np.clip((self.yaw_goal - e._yaw_cmd) / e.GRASP_YAW_SPEED, -1, 1))
        if e.action_mode == "bimanual":
            a = np.array([fwd, turn, *a_c, a_w, a_y])
        else:                                                 # same motion, per hand
            off = axis * (a_w * e.WIDTH_SPEED / 2) / e.CENTRE_SPEED
            a = np.array([fwd, turn, *(a_c + off), *(a_c - off), a_y])
        return np.clip(a, -1, 1)
