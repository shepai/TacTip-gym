"""
TacQuad: a Unitree Go2 (MuJoCo Menagerie) with a TacTip on every foot, learning to walk
stably on ground whose feel changes from tile to tile: friction, softness, bounce and bumps.

    env = gym.make("tactip_mujoco_gym:TacQuad-walk", obs_modes={"tactile": True})

The model is built in code (MjSpec) every time the env is made, so go2.xml itself is never
edited: the Go2 (assets/go2/go2.xml, or any go2.xml found under assets/), its foot spheres
replaced by four prefixed copies of assets/sensor/tactip.xml (FL_, FR_, RL_, RR_; see
sensor_tools.py), position-controlled joints, a head camera and a grid of heightfield tiles.
Works with the MuJoCo Menagerie Go2 and the one in Unitree's unitree_mujoco repo.

See the robot with its TacTip feet, or write it out as a standalone XML:
    python -m tactip_mujoco_gym.envs.tactip_quadruped --view
    python -m tactip_mujoco_gym.envs.tactip_quadruped --save go2_tactip.xml

Observation inputs (obs_modes), as in the arm tasks; proprioception is always included:
    proprio     base angular velocity, gravity in the base frame, velocity command,
                joint positions (minus the standing pose), joint velocities, previous action
    privileged  base linear velocity, base height, foot contact forces, friction and softness
                under each foot, foot heights, a grid of terrain heights around the robot
    tactile     marker displacements of all four TacTips (mm) and their change since last step
    vision      the head camera (VISION_CAMERA = "headcam")

Action: 12 joint targets, a in [-1, 1] -> standing pose + ACTION_SCALE * a, PD at KP / KD.
Task:   follow a velocity command (forward, sideways, turn), resampled every few seconds;
        episodes end when the trunk touches the ground or the robot tips past 60 degrees.
Terrain: N_TILES x N_TILES tiles of TILE metres. Every reset draws, per tile, a friction, a
        softness (contact time constant + impedance) and a damping ratio, and decides whether
        the tile is flat or rough (bumps and mounds). `terrain_level` (0..1) scales how
        extreme they get, and with curriculum=True it rises when the robot copes and falls
        when it doesn't. Tile colour shows the feel: blue = slippery, red-brown = grippy,
        paler = softer. Flat tiles are boxes and rough tiles heightfields (the TacTip pins
        collide with a box several times faster).

Feet:   "tactip" four TacTips (slow: ~150 bodies of sensor); "proxy" one sphere per foot
        with the TacTip's size, reach and mass (same leg geometry, much faster: train a
        privileged teacher on it); "rigid" the Go2's own feet.
"""
from importlib.resources import files

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from tactip_mujoco_gym.envs.base import TactileGymEnv
from tactip_mujoco_gym.sensor_tools import attach_sensor, extract_sensor, mount_pose, sensor_info

LEGS = ("FL", "FR", "RL", "RR")
JOINTS = ("hip", "thigh", "calf")


def _asset(*parts):
    return str(files("tactip_mujoco_gym").joinpath(*parts))


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


class TacQuadruped(TactileGymEnv):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    # ---------------------------------------------------------------- robot + sensor
    ROBOT_XML     = ("assets", "go2.xml")   # else the first go2.xml found under assets/
    SENSOR_XML    = ("assets", "sensor", "tactip.xml")     # made by sensor_tools.py extract
    SENSOR_SOURCE = ("assets", "tactip_arm.xml")           # used if SENSOR_XML doesn't exist
    SENSOR_ROOT   = "auto"        # sensor in SENSOR_SOURCE: found from its markers, or a body name
    SENSOR_AXIS   = None          # tip direction in the sensor's root frame; None = from markers
    FOOT_CENTRE   = (-0.002, 0.0, -0.213)                  # Go2 foot sphere, calf frame
    FOOT_RADIUS   = 0.022
    LEG_EXTENSION = 0.0           # m; > 0 puts the TacTip tip below the original foot
    TIP_DIRECTION = "down"        # "down": tip points straight down when standing (the shins
                                  # slope ~50 deg, so the sensor sits at an angle on them);
                                  # "shin": tip points along the shin
    SENSOR_SPIN   = 0.0           # rad: turn each sensor about its tip axis
    TRACK_BODY, TRACK_DISTANCE, TRACK_ELEVATION = "base", 1.8, -20    # video camera
                                  # (TRACK_BODY is set to the trunk the model actually has)
    VISION_CAMERA = "headcam"     # added on the trunk, looking ahead and 30 degrees down

    DEFAULT_POSE  = np.array([0.0, 0.9, -1.8] * 4)        # standing joint angles (rad)
    ACTION_SCALE  = np.array([0.35, 0.6, 0.6] * 4)         # rad per unit action
    KP, KD        = 30.0, 0.8                              # joint PD gains (Nm/rad, Nm s/rad)
    TIMESTEP      = 0.002
    FRAME_SKIP    = 10                                     # physics steps per action: 50 Hz
    EPISODE_SECONDS = 20.0

    # ---------------------------------------------------------------- terrain
    N_TILES      = 8
    TILE         = 2.0            # m
    TILE_RES     = 40             # heightfield cells per tile side (5 cm)
    HF_RANGE     = 0.3            # m, heightfield elevation range
    MAX_BUMPS    = 0.05           # m, peak-to-peak small bumps at level 1
    MAX_WAVES    = 0.10           # m, height of mounds / dips at level 1
    ROUGH_FRACTION = (0.0, 0.6)   # share of rough tiles at level 0 / level 1
    EDGE_RAMP    = 0.3            # m: rough tiles blend to flat over this near their edges
    FRICTION     = ((0.6, 1.2), (0.15, 1.3))    # (low, high) at level 0 and at level 1
    SOFTNESS_MAX = (0.0, 1.0)                    # max softness at level 0 / level 1
    ROUGH_SOFTNESS_MAX = 0.3      # rough (heightfield) tiles stay fairly hard: a foot sinking
                                  # more than ~half a 5 cm cell can be pushed out sideways
    DAMPRATIO    = ((0.8, 1.2), (0.3, 1.5))
    TIMECONST    = (0.006, 0.05)                 # s: hard .. soft
    SOLIMP_DMIN  = (0.9, 0.3)
    SOLIMP_WIDTH = (0.001, 0.03)                 # m
    SPAWN_FLAT   = 0.6            # m: tiles within this of the spawn point are always flat
    LEVEL_STEP   = 0.05

    # ---------------------------------------------------------------- commands
    CMD_RANGES   = ((-0.5, 1.0), (-0.4, 0.4), (-1.0, 1.0))   # vx, vy (m/s), yaw rate (rad/s)
    CMD_SECONDS  = (4.0, 8.0)     # resample interval
    CMD_ZERO_PROB = 0.1           # chance of a "stand still" command

    # ---------------------------------------------------------------- reward
    REWARD_WEIGHTS = dict(
        tracking_lin_vel=1.0, tracking_ang_vel=0.5,
        lin_vel_z=-2.0, ang_vel_xy=-0.05, orientation=-2.0, base_height=-10.0,
        torques=-2e-4, dof_acc=-2.5e-7, action_rate=-0.01, collision=-1.0,
        feet_air_time=1.0, dof_pos_limits=-10.0, feet_slip=-0.05, stand_still=-0.5,
    )
    TRACKING_SIGMA = 0.25
    AIR_TIME_TARGET = 0.25        # s
    ONLY_POSITIVE_REWARDS = True  # clip each step's reward at 0, as in legged_gym
    CONTACT_FORCE = 2.0           # N: a foot counts as in contact above this
    TILT_LIMIT = -0.5             # gravity z in the base frame; -0.5 = 60 degrees of tilt
    BASE_HEIGHT_OFFSET = -0.04    # m: target base height = straight standing height + this
                                  # (the PD-held standing pose sags ~5 cm under the robot's weight)
    HEIGHT_GRID = (np.arange(-0.3, 0.51, 0.1), np.arange(-0.2, 0.21, 0.1))   # privileged scan

    def __init__(self, feet="tactip", sensor_stiffness=10.0, terrain_level=None, curriculum=True,
                 command=None, obs_modes=None, privileged=None, obs_format="flat",
                 render_mode=None, max_steps=None, sensor_xml=None, robot_xml=None):
        """
        feet              "tactip" (four TacTips), "proxy" (a sphere matching the TacTip's
                          size: same legs, ~3x faster, no tactile input) or "rigid" (the Go2's
                          own feet)
        sensor_stiffness  multiplies the TacTip's joint / tendon stiffness, damping and armature
                          (see sensor_tools.attach_sensor). A skin tuned for finger pressure
                          bottoms out under a 15 kg robot; check_quadruped.py reports how much
                          of the marker travel is used when standing.
        terrain_level     0..1 fixes the terrain difficulty; None starts at 0 and, with
                          curriculum=True, adapts per environment instance
        command           None = random commands; (vx, vy, yaw_rate) = always this one

        reset(options=...) accepts "terrain_level" (0..1, this episode on) and "feel", a dict
        with friction / softness (0..1) / dampratio applied to every tile, e.g. ice:
            env.reset(options={"feel": {"friction": 0.15}})
        """
        gym.Env.__init__(self)
        if feet not in ("tactip", "proxy", "rigid"):
            raise ValueError(f"feet must be 'tactip', 'proxy' or 'rigid', not {feet!r}")
        self.feet = feet
        self.sensor_stiffness = float(sensor_stiffness)
        self.curriculum = curriculum and terrain_level is None
        self.level = 0.0 if terrain_level is None else float(np.clip(terrain_level, 0, 1))
        self.fixed_command = None if command is None else np.asarray(command, float)
        self.render_mode = render_mode
        self.xml_path = None

        spec = self._build_spec(sensor_xml, robot_xml)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index_model()

        # base-class state (renderers, observation selection)
        self._physics = self._renderer = self._view_renderer = self._vision_renderer = None
        self.step_count = 0
        self.dt = self.TIMESTEP * self.FRAME_SKIP
        self.max_steps = max_steps or int(round(self.EPISODE_SECONDS / self.dt))
        self.events = False
        self.action_space = spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
        self._terrain_version = 0
        self._last_outcome = None
        self.obs_format = "flat"
        self.obs_modes = {"vision": False, "tactile": False, "privileged": False}
        self.reset(seed=0)                       # state for the first observation
        self.set_obs_format(obs_format)
        self.set_obs_modes(self.resolve_obs_modes(obs_modes, privileged))   # checks availability

    # ================================================================ model building
    def _sensor_spec(self, sensor_xml):
        path = sensor_xml or _asset(*self.SENSOR_XML)
        try:
            return mujoco.MjSpec.from_file(path)
        except ValueError:
            src = _asset(*self.SENSOR_SOURCE)
            import os
            if sensor_xml is not None or not os.path.exists(src):
                raise
            print(f"[TacQuadruped] {path} not found: using the sensor in {src} "
                  f"(run `python -m tactip_mujoco_gym.sensor_tools extract ...` to make the file)")
            return extract_sensor(src, self.SENSOR_ROOT)

    @classmethod
    def find_robot(cls, robot_xml=None):
        """robot_xml if given, else assets/go2/go2.xml, else the first go2.xml under assets/."""
        import os
        if robot_xml is not None:
            return str(robot_xml)
        path = _asset(*cls.ROBOT_XML)
        if os.path.exists(path):
            return path
        found = sorted(files("tactip_mujoco_gym").joinpath("assets").rglob("go2.xml"))
        if found:
            return str(found[0])
        raise FileNotFoundError(
            f"No go2.xml under {_asset('assets')}. Put the Go2 folder there (go2.xml plus its "
            "assets/ mesh folder), run `python tools/fetch_go2.py`, or pass robot_xml=...")

    def _build_spec(self, sensor_xml=None, robot_xml=None, terrain=True):
        path = self.find_robot(robot_xml)
        try:
            spec = mujoco.MjSpec.from_file(path)
        except ValueError as e:
            raise FileNotFoundError(f"Can't load the robot model {path} ({e}). Download the "
                                    "Unitree Go2 with `python tools/fetch_go2.py`.") from None
        self.robot_xml = path
        # a scene.xml brings its own floor / boxes: the env supplies the ground
        for g in list(spec.worldbody.geoms):
            spec.delete(g)
        trunks = [b for b in spec.bodies
                  if any(j.type == mujoco.mjtJoint.mjJNT_FREE for j in b.joints)]
        if len(trunks) != 1:
            raise ValueError(f"expected one free-floating body (the trunk) in {path}, "
                             f"found {[b.name for b in trunks]}")
        self.trunk_name = trunks[0].name                   # "base" (Menagerie), "base_link" (Unitree)
        spec.modelname = "tacquad"
        spec.option.timestep = self.TIMESTEP
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 640, 480

        # ---- feet: the tip goes where the old foot touched the ground when standing
        if self.TIP_DIRECTION not in ("down", "shin"):
            raise ValueError(f"TIP_DIRECTION must be 'down' or 'shin', not {self.TIP_DIRECTION!r}")
        down = self._standing_down(spec)
        ext = self.LEG_EXTENSION if self.feet != "rigid" else 0.0
        self.foot_bottom = {leg: np.array(self.FOOT_CENTRE) + (self.FOOT_RADIUS + ext) * down[leg]
                            for leg in LEGS}
        self.foot_info = None
        if self.feet in ("tactip", "proxy"):
            sensor = self._sensor_spec(sensor_xml)
            info = sensor_info(sensor, axis=self.SENSOR_AXIS)
            self.foot_info = info
            for leg in LEGS:
                tip_dir = down[leg] if self.TIP_DIRECTION == "down" else np.array([0.0, 0, -1])
                bottom = self.foot_bottom[leg]
                calf = spec.body(f"{leg}_calf")
                for g in list(calf.geoms):                   # old foot: collision + visual mesh
                    if g.name == leg or g.meshname == "foot":
                        spec.delete(g)
                if self.feet == "tactip":
                    mount, quat = mount_pose(info, bottom, tip_dir, spin=self.SENSOR_SPIN)
                    attach_sensor(spec, calf, sensor, prefix=f"{leg}_", pos=mount, quat=quat,
                                  stiffness_scale=self.sensor_stiffness)
                else:
                    r = info["radius"]
                    calf.add_geom(name=f"{leg}_proxy", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                                  size=[r, 0, 0], pos=bottom - r * tip_dir, mass=info["mass"],
                                  rgba=[0.95, 0.95, 0.95, 1], group=1)

        # ---- position-controlled joints (MuJoCo does the PD, implicitly: stable and fast)
        for act in spec.actuators:
            joint = spec.joint(act.target)
            torque = float(act.ctrlrange[1])
            act.gaintype = mujoco.mjtGain.mjGAIN_FIXED       # = <position kp=.. kv=..>
            act.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            act.gainprm[0] = self.KP
            act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -self.KP, -self.KD
            act.ctrlrange = joint.range
            act.ctrllimited = True
            act.forcerange = [-torque, torque]
            act.forcelimited = True

        # ---- head camera, light, sky
        trunks[0].add_camera(name="headcam", pos=[0.33, 0, 0.02], fovy=70,
                             xyaxes=[0, -1, 0, 0.5, 0, 0.866])
        light = spec.worldbody.add_light(pos=[0, 0, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6],
                                         castshadow=False)
        if hasattr(mujoco, "mjtLightType"):                   # MuJoCo >= 3.3
            light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
        else:
            light.directional = True
        spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                         rgb1=[0.3, 0.5, 0.7], rgb2=[0, 0, 0], width=512, height=3072)
        spec.add_texture(name="tile_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, mark=mujoco.mjtMark.mjMARK_EDGE,
                         rgb1=[1, 1, 1], rgb2=[0.85, 0.85, 0.85], markrgb=[0.6, 0.6, 0.6],
                         width=256, height=256)
        spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
        spec.visual.headlight.diffuse = [0.5, 0.5, 0.5]
        if terrain:
            self._add_terrain(spec)
        return spec

    def _add_terrain(self, spec):
        # ---- terrain: per tile a heightfield (rough) and a box (flat), only one of them active,
        # sharing a material (for the colour). priority 2: the tile sets the contact feel.
        n, res, half = self.N_TILES, self.TILE_RES, self.TILE / 2
        for j in range(n):
            for i in range(n):
                name = f"tile_{i}_{j}"
                xy = [(i + 0.5) * self.TILE - n * half, (j + 0.5) * self.TILE - n * half]
                spec.add_hfield(name=name, size=[half, half, self.HF_RANGE, 0.1],
                                nrow=res + 1, ncol=res + 1, userdata=[0.5] * (res + 1) ** 2)
                mat = spec.add_material(name=name, texrepeat=[4, 4])
                tex = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
                tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "tile_grid"
                mat.textures = tex
                spec.worldbody.add_geom(
                    name=name, type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=name, material=name,
                    pos=[*xy, -self.HF_RANGE / 2], priority=2, condim=3, group=0)
                spec.worldbody.add_geom(
                    name=f"flat_{i}_{j}", type=mujoco.mjtGeom.mjGEOM_BOX, material=name,
                    size=[half, half, 0.1], pos=[*xy, -0.1], priority=2, condim=3, group=0)

    def _standing_down(self, spec):
        """World "down" in each calf's frame, with the trunk level and the legs in
        DEFAULT_POSE (the standing pose)."""
        m = spec.copy().compile()
        d = mujoco.MjData(m)
        for k, (leg, j) in enumerate((leg, j) for leg in LEGS for j in JOINTS):
            d.qpos[m.joint(f"{leg}_{j}_joint").qposadr[0]] = self.DEFAULT_POSE[k]
        mujoco.mj_kinematics(m, d)
        return {leg: d.xmat[m.body(f"{leg}_calf").id].reshape(3, 3).T @ np.array([0.0, 0, -1])
                for leg in LEGS}

    def _index_model(self):
        m = self.model
        self._base = m.body(self.trunk_name).id
        self.TRACK_BODY = self.trunk_name
        self._joint_qadr = np.array([m.joint(f"{leg}_{j}_joint").qposadr[0] for leg in LEGS for j in JOINTS])
        self._joint_dadr = np.array([m.joint(f"{leg}_{j}_joint").dofadr[0] for leg in LEGS for j in JOINTS])
        self._act = np.array([m.actuator(f"{leg}_{j}").id for leg in LEGS for j in JOINTS])
        lo, hi = m.actuator_ctrlrange[self._act].T
        mid, half = (lo + hi) / 2, (hi - lo) / 2
        self._soft_lo, self._soft_hi = mid - 0.9 * half, mid + 0.9 * half
        self._ctrl_lo, self._ctrl_hi = lo, hi

        # bodies / geoms
        def subtree(b):
            out, stack = [], [b]
            while stack:
                x = stack.pop()
                out.append(x)
                stack.extend(int(c) for c in np.flatnonzero(m.body_parentid == x) if c != x)
            return out
        ij = [(i, j) for j in range(self.N_TILES) for i in range(self.N_TILES)]
        self._rough_geoms = np.array([m.geom(f"tile_{i}_{j}").id for i, j in ij])
        self._flat_geoms = np.array([m.geom(f"flat_{i}_{j}").id for i, j in ij])
        self._tile_geoms = np.concatenate([self._rough_geoms, self._flat_geoms])
        self._tile_hfields = np.array([m.geom_dataid[g] for g in self._rough_geoms])
        self._tile_mats = np.array([m.geom_matid[g] for g in self._rough_geoms])
        self._is_tile = np.zeros(m.ngeom, bool)
        self._is_tile[self._tile_geoms] = True
        self._tile_rough = np.zeros(self.N_TILES ** 2, bool)

        robot_bodies = subtree(self._base)
        collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
        self._geom_foot = np.full(m.ngeom, -1, dtype=int)
        self._foot_body = []
        for k, leg in enumerate(LEGS):
            if self.feet == "tactip":
                root = m.body(f"{leg}_{self.foot_info['root']}").id
                bodies = subtree(root)
                self._foot_body.append(root)
            else:
                g = m.geom(leg if self.feet == "rigid" else f"{leg}_proxy").id
                bodies = []
                self._geom_foot[g] = k
                self._foot_body.append(int(m.geom_bodyid[g]))
            for b in bodies:
                self._geom_foot[np.flatnonzero((m.geom_bodyid == b) & collides)] = k
        self._foot_body = np.array(self._foot_body)
        # foot reference point: the bottom of the foot (where the old foot sphere ended and the
        # TacTip tip now ends), fixed in the calf frame
        self._calf = np.array([m.body(f"{leg}_calf").id for leg in LEGS])
        self._foot_tip_local = np.array([self.foot_bottom[leg] for leg in LEGS])
        in_robot = np.isin(m.geom_bodyid, robot_bodies) & collides
        self._is_trunk = in_robot & (m.geom_bodyid == self._base)
        self._is_shin = in_robot & (self._geom_foot < 0) & ~self._is_trunk   # thighs, calves, hips
        self._foot_geoms = [np.flatnonzero(self._geom_foot == k) for k in range(4)]

        # TacTip markers, one frame per foot
        self.n_markers = 0
        self._marker_ids = np.zeros(0, dtype=int)
        if self.feet == "tactip":
            ids, owner, frames = [], [], []
            for k, leg in enumerate(LEGS):
                pre = f"{leg}_{self.MARKER_PREFIX}"
                mk = [i for i in range(m.nsite) if m.site(i).name.startswith(pre)]
                ids += mk
                owner += [k] * len(mk)
                site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"{leg}_{self.SENSOR_SITE}")
                frames.append(("site", site) if site >= 0 else ("body", int(self._foot_body[k])))
            self._marker_ids = np.array(ids, dtype=int)
            self._marker_owner = np.array(owner, dtype=int)
            self._foot_frames = frames
            self.n_markers = len(ids)
            self.markers_per_foot = self.n_markers // 4
            d = mujoco.MjData(m)                    # model rest pose = undeformed skin
            mujoco.mj_forward(m, d)
            self._nodes_rest = self._nodes(d)
            self._nodes_last = self._nodes_rest.copy()
            self._nodes_delta = np.zeros_like(self._nodes_rest)

        # standing height: base height above the lowest foot point in the standing pose
        d = mujoco.MjData(m)
        d.qpos[self._joint_qadr] = self.DEFAULT_POSE
        d.qpos[:3] = [0, 0, 1.0]
        d.qpos[3:7] = [1, 0, 0, 0]
        mujoco.mj_forward(m, d)
        self.stand_height = 1.0 - self._lowest_foot_z(d).min()
        self.base_height_target = self.stand_height + self.BASE_HEIGHT_OFFSET
        self.robot_mass = float(m.body_subtreemass[self._base])

        # terrain grid (shared edges between neighbouring tiles)
        side = self.N_TILES * self.TILE_RES + 1
        self._heights = np.zeros((side, side))
        self._cell = self.TILE / self.TILE_RES
        self._origin = -self.N_TILES * self.TILE / 2
        self._tile_friction = np.ones(self.N_TILES ** 2)
        self._tile_softness = np.zeros(self.N_TILES ** 2)

    def _lowest_foot_z(self, d):
        """Lowest collision point of each foot (world z), from the geoms' bounding boxes."""
        m = self.model
        out = np.full(4, np.inf)
        signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
        for k in range(4):
            for g in self._foot_geoms[k]:
                c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
                R = d.geom_xmat[g].reshape(3, 3)
                z = (d.geom_xpos[g] + (c + signs * h) @ R.T)[:, 2].min()
                out[k] = min(out[k], z)
        return out

    # ================================================================ terrain
    def terrain_height(self, x, y):
        """Ground height at world (x, y) (arrays allowed), bilinear in the heightfield."""
        gx = (np.asarray(x) - self._origin) / self._cell
        gy = (np.asarray(y) - self._origin) / self._cell
        n = self._heights.shape[0] - 1
        gx, gy = np.clip(gx, 0, n - 1e-6), np.clip(gy, 0, n - 1e-6)
        i, j = gx.astype(int), gy.astype(int)
        fx, fy = gx - i, gy - j
        H = self._heights
        return ((1 - fx) * (1 - fy) * H[j, i] + fx * (1 - fy) * H[j, i + 1]
                + (1 - fx) * fy * H[j + 1, i] + fx * fy * H[j + 1, i + 1])

    def _tile_index(self, x, y):
        i = np.clip(((np.asarray(x) - self._origin) // self.TILE).astype(int), 0, self.N_TILES - 1)
        j = np.clip(((np.asarray(y) - self._origin) // self.TILE).astype(int), 0, self.N_TILES - 1)
        return j * self.N_TILES + i

    def _generate_terrain(self, spawn_xy, feel=None):
        rng, lvl, n, res = self.np_random, self.level, self.N_TILES, self.TILE_RES
        m = self.model
        side = self._heights.shape[0]

        # which tiles are rough: never the ones within SPAWN_FLAT of the spawn point
        lerp = lambda a, b: a + lvl * (b - a)                      # noqa: E731
        rough = rng.uniform(size=n * n) < lerp(*self.ROUGH_FRACTION)
        centres = self._origin + (np.arange(n) + 0.5) * self.TILE
        cx, cy = np.meshgrid(centres, centres)
        near = (np.abs(cx - spawn_xy[0]) < self.TILE / 2 + self.SPAWN_FLAT) & \
               (np.abs(cy - spawn_xy[1]) < self.TILE / 2 + self.SPAWN_FLAT)
        rough &= ~near.ravel()

        # rough tiles: mounds / dips (a few smooth Gaussian blobs) + 10 cm bumps, blended to
        # zero at the tile edges so every tile meets its neighbours at height 0
        u = np.linspace(0, self.TILE, res + 1)
        U, V = np.meshgrid(u, u)
        edge = np.minimum.reduce([U, V, self.TILE - U, self.TILE - V])
        window = np.clip(edge / self.EDGE_RAMP, 0, 1)
        window = window * window * (3 - 2 * window)                # smoothstep
        H = np.zeros((side, side))
        for t in np.flatnonzero(rough):
            i, j = t % n, t // n
            patch = np.zeros_like(U)
            for _ in range(rng.integers(1, 4)):
                c = rng.uniform(0.4, self.TILE - 0.4, 2)
                w = rng.uniform(0.25, 0.6)
                patch += rng.uniform(-1, 1) * np.exp(-((U - c[0]) ** 2 + (V - c[1]) ** 2) / (2 * w * w))
            patch *= lvl * self.MAX_WAVES / max(1.0, np.abs(patch).max())
            coarse = rng.uniform(-0.5, 0.5, size=(res // 2 + 1, res // 2 + 1))
            patch += lvl * self.MAX_BUMPS * rng.uniform(0.3, 1.0) * np.kron(
                coarse, np.ones((2, 2)))[:res + 1, :res + 1]
            patch = np.clip(patch * window, -self.HF_RANGE / 2, self.HF_RANGE / 2)
            H[j * res:j * res + res + 1, i * res:i * res + res + 1] = patch
            adr = m.hfield_adr[self._tile_hfields[t]]
            m.hfield_data[adr:adr + patch.size] = (patch / self.HF_RANGE + 0.5).ravel()
        self._heights = H
        self._tile_rough = rough

        # switch each tile's box / heightfield on or off (collisions and drawing)
        on, off = (1, 1), (0, 0)
        for t in range(n * n):
            gr, gf = self._rough_geoms[t], self._flat_geoms[t]
            (m.geom_contype[gr], m.geom_conaffinity[gr]) = on if rough[t] else off
            (m.geom_contype[gf], m.geom_conaffinity[gf]) = off if rough[t] else on
            m.geom_group[gr], m.geom_group[gf] = (0, 5) if rough[t] else (5, 0)   # 5 = hidden

        # feel, per tile
        f_lo = lerp(*[f[0] for f in self.FRICTION])
        f_hi = lerp(*[f[1] for f in self.FRICTION])
        friction = np.exp(rng.uniform(np.log(f_lo), np.log(f_hi), n * n))
        soft = rng.uniform(0, lerp(*self.SOFTNESS_MAX), n * n)
        damp = rng.uniform(lerp(*[d[0] for d in self.DAMPRATIO]), lerp(*[d[1] for d in self.DAMPRATIO]), n * n)
        if feel:                                    # the same feel everywhere (tests, evaluation)
            friction = np.full(n * n, float(feel.get("friction", 0.8)))
            soft = np.full(n * n, float(feel.get("softness", 0.0)))
            damp = np.full(n * n, float(feel.get("dampratio", 1.0)))
        soft = np.where(rough, np.minimum(soft, self.ROUGH_SOFTNESS_MAX), soft)
        tc = self.TIMECONST[0] + soft * (self.TIMECONST[1] - self.TIMECONST[0])
        dmin = self.SOLIMP_DMIN[0] + soft * (self.SOLIMP_DMIN[1] - self.SOLIMP_DMIN[0])
        width = self.SOLIMP_WIDTH[0] + soft * (self.SOLIMP_WIDTH[1] - self.SOLIMP_WIDTH[0])
        for g in (self._rough_geoms, self._flat_geoms):
            m.geom_friction[g, 0] = friction
            m.geom_solref[g, 0] = tc
            m.geom_solref[g, 1] = damp
            m.geom_solimp[g, 0] = dmin
            m.geom_solimp[g, 1] = 0.95
            m.geom_solimp[g, 2] = width
        # colour: slippery = blue, grippy = red-brown; softer = paler
        grip = np.clip((np.log(friction) - np.log(0.15)) / (np.log(1.3) - np.log(0.15)), 0, 1)
        base = np.outer(1 - grip, [0.25, 0.45, 0.85]) + np.outer(grip, [0.6, 0.35, 0.2])
        rgb = base + soft[:, None] * (1 - base) * 0.6
        m.mat_rgba[self._tile_mats, :3] = rgb
        self._tile_friction, self._tile_softness = friction, soft
        self._terrain_version += 1

    def set_terrain_level(self, level, curriculum=False):
        """Fix the terrain difficulty (0..1) from the next reset on."""
        self.level = float(np.clip(level, 0, 1))
        self.curriculum = curriculum

    def sync_renderer(self, renderer):
        """Upload the current heightfields to a mujoco.Renderer (or passive viewer) if they
        changed since it last drew; call before rendering after a reset."""
        if getattr(renderer, "_tacquad_terrain", None) == self._terrain_version:
            return
        try:
            if hasattr(renderer, "update_hfield"):                 # mujoco.viewer handle
                for h in self._tile_hfields:
                    renderer.update_hfield(int(h))
            else:
                renderer._gl_context.make_current()
                for h in self._tile_hfields:
                    mujoco.mjr_uploadHField(self.model, renderer._mjr_context, int(h))
            renderer._tacquad_terrain = self._terrain_version
        except AttributeError:
            pass

    # ================================================================ tactip read-out
    def _nodes(self, d=None):
        """Marker positions in their own foot's sensor frame, (n_markers, 3)."""
        d = self.data if d is None else d
        pos, mat = [], []
        for kind, i in self._foot_frames:
            if kind == "site":
                pos.append(d.site_xpos[i])
                mat.append(d.site_xmat[i].reshape(3, 3))
            else:
                pos.append(d.xpos[i])
                mat.append(d.xmat[i].reshape(3, 3))
        c = np.array(pos)[self._marker_owner]
        R = np.array(mat)[self._marker_owner]
        p = d.site_xpos[self._marker_ids]
        return np.einsum("ni,nij->nj", p - c, R).astype(np.float32)

    def get_nodes(self):
        return self._nodes()

    def get_foot_nodes(self):
        """Marker displacement from rest per foot, (4, markers_per_foot, 3) in metres."""
        disp = self._nodes() - self._nodes_rest
        return disp.reshape(4, -1, 3)

    def _update_marker_history(self):
        if self.n_markers:
            super()._update_marker_history()

    def _reset_marker_history(self):
        if self.n_markers:
            super()._reset_marker_history()

    # ================================================================ state helpers
    def _base_state(self):
        d = self.data
        R = d.xmat[self._base].reshape(3, 3)
        lin = R.T @ d.qvel[0:3]                  # base frame
        ang = d.qvel[3:6].copy()                 # free-joint angular velocity: already local
        grav = R.T @ np.array([0.0, 0.0, -1.0])
        return R, lin, ang, grav

    def _contacts(self):
        """Per foot normal force (N), robot-other-than-feet contacts, trunk contact."""
        m, d = self.model, self.data
        n = d.ncon
        forces = np.zeros(4)
        if n == 0:
            return forces, 0, False
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        t1, t2 = self._is_tile[g1], self._is_tile[g2]
        ground = t1 | t2
        other = np.where(t1, g2, g1)
        foot = np.where(ground, self._geom_foot[other], -1)
        idx = np.flatnonzero(foot >= 0)
        if idx.size:
            if m.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC:
                adr = d.contact.efc_address[idx]
                f = np.where(adr >= 0, d.efc_force[np.maximum(adr, 0)], 0.0)
            else:
                f = np.empty(idx.size)
                buf = np.zeros(6)
                for k, i in enumerate(idx):
                    mujoco.mj_contactForce(m, d, int(i), buf)
                    f[k] = buf[0]
            np.add.at(forces, foot[idx], np.abs(f))
        shin = int(np.count_nonzero(ground & self._is_shin[other]))
        trunk = bool(np.any(ground & self._is_trunk[other]))
        return forces, shin, trunk

    def _foot_pos(self):
        """World position of each foot's tip (bottom of the foot in the standing pose), (4, 3)."""
        d = self.data
        R = d.xmat[self._calf].reshape(4, 3, 3)
        return d.xpos[self._calf] + np.einsum("kij,kj->ki", R, self._foot_tip_local)

    # ================================================================ observation
    def _obs_parts(self):
        d = self.data
        R, lin, ang, grav = self._base_state()
        q = d.qpos[self._joint_qadr] - self.DEFAULT_POSE
        qd = d.qvel[self._joint_dadr]
        parts = {"proprio": np.concatenate([
            ang * 0.25, grav, self._command * [2.0, 2.0, 0.25], q, qd * 0.05, self._last_action])}
        if self.obs_modes["privileged"]:
            base = d.xpos[self._base]
            yaw = np.arctan2(R[1, 0], R[0, 0])
            gx, gy = np.meshgrid(*self.HEIGHT_GRID)
            c, s = np.cos(yaw), np.sin(yaw)
            px = base[0] + c * gx - s * gy
            py = base[1] + s * gx + c * gy
            scan = base[2] - self.terrain_height(px, py) - self.stand_height
            feet = self._foot_pos()
            tiles = self._tile_index(feet[:, 0], feet[:, 1])
            foot_h = feet[:, 2] - self.terrain_height(feet[:, 0], feet[:, 1])
            parts["privileged"] = np.concatenate([
                lin * 2.0,
                [base[2] - self.terrain_height(base[0], base[1]) - self.stand_height],
                self._foot_force / (self.robot_mass * 9.81),
                self._tile_friction[tiles], self._tile_softness[tiles],
                np.clip(foot_h, -0.1, 0.3) * 5.0,
                np.clip(scan.ravel(), -1, 1)])
        if self.obs_modes["tactile"]:
            parts["tactile"] = self._tactile_obs()
        if self.obs_modes["vision"]:
            parts["vision"] = self.get_vision()
        return parts

    def _get_obs(self):
        return self._format_obs(self._obs_parts())

    # ================================================================ commands
    def _sample_command(self):
        if self.fixed_command is not None:
            self._command = self.fixed_command.copy()
        else:
            rng = self.np_random
            cmd = np.array([rng.uniform(*r) for r in self.CMD_RANGES])
            if rng.uniform() < self.CMD_ZERO_PROB or np.linalg.norm(cmd[:2]) < 0.1 and abs(cmd[2]) < 0.1:
                cmd[:] = 0.0
            self._command = cmd
        self._cmd_timer = int(self.np_random.uniform(*self.CMD_SECONDS) / self.dt)

    # ================================================================ gym API
    def reset(self, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        options = options or {}
        if "terrain_level" in options:
            self.level = float(np.clip(options["terrain_level"], 0, 1))
        elif self.curriculum and self._last_outcome is not None:
            if self._last_outcome == "fell":
                self.level = max(0.0, self.level - self.LEVEL_STEP)
            elif self._last_outcome == "good":
                self.level = min(1.0, self.level + self.LEVEL_STEP)
        self._last_outcome = None

        m, d, rng = self.model, self.data, self.np_random
        mujoco.mj_resetData(m, d)
        spawn = rng.uniform(-0.5, 0.5, size=2)
        self._generate_terrain(spawn, options.get("feel"))

        yaw = rng.uniform(-np.pi, np.pi)
        d.qpos[0:2] = spawn
        d.qpos[2] = 1.0
        d.qpos[3:7] = _yaw_quat(yaw)
        d.qpos[self._joint_qadr] = self.DEFAULT_POSE + rng.uniform(-0.1, 0.1, 12)
        mujoco.mj_forward(m, d)
        feet = self._foot_pos()
        clearance = self._lowest_foot_z(d) - self.terrain_height(feet[:, 0], feet[:, 1])
        d.qpos[2] -= clearance.min() - 0.01
        d.qvel[0:2] = rng.uniform(-0.2, 0.2, 2)
        d.qvel[3:6] = rng.uniform(-0.2, 0.2, 3)
        d.ctrl[self._act] = np.clip(self.DEFAULT_POSE, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_forward(m, d)

        self.step_count = 0
        self._last_action = np.zeros(12)
        self._last_qvel = d.qvel[self._joint_dadr].copy()
        self._last_feet = self._foot_pos()
        self._air_time = np.zeros(4)
        self._last_contact = np.zeros(4, bool)
        self._foot_force = np.zeros(4)
        self._ep_terms = {k: 0.0 for k in self.REWARD_WEIGHTS}
        self._fail_reason = ""
        self._sample_command()
        self._reset_marker_history()
        return self._get_obs(), self._info()

    def step(self, action):
        m, d = self.model, self.data
        a = np.clip(np.asarray(action, dtype=np.float64).ravel(), -1.0, 1.0)
        target = self.DEFAULT_POSE + self.ACTION_SCALE * a
        d.ctrl[self._act] = np.clip(target, self._ctrl_lo, self._ctrl_hi)
        mujoco.mj_step(m, d, self.FRAME_SKIP)
        self.step_count += 1
        self._update_marker_history()

        self._foot_force, shin, trunk = self._contacts()
        R, lin, ang, grav = self._base_state()
        reward, terms = self._reward(a, lin, ang, grav, shin)
        for k, v in terms.items():
            self._ep_terms[k] += v

        status = None
        if trunk:
            status, self._fail_reason = "fail", "body hit the ground"
        elif grav[2] > self.TILT_LIMIT:
            status, self._fail_reason = "fail", "tipped over"
        terminated = status is not None
        base = d.xpos[self._base]
        out = np.abs(base[:2]).max() > self.N_TILES * self.TILE / 2 - 1.0
        truncated = not terminated and (self.step_count >= self.max_steps or out)
        if not np.all(np.isfinite(d.qpos)):                 # physics blew up: end cleanly
            terminated, status, self._fail_reason = True, "fail", "simulation unstable"
            mujoco.mj_resetData(m, d)

        self._last_action = a
        self._cmd_timer -= 1
        if self._cmd_timer <= 0:
            self._sample_command()
        if terminated or truncated:
            track = self._ep_terms["tracking_lin_vel"] / max(1, self.step_count) / (
                self.REWARD_WEIGHTS["tracking_lin_vel"] * self.dt)
            self._last_outcome = "fell" if terminated else ("good" if track > 0.7 else "ok")
        info = self._info(status, truncated)
        return self._get_obs(), float(reward), terminated, truncated, info

    def _info(self, status=None, truncated=False):
        info = {"is_success": bool(truncated and status is None and self.step_count >= self.max_steps),
                "fail_reason": self._fail_reason if status == "fail" else "",
                "terrain_level": self.level, "command": self._command.copy()}
        if status is not None or truncated:
            info["reward_terms"] = dict(self._ep_terms)
        return info

    # ================================================================ reward
    def _reward(self, a, lin, ang, grav, shin):
        d, w, dt = self.data, self.REWARD_WEIGHTS, self.dt
        cmd = self._command
        q = d.qpos[self._joint_qadr]
        qd = d.qvel[self._joint_dadr]
        base = d.xpos[self._base]
        moving = np.linalg.norm(cmd) > 0.1

        contact = self._foot_force > self.CONTACT_FORCE
        contact_filt = contact | self._last_contact
        first = (self._air_time > 0) & contact_filt
        self._air_time += dt
        air = np.sum((self._air_time - self.AIR_TIME_TARGET) * first) * moving
        self._air_time *= ~contact_filt
        self._last_contact = contact

        feet = self._foot_pos()
        foot_vel = (feet - self._last_feet) / dt
        self._last_feet = feet

        raw = dict(
            tracking_lin_vel=np.exp(-np.sum((cmd[:2] - lin[:2]) ** 2) / self.TRACKING_SIGMA),
            tracking_ang_vel=np.exp(-(cmd[2] - ang[2]) ** 2 / self.TRACKING_SIGMA),
            lin_vel_z=lin[2] ** 2,
            ang_vel_xy=np.sum(ang[:2] ** 2),
            orientation=np.sum(grav[:2] ** 2),
            base_height=(base[2] - self.terrain_height(base[0], base[1]) - self.base_height_target) ** 2,
            torques=np.sum(d.actuator_force[self._act] ** 2),
            dof_acc=np.sum(((qd - self._last_qvel) / dt) ** 2),
            action_rate=np.sum((a - self._last_action) ** 2),
            collision=float(shin),
            feet_air_time=air,
            dof_pos_limits=np.sum(np.clip(q - self._soft_hi, 0, None) + np.clip(self._soft_lo - q, 0, None)),
            feet_slip=np.sum(contact * np.sum(foot_vel[:, :2] ** 2, axis=1)),
            stand_still=np.sum(np.abs(q - self.DEFAULT_POSE)) * (not moving),
        )
        self._last_qvel = qd.copy()
        terms = {k: w[k] * v * dt for k, v in raw.items()}
        total = sum(terms.values())
        if self.ONLY_POSITIVE_REWARDS:
            total = max(total, 0.0)
        return total, terms

    # ================================================================ rendering
    def render(self):
        if self._view_renderer is None:
            self._view_renderer = mujoco.Renderer(self.model, height=480, width=640)
            self._view_cam = mujoco.MjvCamera()
            self._view_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._view_cam.trackbodyid = self._base
            self._view_cam.distance = self.TRACK_DISTANCE
            self._view_cam.azimuth, self._view_cam.elevation = 135, self.TRACK_ELEVATION
        self.sync_renderer(self._view_renderer)
        self._view_renderer.update_scene(self.data, camera=self._view_cam)
        return self._view_renderer.render().copy()

    def get_vision(self):
        new = self._vision_renderer is None
        if not new:
            self.sync_renderer(self._vision_renderer)
        img = super().get_vision()
        if new:                                  # a new renderer uploads the current terrain
            self._vision_renderer._tacquad_terrain = self._terrain_version
        return img


# ==================================================================== view / save the robot
def robot_spec(feet="tactip", sensor_stiffness=10.0, sensor_xml=None, robot_xml=None):
    """The Go2 with its TacTip (or proxy / rigid) feet on a plain floor, without the RL
    terrain: for looking at, or for use outside this env. Returns (MjSpec, standing qpos)."""
    env = TacQuadruped.__new__(TacQuadruped)      # only the model-building parts are needed
    env.feet, env.sensor_stiffness = feet, float(sensor_stiffness)
    spec = env._build_spec(sensor_xml, robot_xml, terrain=False)
    spec.add_texture(name="floor_grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, rgb1=[0.2, 0.3, 0.4],
                     rgb2=[0.1, 0.2, 0.3], width=300, height=300)
    mat = spec.add_material(name="floor_grid", texrepeat=[8, 8], texuniform=True)
    tex = [""] * int(mujoco.mjtTextureRole.mjNTEXROLE)
    tex[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = "floor_grid"
    mat.textures = tex
    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05],
                            material="floor_grid")
    m = spec.compile()
    d = mujoco.MjData(m)
    q = [m.joint(f"{leg}_{j}_joint").qposadr[0] for leg in LEGS for j in JOINTS]
    d.qpos[q] = TacQuadruped.DEFAULT_POSE
    d.qpos[2] = 1.0
    mujoco.mj_forward(m, d)
    env.model = m
    env._foot_geoms = [np.flatnonzero(_foot_mask(m, leg, feet)) for leg in LEGS]
    d.qpos[2] = 1.0 - TacQuadruped._lowest_foot_z(env, d).min() + 0.005
    spec.add_key(name="stand", qpos=d.qpos.copy(), ctrl=TacQuadruped.DEFAULT_POSE)
    return spec


def print_snippet(sensor_xml=None):
    """The <attach> XML that puts the sensor on each Go2 foot, for editing go2.xml by hand
    (MuJoCo >= 3.2). The env doesn't need this: it attaches the sensors itself."""
    env = TacQuadruped.__new__(TacQuadruped)
    path = sensor_xml or _asset(*TacQuadruped.SENSOR_XML)
    info = sensor_info(env._sensor_spec(sensor_xml), axis=TacQuadruped.SENSOR_AXIS)
    leg = "FL"                                     # every leg stands in the same pose
    down = env._standing_down(mujoco.MjSpec.from_file(TacQuadruped.find_robot()))[leg]
    tip_dir = down if TacQuadruped.TIP_DIRECTION == "down" else np.array([0.0, 0, -1])
    bottom = np.array(TacQuadruped.FOOT_CENTRE) + (TacQuadruped.FOOT_RADIUS + TacQuadruped.LEG_EXTENSION) * down
    pos, q = mount_pose(info, bottom, tip_dir, spin=TacQuadruped.SENSOR_SPIN)
    fmt = lambda v: " ".join(f"{0.0 if abs(x) < 1e-9 else x:.6g}" for x in v)   # noqa: E731
    print(f"""<!-- in <asset> (file path relative to go2.xml): -->
<model name="tactip" file="{path}"/>

<!-- in each <body name="XX_calf">, replacing <geom name="XX" class="foot"/> and the
     <geom ... mesh="foot" .../> visual, with XX = FL, FR, RL, RR: -->
<frame pos="{fmt(pos)}" quat="{fmt(q)}">
  <attach model="tactip" body="{info['root']}" prefix="XX_"/>
</frame>

<!-- MuJoCo ignores contacts between the calf and the mount (its child). If deeper sensor
     bodies reach the calf's collision geoms, add in <contact> for each of them:
     <exclude body1="XX_calf" body2="XX_<sensor body>"/>   (the env does this for you) -->""")


def _foot_mask(m, leg, feet):
    """Colliding geoms of one foot (its sensor subtree, proxy sphere or foot sphere)."""
    collides = (m.geom_contype != 0) | (m.geom_conaffinity != 0)
    if feet == "rigid":
        return (np.arange(m.ngeom) == m.geom(leg).id) & collides
    if feet == "proxy":
        return (np.arange(m.ngeom) == m.geom(f"{leg}_proxy").id) & collides
    calf = m.body(f"{leg}_calf").id
    sub = [b for b in range(m.nbody) if b != calf and _is_under(m, b, calf)]
    return np.isin(m.geom_bodyid, sub) & collides


def _is_under(m, body, ancestor):
    while body != 0:
        if body == ancestor:
            return True
        body = m.body_parentid[body]
    return False


def _main():
    import argparse
    ap = argparse.ArgumentParser(description="View or save the Go2 with TacTip feet.")
    ap.add_argument("--view", action="store_true", help="open it in the MuJoCo viewer")
    ap.add_argument("--save", metavar="XML", help="write it as a standalone MJCF file")
    ap.add_argument("--snippet", action="store_true",
                    help="print the XML to attach the sensors by hand in go2.xml instead")
    ap.add_argument("--feet", default="tactip", choices=["tactip", "proxy", "rigid"])
    ap.add_argument("--stiffness", type=float, default=10.0, help="sensor_stiffness")
    ap.add_argument("--robot", default=None, help="go2.xml to use (default: found under assets/)")
    ap.add_argument("--sensor", default=None, help="sensor XML (default: assets/sensor/tactip.xml)")
    args = ap.parse_args()
    if args.snippet:
        print_snippet(args.sensor)
        return
    spec = robot_spec(args.feet, args.stiffness, args.sensor, args.robot)
    m = spec.compile()
    print(f"{TacQuadruped.find_robot(args.robot)} + {args.feet} feet: {m.nbody} bodies, "
          f"{m.njnt} joints, {sum(m.site(i).name[3:].startswith('s_c') for i in range(m.nsite))} "
          "TacTip markers; keyframe 'stand' holds the standing pose")
    if args.save:
        from tactip_mujoco_gym.sensor_tools import save_spec
        save_spec(spec, args.save)
        print("saved", args.save)
    if args.view or not args.save:
        import mujoco.viewer
        d = mujoco.MjData(m)
        mujoco.mj_resetDataKeyframe(m, d, m.key("stand").id)
        mujoco.viewer.launch(m, d)


if __name__ == "__main__":
    _main()