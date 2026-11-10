"""
More sensor hardware, built in code: the PressTip, a GelSight, a TacTip read as a depth image
(the Tactile Gym 2.0 way) and an array of rat-like whiskers. Each is registered as a sensor
model, so any robot takes it by name, with readouts to match:

    gym.make("TacRig-v0", sensor_xml="gelsight")              # "tactile_image": GelSight RGB
    gym.make("TacArm-surface", sensor_xml="whiskers")         # "tactile": bend + vibration
    gym.make("TacGripper-pick", sensor_xml="presstip")        # "tactile": 15 pads, piezo, accel
    gym.make("TacArm-edge", sensor_xml="tactip_depth", obs_modes=["tactile_image"])

    name           readouts it brings (on top of the defaults)
    presstip       "tactile"        presstip: 15 force pads, piezo vibration, 3-axis accel
    gelsight       "tactile_image"  contact_depth in "gelsight" mode (RGB from three lights)
                   "tactile"        contact_map: a coarse grid of contact depth (mm)
    tactip_depth   "tactile_image"  contact_depth: contact depth image (Tactile Gym 2.0)
                   "tactile"        contact_map
    whiskers       "tactile"        whiskers: per whisker bend (2 axes), contact, vibration

The builders take the sizes and stiffnesses, so variants are one call away:

    from tactip_mujoco_gym.sensors import hardware, register_sensor_model
    register_sensor_model("whiskers_5x6", lambda: hardware.whisker_array(rows=5, cols=6),
                          "a bigger pad", readouts=hardware.READOUTS["whiskers"])
    hardware.write("gelsight", "gelsight.xml")               # the MJCF, to read or edit

All of them point their sensing side along +z and carry a "sensor_tip" site there, which is
how robots mount them (sensors/models.py). They are rigid parts on sprung joints, which the
robot's 1 ms time step handles; each joint's armature keeps its natural frequency below
MAX_OMEGA. No hinge has a range, so the angle unit of the scene they are attached to never
matters (slides have ranges, in metres).

Modelling notes, and where they depart from the real devices:
  PressTip  (Shepherd et al.; shepai.github.io/dev/PressTip.html) 42 mm across; a silicone
            door-stopper pad over 15 Velostat force pads on the underside of the board, a
            piezoelectric film and a 3-axis accelerometer. The pad layout isn't published:
            here a centre pad and rings of 6 and 8. The silicone is a rigid cap on springs
            (press, shear, tilt); a pad reads the contact force spread over the pads around
            the contact point. The piezo reads the high-passed pad force; the accelerometer
            sits on the cap (accel_on="board" puts it on the rigid board, where in a
            simulation it only feels the robot).
  GelSight  a GelSight Mini-sized gel (18.6 x 14.3 mm) with a camera behind it. The gel is a
            rigid slab on springs; what the camera sees is how far objects come into the
            space the gel fills at rest (rendered, so any shape works), and the "gelsight"
            image shades that depth map with three coloured lights.
  TacTip depth  a dome the size of the soft TacTip's (35 mm radius), rigid on springs, with
            a depth camera at its centre: the contact depth image of Tactile Gym 2.0 (Lin et
            al., 2022), which used it for sim-to-real with real TacTip images.
  whiskers  rows x cols tapered whiskers fanned out from a pad, each a chain of capsules on
            two-axis sprung hinges, curved like a rat's, longer towards the back. Whiskers
            don't collide with each other or the pad, only with the world.
"""
import math

import numpy as np

MAX_OMEGA = 250.0            # rad/s: the fastest any sensor joint may ring (armature sets it)
TIP = '<site name="sensor_tip" pos="0 0 {z:.6g}" zaxis="0 0 1" size="0.001" rgba="0 0 0 0"/>'


# ----------------------------------------------------------------------------- helpers
def _f(*v):
    return " ".join(f"{float(x):.6g}" for x in v)


def _quat_z_to(d):
    """Quaternion (w, x, y, z) turning +z onto direction d."""
    d = np.asarray(d, float) / np.linalg.norm(d)
    z = np.array([0.0, 0.0, 1.0])
    c = float(z @ d)
    if c > 1 - 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    if c < -1 + 1e-12:
        return np.array([0.0, 1.0, 0.0, 0.0])
    ax = np.cross(z, d)
    ax /= np.linalg.norm(ax)
    a = math.acos(c)
    return np.array([math.cos(a / 2), *(math.sin(a / 2) * ax)])


def _quat_axis(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    return np.array([math.cos(angle / 2), *(math.sin(angle / 2) * axis)])


def _qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _qrot(q, v):
    w, x, y, z = q
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                  [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                  [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)


def _joint(name, kind, axis, k, mass, zeta=0.7, rng=None):
    """A sprung joint whose armature keeps its natural frequency, sqrt(k / (inertia +
    armature)), below MAX_OMEGA whatever it moves; `mass` (the inertia it moves, roughly)
    sets the damping for a damping ratio of about `zeta`. stiffness_scale (attach_sensor)
    scales stiffness, damping and armature together, so this still holds."""
    arm = k / MAX_OMEGA ** 2
    damping = 2 * zeta * math.sqrt(k * (mass + arm))
    r = "" if rng is None else f' range="{_f(*rng)}"'
    return (f'<joint name="{name}" type="{kind}" axis="{_f(*axis)}" stiffness="{k:.6g}" '
            f'damping="{damping:.6g}" armature="{arm:.6g}"{r}/>')


def _mujoco(model, body, extra=""):
    return f"""<mujoco model="{model}">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.001" integrator="implicitfast"/>
  <worldbody>
{body}
  </worldbody>{extra}
</mujoco>
"""


def _spec(xml):
    import mujoco
    return mujoco.MjSpec.from_string(xml)


# ----------------------------------------------------------------------------- PressTip
def presstip_pads(rings=((0, 0.0), (6, 0.008), (8, 0.0145))):
    """Pad centres (x, y) in metres: (count, radius) per ring, each ring turned half a step."""
    xy = []
    for i, (n, r) in enumerate(rings):
        if n == 1 or r == 0.0:
            xy.append((0.0, 0.0))
            continue
        off = math.pi / n * (i % 2)
        xy += [(r * math.cos(2 * math.pi * k / n + off), r * math.sin(2 * math.pi * k / n + off))
               for k in range(n)]
    return np.array(xy)


def presstip_xml(diameter=0.042, depth=0.030, dome_radius=0.017, dome_height=0.010,
                 rings=((0, 0.0), (6, 0.008), (8, 0.0145)), pad_size=0.005,
                 stiffness=2000.0, shear_stiffness=1500.0, tilt_stiffness=0.3,
                 travel=0.004, accel_on="pad"):
    """The PressTip as MJCF text (sizes in m, stiffness in N/m and N m/rad).

    diameter, depth          the round housing (the board is its top face, at z = 0)
    dome_radius, dome_height the silicone cap over the pads (base radius, height)
    rings, pad_size          the force pads: (count, radius) rings, square pads
    stiffness ...            the silicone: pressing, sideways and tilting
    travel                   how far the silicone can be pressed in (m)
    accel_on                 "pad" (the silicone cap) or "board"
    """
    R = (dome_radius ** 2 + dome_height ** 2) / (2 * dome_height)   # cap sphere radius
    zc = dome_height - R
    if R > diameter / 2 or zc - R < -depth:
        raise ValueError("the silicone cap doesn't fit in the housing: make it flatter")
    pads = "\n".join(
        f'      <site name="pad_{k}" type="box" pos="{_f(x, y, 0.0002)}" '
        f'size="{_f(pad_size / 2, pad_size / 2, 0.0002)}" rgba="0.05 0.05 0.05 1"/>'
        for k, (x, y) in enumerate(presstip_pads(rings)))
    m_cap = 0.006
    accel = f'<site name="accel" pos="0 0 {dome_height / 2:.6g}" size="0.0015" rgba="0.9 0.6 0.1 1"/>'
    body = f"""    <body name="presstip">
      <geom name="housing" type="cylinder" size="{_f(diameter / 2, depth / 2)}" pos="0 0 {-depth / 2:.6g}"
            mass="0.03" rgba="0.12 0.12 0.14 1" friction="0.8 0.005 0.0001"/>
      <geom name="board" type="cylinder" size="{_f(diameter / 2 - 0.001, 0.0004)}" pos="0 0 -0.0002"
            contype="0" conaffinity="0" mass="0" rgba="0.08 0.35 0.18 1"/>
      <site name="board" pos="0 0 0" size="0.001" rgba="0 0 0 0"/>
      {accel if accel_on == "board" else ""}
{pads}
      <body name="pad" pos="0 0 0">
        {_joint("press_z", "slide", (0, 0, 1), stiffness, m_cap, rng=(-travel, 0.001))}
        {_joint("shear_x", "slide", (1, 0, 0), shear_stiffness, m_cap, rng=(-0.003, 0.003))}
        {_joint("shear_y", "slide", (0, 1, 0), shear_stiffness, m_cap, rng=(-0.003, 0.003))}
        {_joint("tilt_x", "hinge", (1, 0, 0), tilt_stiffness, 0.4 * m_cap * R * R)}
        {_joint("tilt_y", "hinge", (0, 1, 0), tilt_stiffness, 0.4 * m_cap * R * R)}
        <geom name="surface" type="sphere" size="{R:.6g}" pos="0 0 {zc:.6g}" mass="{m_cap}"
              rgba="0.86 0.82 0.72 0.75" friction="1 0.005 0.0001" solref="0.005 1"
              solimp="0.95 0.99 0.001"/>
        {accel if accel_on != "board" else ""}
      </body>
      {TIP.format(z=dome_height)}
    </body>"""
    return _mujoco("presstip", body)


def presstip(**kw):
    """The PressTip as an MjSpec (arguments: presstip_xml)."""
    return _spec(presstip_xml(**kw))


# ----------------------------------------------------------------------------- GelSight
def gelsight_xml(gel=(0.0186, 0.0143), thickness=0.004, housing=(0.032, 0.028, 0.024),
                 camera_distance=0.015, stiffness=5000.0, tilt_stiffness=0.5, travel=0.0018,
                 proud=0.002):
    """A GelSight-style sensor as MJCF text.

    gel              sensing area (x, y), m; GelSight Mini's by default
    thickness        of the gel
    housing          outer size (x, y, depth)
    camera_distance  from the camera to the gel surface (sets the camera's field of view)
    stiffness        of the gel pressed straight in (N/m) and tilted (N m/rad)
    travel           how far the gel can be pushed in before it bottoms out
    proud            how far the gel stands out of the housing
    """
    gx, gy = gel
    hx, hy, hd = housing
    fovy = math.degrees(2 * math.atan(gy / 2 / camera_distance))
    m_gel = 0.004
    front = -proud
    rim = []                                  # the frame around the gel window
    for name, (sx, sy, px, py) in {
            "rim_x0": ((hx - gx) / 4, hy / 2, -(hx + gx) / 4, 0.0),
            "rim_x1": ((hx - gx) / 4, hy / 2, (hx + gx) / 4, 0.0),
            "rim_y0": (gx / 2, (hy - gy) / 4, 0.0, -(hy + gy) / 4),
            "rim_y1": (gx / 2, (hy - gy) / 4, 0.0, (hy + gy) / 4)}.items():
        rim.append(f'      <geom name="{name}" type="box" size="{_f(sx, sy, (thickness - proud) / 2)}" '
                   f'pos="{_f(px, py, front - (thickness - proud) / 2)}" mass="0.002" rgba="0.15 0.15 0.17 1"/>')
    back = hd - (thickness - proud) - proud
    body = f"""    <body name="gelsight">
      <geom name="housing" type="box" size="{_f(hx / 2, hy / 2, back / 2)}" pos="0 0 {-thickness - back / 2:.6g}"
            mass="0.03" rgba="0.15 0.15 0.17 1"/>
{chr(10).join(rim)}
      <camera name="depth_cam" pos="0 0 {-camera_distance:.6g}" xyaxes="1 0 0 0 -1 0" fovy="{fovy:.4g}"/>
      <body name="gel" pos="0 0 0">
        {_joint("press_z", "slide", (0, 0, 1), stiffness, m_gel, rng=(-travel, 0.0005))}
        {_joint("tilt_x", "hinge", (1, 0, 0), tilt_stiffness, m_gel * gy * gy / 12)}
        {_joint("tilt_y", "hinge", (0, 1, 0), tilt_stiffness, m_gel * gx * gx / 12)}
        <geom name="surface" type="box" size="{_f(gx / 2, gy / 2, thickness / 2)}" pos="0 0 {-thickness / 2:.6g}"
              mass="{m_gel}" rgba="0.78 0.78 0.8 0.85" friction="1 0.005 0.0001"
              solref="0.005 1" solimp="0.95 0.99 0.001"/>
      </body>
      {TIP.format(z=0.0)}
    </body>"""
    return _mujoco("gelsight", body)


def gelsight(**kw):
    """A GelSight-style sensor as an MjSpec (arguments: gelsight_xml)."""
    return _spec(gelsight_xml(**kw))


# ----------------------------------------------------------------------------- TacTip (depth)
def tactip_depth_xml(radius=0.035, mount_radius=0.04, mount_length=0.04, fovy=120.0,
                     stiffness=800.0, shear_stiffness=600.0, tilt_stiffness=0.15, travel=0.006):
    """A TacTip read as a depth image, as MJCF text: a rigid dome on springs with a depth
    camera at its centre (Tactile Gym 2.0). Sized like the soft TacTip by default."""
    if radius > mount_radius:
        raise ValueError("the dome must fit on the mount")
    m_tip = 0.01
    I_tip = 0.4 * m_tip * radius ** 2
    body = f"""    <body name="tactip_depth">
      <geom name="mount" type="cylinder" size="{_f(mount_radius, mount_length / 2)}" pos="0 0 {-mount_length / 2:.6g}"
            mass="0.05" rgba="0.1 0.1 0.1 1"/>
      <camera name="depth_cam" pos="0 0 0" xyaxes="1 0 0 0 -1 0" fovy="{fovy:.4g}"/>
      <body name="tip" pos="0 0 0">
        {_joint("press_z", "slide", (0, 0, 1), stiffness, m_tip, rng=(-travel, 0.001))}
        {_joint("shear_x", "slide", (1, 0, 0), shear_stiffness, m_tip, rng=(-0.004, 0.004))}
        {_joint("shear_y", "slide", (0, 1, 0), shear_stiffness, m_tip, rng=(-0.004, 0.004))}
        {_joint("tilt_x", "hinge", (1, 0, 0), tilt_stiffness, I_tip)}
        {_joint("tilt_y", "hinge", (0, 1, 0), tilt_stiffness, I_tip)}
        <geom name="surface" type="sphere" size="{radius:.6g}" mass="{m_tip}" rgba="0.95 0.95 0.95 1"
              friction="1 0.005 0.0001" solref="0.005 1" solimp="0.95 0.99 0.001"/>
      </body>
      {TIP.format(z=radius)}
    </body>"""
    return _mujoco("tactip_depth", body)


def tactip_depth(**kw):
    """The depth-image TacTip as an MjSpec (arguments: tactip_depth_xml)."""
    return _spec(tactip_depth_xml(**kw))


# ----------------------------------------------------------------------------- whiskers
def whisker_layout(rows=3, cols=4, spacing=(0.012, 0.009), fan=0.2, length=(0.03, 0.06)):
    """[(base xy, direction quaternion, length)] per whisker, row by row. Whiskers fan out
    by `fan` rad per step from the middle, and get longer along x (the back of the pad)."""
    out = []
    for r in range(rows):
        for c in range(cols):
            u, v = c - (cols - 1) / 2, r - (rows - 1) / 2
            d = (math.tan(fan * u), math.tan(fan * v), 1.0)
            L = length[0] + (length[1] - length[0]) * (c / (cols - 1) if cols > 1 else 0.5)
            out.append(((u * spacing[0], v * spacing[1]), _quat_z_to(d), L))
    return out


def whisker_tips(rows=3, cols=4, spacing=(0.012, 0.009), fan=0.2, length=(0.03, 0.06),
                 curve=0.3, segments=4):
    """Where each whisker's tip is at rest (pad frame), shape (rows * cols, 3)."""
    tips = []
    for (x, y), q, L in whisker_layout(rows, cols, spacing, fan, length):
        seg = L / segments
        p, qq = np.array([x, y, 0.0]), q.copy()
        for s in range(segments):
            if s:
                qq = _qmul(qq, _quat_axis((1, 0, 0), curve / (segments - 1)))
            p = p + _qrot(qq, (0, 0, seg))
        tips.append(p)
    return np.array(tips)


def whisker_array_xml(rows=3, cols=4, spacing=(0.012, 0.009), fan=0.2, length=(0.03, 0.06),
                      curve=0.3, segments=4, radius=(0.0006, 0.0002), base_stiffness=1e-3,
                      stiffness=2e-3, zeta=0.2, pad=(0.044, 0.032, 0.016)):
    """An array of whiskers as MJCF text.

    rows, cols        whiskers on the pad (rows across y, columns along x)
    spacing, fan      base spacing (m) and how far each whisker turns out per step (rad)
    length            shortest and longest whisker (m), increasing along x
    curve             total intrinsic curvature of a whisker (rad, bending towards -y)
    segments          capsules per whisker (each joint: two sprung hinges)
    radius            whisker radius at the base and the tip (thicker than a rat's, so
                      contacts are robust at a 1 ms step)
    base_stiffness    follicle stiffness (N m/rad), and `stiffness` of the joints along the
                      whisker at its base; it falls with the radius squared towards the tip
    zeta              damping ratio of every joint: low, so whiskers ring when they flick
    pad               the pad (x, y, depth) the whiskers sprout from (front face at z = 0)

    The "sensor_tip" is level with the furthest whisker tip, so a robot that brings its tip
    to a surface makes the first contact with the longest whiskers, and pressing further
    bends them and brings in the shorter ones.
    """
    px, py, pd = pad
    lines = []
    for i, ((x, y), q, L) in enumerate(whisker_layout(rows, cols, spacing, fan, length)):
        seg = L / segments
        bend = _quat_axis((1, 0, 0), curve / (segments - 1)) if segments > 1 else np.array([1.0, 0, 0, 0])
        ind = "      "
        opened = 0
        for s in range(segments):
            r0 = radius[0] + (radius[1] - radius[0]) * (s + 0.5) / segments
            mass = 1300.0 * math.pi * r0 * r0 * seg                  # keratin-ish
            reach = (segments - s) * seg                               # what this joint swings
            moved = mass * (segments - s) * (reach / 2) ** 2           # rough inertia it moves
            if s == 0:
                pos, quat, k = (x, y, 0.0), q, base_stiffness
                name = f"w{i}"
            else:
                pos, quat = (0.0, 0.0, seg), bend
                k = stiffness * (r0 / radius[0]) ** 2
                name = f"w{i}_b{s}"
            jx, jy = ("bx", "by") if s == 0 else (f"j{s}x", f"j{s}y")
            lines.append(f'{ind}<body name="{name}" pos="{_f(*pos)}" quat="{_f(*quat)}" gravcomp="1">')
            ind += "  "
            opened += 1
            lines.append(f"{ind}{_joint(f'w{i}_{jx}', 'hinge', (1, 0, 0), k, moved, zeta)}")
            lines.append(f"{ind}{_joint(f'w{i}_{jy}', 'hinge', (0, 1, 0), k, moved, zeta)}")
            if s == 0:
                lines.append(f'{ind}<site name="w{i}_base" size="0.0012" rgba="0.8 0.3 0.2 1"/>')
            lines.append(f'{ind}<geom name="w{i}_s{s}" class="whisker" size="{r0:.6g}" '
                         f'fromto="0 0 0 0 0 {seg:.6g}" mass="{mass:.6g}"/>')
            if s == segments - 1:
                lines.append(f'{ind}<site name="w{i}_tip" pos="0 0 {seg:.6g}" size="0.0008" rgba="0.2 0.2 0.2 1"/>')
        for _ in range(opened):
            ind = ind[:-2]
            lines.append(f"{ind}</body>")
    tip_z = float(whisker_tips(rows, cols, spacing, fan, length, curve, segments)[:, 2].max())
    body = f"""    <body name="whiskers">
      <geom name="pad" type="box" size="{_f(px / 2, py / 2, pd / 2)}" pos="0 0 {-pd / 2:.6g}" mass="0.03"
            contype="4" conaffinity="1" rgba="0.55 0.42 0.38 1"/>
      <geom name="mount" type="cylinder" size="0.02 0.01" pos="0 0 {-pd - 0.01:.6g}" mass="0.02"
            contype="4" conaffinity="1" rgba="0.12 0.12 0.14 1"/>
{chr(10).join(lines)}
      {TIP.format(z=tip_z)}
    </body>"""
    default = """
  <default>
    <default class="whisker">
      <geom type="capsule" contype="2" conaffinity="1" condim="3" friction="0.5 0.005 0.0001"
            rgba="0.92 0.9 0.85 1"/>
    </default>
  </default>"""
    xml = _mujoco("whiskers", body)
    return xml.replace("  <worldbody>", default.strip("\n") + "\n  <worldbody>", 1)


def whisker_array(**kw):
    """Whiskers as an MjSpec (arguments: whisker_array_xml)."""
    return _spec(whisker_array_xml(**kw))


# ----------------------------------------------------------------------------- registry
BUILDERS = {"presstip": presstip_xml, "gelsight": gelsight_xml, "tactip_depth": tactip_depth_xml,
            "whiskers": whisker_array_xml}

READOUTS = {
    "presstip": {"tactile": "presstip"},
    "gelsight": {"tactile_image": {"type": "contact_depth", "mode": "gelsight", "width": 64,
                                   "height": 48, "max_depth": 0.0015},
                 "tactile": {"type": "contact_map", "width": 64, "height": 48, "max_depth": 0.0015}},
    "tactip_depth": {"tactile_image": {"type": "contact_depth", "max_depth": 0.005},
                     "tactile": {"type": "contact_map", "max_depth": 0.005}},
    "whiskers": {"tactile": "whiskers"},
}

DESCRIPTIONS = {
    "presstip": "the PressTip (Shepherd et al.): 15 force pads under silicone, piezo, accelerometer",
    "gelsight": "a GelSight-style gel with a camera behind it (contact depth, RGB shading)",
    "tactip_depth": "a TacTip read as a contact depth image, as in Tactile Gym 2.0",
    "whiskers": "a 3 x 4 array of rat-like whiskers: bend direction, contact, vibration",
}


def write(name, path, **kw):
    """Write the MJCF of a built-in sensor (`name` in BUILDERS) to `path`."""
    with open(path, "w") as f:
        f.write(BUILDERS[name](**kw))
    return path
