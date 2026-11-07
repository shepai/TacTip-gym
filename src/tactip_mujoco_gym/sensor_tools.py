"""
Reuse one TacTip model on any robot: extract it once, attach as many prefixed copies as needed.

Why not <include> it several times? <include> pastes the XML text, so every copy has the same
names (core, pin0_slide, s_c0, ...) and MuJoCo refuses duplicates. Copying the file to
sensor1/2/3 by hand works, but every equality, tendon and exclude has to be renamed in step.
MuJoCo's model editing API (MjSpec, MuJoCo >= 3.2) does the renaming for you: attaching a
body subtree with a prefix renames every body, joint, geom, site, camera, equality, tendon and
the references between them (FL_core, FL_pin0_slide, FL_s_c0, ...).

1. Extract the sensor from your arm scene into a standalone file (once):

       python -m tactip_mujoco_gym.sensor_tools extract \\
           src/tactip_mujoco_gym/assets/tactip_arm.xml \\
           --out src/tactip_mujoco_gym/assets/sensor/tactip.xml

   The sensor is found from its marker sites (s_c*): everything attached to the arm link
   that carries them (link06) is kept - mount, skin, pins, cameras you choose to keep - plus
   the equality constraints and tendons between those parts and the meshes / materials they
   use. The link's own joint and arm geometry are dropped; what is left of it becomes the
   sensor's root body, "tactip". `--body NAME` keeps just one body's subtree instead.
   `python -m tactip_mujoco_gym.sensor_tools tree <scene>` shows where the markers are.

2. Attach copies in Python:

       spec = mujoco.MjSpec.from_file("go2.xml")
       for leg in ("FL", "FR", "RL", "RR"):
           attach_sensor(spec, spec.body(f"{leg}_calf"), "tactip.xml", prefix=f"{leg}_",
                         tip_at=[0, 0, -0.235], axis_to=[0, 0, -1])   # tip there, pointing down
       model = spec.compile()

   or in XML (MuJoCo >= 3.2), without Python:

       <asset> <model name="tactip" file="sensor/tactip.xml"/> </asset>
       ...
       <body name="FL_calf" ...>
         <attach model="tactip" body="tactip" prefix="FL_"/>
       </body>

   Python is the more flexible route: attach_sensor also points the sensor where you want,
   can stiffen it, and stops it colliding with the link it is mounted on.
"""
import argparse
import os
import warnings

import mujoco
import numpy as np

MARKER_PREFIX = "s_c"

# Notices MuJoCo prints whenever a sensor is attached to a robot, all expected here:
#   "Attach conflict ... keeping parent value"  the robot's simulation options (timestep,
#       integrator, tolerance) win over the sensor file's, which is what we want
#   "Attaching a spec with an empty prefix"     the arm's own TacTip and the RB-Y1 are attached
#       without a name prefix; this only matters for MjSpec.to_xml, which save_spec handles
_ATTACH_NOTICES = ("Attach conflict when attaching", "Attaching a spec with an empty prefix")


_OPTION_FIELDS = ("timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance",
                  "ccd_tolerance", "density", "viscosity", "o_margin", "integrator", "cone",
                  "jacobian", "solver", "iterations", "ls_iterations", "noslip_iterations",
                  "ccd_iterations", "disableflags", "enableflags", "disableactuator",
                  "sdf_initpoints", "sdf_iterations", "gravity", "wind", "magnetic", "o_solref",
                  "o_solimp", "o_friction")


def match_options(target, source):
    """Copy `source`'s simulation options (timestep, integrator, ...) into `target`. Called
    on a sensor spec before it is attached to a robot: the robot's options win anyway, but
    MuJoCo prints an "Attach conflict" warning (from C, in every process) for each one that
    differs. Fields this MuJoCo version doesn't have are skipped."""
    for name in _OPTION_FIELDS:
        try:
            value = getattr(source.option, name)
        except AttributeError:
            continue
        try:
            setattr(target.option, name, np.array(value) if np.ndim(value) else value)
        except (AttributeError, TypeError, ValueError):
            try:
                getattr(target.option, name)[:] = value
            except Exception:
                pass


def compile_spec(spec):
    """spec.compile() without MuJoCo's expected attach notices (every other warning shows)."""
    with warnings.catch_warnings():
        for msg in _ATTACH_NOTICES:
            warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        return spec.compile()


# ----------------------------------------------------------------------------- extract
def _subtree_bodies(body):
    out = [body]
    for child in body.bodies:
        out.extend(_subtree_bodies(child))
    return out


def locate_sensor(scene_xml, marker_prefix=MARKER_PREFIX):
    """Where the sensor is in a scene. Returns (mode, body name, report):
        ("link", L)     the markers hang off actuated link L (e.g. link06): the sensor is
                        everything attached to L
        ("subtree", B)  no actuated link above them: the sensor is the whole tree B (a
                        top-level body) that holds the markers"""
    m = compile_spec(mujoco.MjSpec.from_file(str(scene_xml)))
    markers = [i for i in range(m.nsite) if m.site(i).name.startswith(marker_prefix)]
    if not markers:
        prefixes = sorted({m.site(i).name.rstrip("0123456789") for i in range(m.nsite)})[:20]
        raise ValueError(f"no '{marker_prefix}*' sites in {scene_xml}; site names start with "
                         f"{prefixes}. Pass marker_prefix=... if your markers are named differently.")

    def chain(b):                                    # b, parent, ..., world (0)
        out = [int(b)]
        while out[-1] != 0:
            out.append(int(m.body_parentid[out[-1]]))
        return out
    chains = [chain(b) for b in set(m.site_bodyid[markers])]
    common = [b for b in chains[0] if all(b in c for c in chains)]
    lca = common[0]
    actuated = {int(m.jnt_bodyid[m.actuator_trnid[a, 0]]) for a in range(m.nu)
                if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}
    path = chain(lca)
    link = next((b for b in path if b in actuated), None)
    if link is not None:
        mode, body = "link", m.body(link).name
    else:
        mode, body = "subtree", m.body(path[-2]).name     # top-level ancestor (child of world)
    return mode, body, dict(n_markers=len(markers), marker_bodies=len(chains),
                            common_parent=m.body(lca).name, path=[m.body(b).name for b in path[:-1]])


def extract_sensor(scene_xml, root_body="auto", out_xml=None, keep_cameras=("sensor_cam",),
                   marker_prefix=MARKER_PREFIX):
    """Return an MjSpec holding only the sensor, placed at the origin, and write it to out_xml
    if given. root_body="auto" finds the sensor from its markers (see locate_sensor);
    a body name keeps just that body's subtree. Mesh / texture paths are rewritten relative to
    out_xml, so the file works from wherever it is saved."""
    src = mujoco.MjSpec.from_file(str(scene_xml))
    if root_body == "auto":
        mode, root_body, _ = locate_sensor(scene_xml, marker_prefix)
    else:
        mode = "subtree"
        if src.body(root_body) is None:
            names = [b.name for b in src.bodies if b.name][:30]
            raise ValueError(f"no body {root_body!r} in {scene_xml}; bodies include {names}")
    spec = mujoco.MjSpec()
    spec.modelname = "tactip"
    spec.compiler.degree = src.compiler.degree
    spec.compiler.autolimits = src.compiler.autolimits
    match_options(spec, src)                             # (no "Attach conflict" to report)
    root = spec.worldbody.add_frame().attach_body(src.body(root_body), "", "")
    root.pos = [0, 0, 0]
    root.quat = [1, 0, 0, 0]
    if mode == "link":
        _strip_link(spec, root)

    n = sum(s.name.startswith(marker_prefix) for s in spec.sites)
    if n == 0:
        mode, auto, info = locate_sensor(scene_xml, marker_prefix)
        raise ValueError(
            f"body {root_body!r} holds none of the '{marker_prefix}*' marker sites. They are "
            f"under {info['common_parent']!r} (path {' > '.join(reversed(info['path']))}). "
            f"Leave out --body (or use root_body='auto') to extract the whole sensor from "
            f"{auto!r}.")

    for cam in list(spec.cameras):
        if cam.name not in keep_cameras:
            spec.delete(cam)
    _prune_assets(spec)

    # the copied meshes / textures still name files relative to the scene: make them absolute
    _absolute_paths(spec, src.modelfiledir or os.path.dirname(str(scene_xml)),
                    src.meshdir, src.texturedir)
    if out_xml is not None:
        save_spec(spec, out_xml)
        spec = mujoco.MjSpec.from_file(str(out_xml))     # re-read: paths now resolve from there
    compile_spec(spec)                                   # fail here, not later, if it is broken
    return spec


def _strip_link(spec, root, name="tactip"):
    """Turn a copied arm link into a bare mount: drop its joint(s), its own geoms and the
    actuators / sensors / keyframes of the arm, give it a tiny mass, and rename it."""
    old = root.name
    gone = {j.name for j in root.joints}
    for j in list(root.joints):
        spec.delete(j)
    for g in list(root.geoms):
        spec.delete(g)
    for a in list(spec.actuators):
        spec.delete(a)
    for s in list(spec.sensors):
        if s.objname in gone or s.objname == old:
            spec.delete(s)
    for k in list(spec.keys):
        spec.delete(k)
    root.explicitinertial = True
    root.mass = 1e-3
    root.ipos = [0, 0, 0]
    root.inertia = [1e-7, 1e-7, 1e-7]
    root.name = name
    for e in spec.excludes:
        e.bodyname1 = name if e.bodyname1 == old else e.bodyname1
        e.bodyname2 = name if e.bodyname2 == old else e.bodyname2
    for eq in spec.equalities:
        eq.name1 = name if eq.name1 == old else eq.name1
        eq.name2 = name if eq.name2 == old else eq.name2


def describe(scene_xml, marker_prefix=MARKER_PREFIX):
    """Print where the sensor is and what extract will keep."""
    mode, body, info = locate_sensor(scene_xml, marker_prefix)
    m = compile_spec(mujoco.MjSpec.from_file(str(scene_xml)))
    print(f"{info['n_markers']} '{marker_prefix}*' markers on {info['marker_bodies']} bodies; "
          f"they all hang under {info['common_parent']!r}")
    print("path from world: " + " > ".join(reversed(info["path"])))
    top = m.body(body).id
    print(f"extract keeps {'everything attached to the arm link' if mode == 'link' else 'the tree'} "
          f"{body!r}{' (its own joint and geoms are dropped)' if mode == 'link' else ''}:")
    kids = [b for b in range(m.nbody) if m.body_parentid[b] == top and b != top]
    for b in kids:
        sub = [x for x in range(m.nbody) if _ancestor(m, x, b)]
        mk = sum(m.site(i).name.startswith(marker_prefix) for i in range(m.nsite) if m.site_bodyid[i] in sub)
        print(f"  {m.body(b).name or '(unnamed)':<24} {len(sub):4d} bodies, "
              f"{int(np.isin(m.geom_bodyid, sub).sum()):4d} geoms, "
              f"{int(np.isin(m.jnt_bodyid, sub).sum()):4d} joints, {mk:4d} markers")


def _ancestor(m, body, ancestor):
    while True:
        if body == ancestor:
            return True
        if body == 0:
            return False
        body = int(m.body_parentid[body])


def _absolute_paths(spec, base, meshdir="", texturedir=""):
    """Rewrite relative mesh / texture / heightfield file names as absolute paths."""
    base = os.path.abspath(base or ".")
    for items, sub in ((spec.meshes, meshdir), (spec.hfields, meshdir), (spec.textures, texturedir)):
        for a in items:
            if getattr(a, "file", "") and not os.path.isabs(a.file):
                a.file = os.path.normpath(os.path.join(base, sub or "", a.file))


def save_spec(spec, out_xml):
    """Write an MjSpec as an MJCF file that loads from where it is saved: asset file paths
    are made relative to out_xml (the spec's own relative paths are resolved first)."""
    _absolute_paths(spec, spec.modelfiledir, spec.meshdir, spec.texturedir)
    spec.meshdir = spec.texturedir = ""
    out_dir = os.path.dirname(os.path.abspath(out_xml))
    for items in (spec.meshes, spec.hfields, spec.textures):
        for a in items:
            if getattr(a, "file", ""):
                a.file = os.path.relpath(a.file, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_xml, "w") as f:
        f.write(_fix_defaults(spec.to_xml()))


def _fix_defaults(xml):
    """MjSpec.to_xml writes the attached model's main defaults as an unnamed <default> nested
    inside the top-level one, which MuJoCo can't read back. Both are the main default, so
    move the nested one's contents up a level."""
    import xml.etree.ElementTree as ET
    root = ET.fromstring(xml)

    def unwrap(parent):
        for child in list(parent):
            if child.tag == "default" and "class" not in child.attrib:
                unwrap(child)
                i = list(parent).index(child)
                parent.remove(child)
                for k, sub in enumerate(list(child)):
                    parent.insert(i + k, sub)
    for top in root.findall("default"):
        unwrap(top)
    ET.indent(root, "  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def _prune_assets(spec):
    """Drop meshes, materials and textures the sensor doesn't use (the arm's meshes etc.)."""
    meshes = {g.meshname for g in spec.geoms if g.meshname}
    materials = {x.material for x in list(spec.geoms) + list(spec.sites) + list(spec.tendons)
                 if getattr(x, "material", "")}
    for x in list(spec.skins) + list(spec.flexes):
        materials.add(x.material)
    for m in list(spec.meshes):
        if m.name not in meshes:
            spec.delete(m)
    for mat in list(spec.materials):
        if mat.name not in materials:
            spec.delete(mat)
    textures = {t for mat in spec.materials for t in mat.textures if t}
    for t in list(spec.textures):
        if t.name not in textures:
            spec.delete(t)


# ----------------------------------------------------------------------------- inspect
TIP_SITE = "sensor_tip"       # optional site marking the tip: position = tip, z axis = outward
TIP_DEPTH = 0.01              # m behind the tip that counts as "the tip end" for `radius`
                              # (sensors without markers)


def sensor_info(sensor, marker_prefix=MARKER_PREFIX, sensor_site="ee_site", axis=None):
    """Geometry of a standalone sensor at rest, in its root body's frame:
        axis    tip direction: the z axis of a "sensor_tip" site if the sensor has one, else
                from the centre of a sphere fitted to the markers (the dome) through their
                centroid; or the `axis` given
        centre  that dome centre (the tip itself for a sensor with a "sensor_tip" site)
        tip     the "sensor_tip" site, else the furthest point of the collision geometry along
                the axis, on the axis line
        reach   distance from the root origin to the tip along the axis (m)
        radius  sideways extent of the tip end (dome + skin) from the axis (m)
        n_markers, mass, timestep, has_site (whether sensor_site exists), root

    A sensor without TacTip markers needs a "sensor_tip" site (or `axis`) so it can be
    mounted: <site name="sensor_tip" pos="0 0 0.03" zaxis="0 0 1"/>."""
    spec = _as_spec(sensor)
    m = compile_spec(spec)
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    root = spec.worldbody.first_body()
    rid = m.body(root.name).id
    origin, R = d.xpos[rid], d.xmat[rid].reshape(3, 3)
    markers = [i for i in range(m.nsite) if m.site(i).name.startswith(marker_prefix)]
    tip_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TIP_SITE)
    has_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, sensor_site) >= 0
    if tip_site >= 0 or not markers:
        return _info_without_dome(m, d, rid, origin, R, markers, tip_site, axis, has_site,
                                  root.name, marker_prefix)
    P = (d.site_xpos[markers] - origin) @ R            # markers in the root frame
    centroid = P.mean(axis=0)
    centre = _sphere_centre(P)
    if axis is None:
        v = centroid - centre
        if np.linalg.norm(v) < 1e-6:                    # no dome shape: fall back to the root
            v = centroid
        axis = v / np.linalg.norm(v)
        if axis @ (centroid - 0.0) < 0 and np.linalg.norm(centroid) > 1e-6:
            axis = -axis                                # point away from the mount
    else:
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
    pts = (_collision_points(m, d, axis=R @ axis) - origin) @ R
    along = (pts - centre) @ axis
    tip = centre + along.max() * axis
    start = ((P - centre) @ axis).min() - 1e-3          # the dome end: from the lowest marker on
    near = pts[along >= start]
    lateral = near - centre - np.outer((near - centre) @ axis, axis)
    radius = float(np.linalg.norm(lateral, axis=1).max())
    return dict(axis=axis, centre=centre, tip=tip, reach=float(tip @ axis), radius=radius,
                n_markers=len(markers), mass=float(m.body_subtreemass[rid]),
                timestep=float(m.opt.timestep), has_site=has_site, root=root.name)


def _info_without_dome(m, d, rid, origin, R, markers, tip_site, axis, has_site, root_name,
                       marker_prefix):
    """sensor_info for a sensor with a "sensor_tip" site, or without markers."""
    if tip_site >= 0:
        tip = (d.site_xpos[tip_site] - origin) @ R
        if axis is None:
            axis = R.T @ d.site_xmat[tip_site].reshape(3, 3)[:, 2]
    elif axis is None:
        raise ValueError(
            f"the sensor (root body {root_name!r}) has no '{marker_prefix}*' marker sites to "
            f"find its tip from. Add a site named '{TIP_SITE}' at the centre of the sensing "
            "surface, with its z axis pointing outward, e.g. "
            f'<site name="{TIP_SITE}" pos="0 0 0.03" zaxis="0 0 1"/> (or pass axis=...).')
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    pts = (_collision_points(m, d, axis=R @ axis, exact_cylinders=True) - origin) @ R
    if tip_site < 0:                                    # furthest point along the axis line
        tip = (pts @ axis).max() * axis
    along = (pts - tip) @ axis
    near = pts[along >= -TIP_DEPTH]
    if len(near) == 0:
        near = pts
    lateral = near - tip - np.outer((near - tip) @ axis, axis)
    radius = float(np.linalg.norm(lateral, axis=1).max())
    return dict(axis=axis, centre=np.asarray(tip, float).copy(), tip=np.asarray(tip, float),
                reach=float(np.asarray(tip) @ axis), radius=radius, n_markers=len(markers),
                mass=float(m.body_subtreemass[rid]), timestep=float(m.opt.timestep),
                has_site=has_site, root=root_name)


def _sphere_centre(P):
    """Least-squares sphere through points P (n, 3); returns its centre."""
    if len(P) < 4:
        return P.mean(axis=0)
    A = np.c_[2 * P, np.ones(len(P))]
    sol, *_ = np.linalg.lstsq(A, (P ** 2).sum(axis=1), rcond=None)
    return sol[:3]


def _collision_points(m, d, axis=None, exact_cylinders=False):
    """Extreme points of every colliding geom (world frame): exact for spheres and capsules
    (along `axis` and perpendicular to it), and for cylinders with exact_cylinders (both rims;
    off by default so the TacTip's measured geometry is unchanged), bounding-box corners for
    other shapes."""
    pts = []
    signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float)
    if axis is not None:
        a = np.asarray(axis, float)
        p1 = np.cross(a, [1, 0, 0] if abs(a[0]) < 0.9 else [0, 1, 0])
        p1 /= np.linalg.norm(p1)
        dirs = np.array([a, -a, p1, -p1, np.cross(a, p1), -np.cross(a, p1)])
    for g in range(m.ngeom):
        if m.geom_contype[g] == 0 and m.geom_conaffinity[g] == 0:
            continue
        R = d.geom_xmat[g].reshape(3, 3)
        t = m.geom_type[g]
        if axis is not None and t == mujoco.mjtGeom.mjGEOM_SPHERE:
            pts.append(d.geom_xpos[g] + m.geom_size[g, 0] * dirs)
        elif axis is not None and t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            for e in (-1, 1):
                pts.append(d.geom_xpos[g] + e * m.geom_size[g, 1] * R[:, 2] + m.geom_size[g, 0] * dirs)
        elif axis is not None and exact_cylinders and t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            z, r, h = R[:, 2], m.geom_size[g, 0], m.geom_size[g, 1]
            perp = dirs - np.outer(dirs @ z, z)             # each direction, across the cylinder
            n = np.linalg.norm(perp, axis=1, keepdims=True)
            rim = np.where(n > 1e-9, r * perp / np.maximum(n, 1e-12), 0.0)
            for e in (-1, 1):                               # both end faces
                pts.append(d.geom_xpos[g] + e * h * z + rim)
        else:
            c, h = m.geom_aabb[g, :3], m.geom_aabb[g, 3:]
            pts.append(d.geom_xpos[g] + (c + signs * h) @ R.T)
    return np.concatenate(pts) if pts else np.zeros((1, 3))


def mount_pose(info, tip_at, tip_dir, spin=0.0):
    """Position and quat for the sensor's root body so that its tip sits at `tip_at` and
    points along `tip_dir` (both in the parent body's frame), turned by `spin` radians about
    that direction."""
    q = quat_between(info["axis"], tip_dir)
    if spin:
        u = np.asarray(tip_dir, float) / np.linalg.norm(tip_dir)
        qs = np.array([np.cos(spin / 2), *(np.sin(spin / 2) * u)])
        out = np.zeros(4)
        mujoco.mju_mulQuat(out, qs, q)
        q = out
    Rq = np.zeros(9)
    mujoco.mju_quat2Mat(Rq, q)
    pos = np.asarray(tip_at, float) - Rq.reshape(3, 3) @ info["tip"]
    return pos, q


def _as_spec(sensor):
    if isinstance(sensor, mujoco.MjSpec):
        return sensor.copy()
    spec = mujoco.MjSpec.from_file(str(sensor))
    # absolute asset paths, so the copies still find their meshes inside another model
    _absolute_paths(spec, spec.modelfiledir or os.path.dirname(str(sensor)), spec.meshdir,
                    spec.texturedir)
    return spec


# ----------------------------------------------------------------------------- attach
def quat_between(a, b):
    """Shortest rotation taking unit vector a to unit vector b, as a MuJoCo quat (w, x, y, z)."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    c = float(a @ b)
    if c < -1 + 1e-9:                                    # opposite: any perpendicular axis
        perp = np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 1, 0])
        perp /= np.linalg.norm(perp)
        return np.array([0.0, *perp])
    q = np.array([1 + c, *np.cross(a, b)])
    return q / np.linalg.norm(q)


# MuJoCo 3.7 turned mjsJoint / mjsTendon `stiffness` and `damping` into arrays (linear
# coefficient first, then higher-order polynomial terms); earlier versions use plain numbers.
# These work with both.
def set_coef(obj, name, value):
    """Set a spec joint's / tendon's stiffness or damping to a linear value (no higher-order
    terms)."""
    cur = getattr(obj, name)
    if np.ndim(cur):
        arr = np.zeros(np.shape(cur))
        arr.flat[0] = value
        setattr(obj, name, arr)
    else:
        setattr(obj, name, float(value))


def scale_coef(obj, name, k):
    """Multiply a spec joint's / tendon's stiffness or damping (every term) by k."""
    cur = getattr(obj, name)
    setattr(obj, name, np.asarray(cur, dtype=float) * k if np.ndim(cur) else float(cur) * k)


def attach_sensor(spec, parent, sensor, prefix, pos=(0, 0, 0), quat=None, axis=None,
                  axis_to=None, tip_at=None, stiffness_scale=1.0, exclude_parent=True):
    """Attach a prefixed copy of a standalone sensor to body `parent` of `spec`.

        pos              where the sensor's root body origin goes, in the parent's frame
        quat             its orientation in the parent's frame, or
        axis_to          a direction in the parent's frame for the sensor tip to point along
                         (the tip axis is measured from the markers unless `axis` is given)
        tip_at           with axis_to: put the tip of the sensor at this point (parent frame)
                         instead of placing the root body at `pos`
        stiffness_scale  multiplies stiffness, damping and armature of the sensor's joints and
                         tendons. Scaling all three keeps the natural frequency (so the time
                         step it needs) and the damping ratio unchanged, but carries
                         stiffness_scale x the load at the same deflection: a skin tuned for a
                         few newtons of finger pressure saturates under a robot's weight.
        exclude_parent   no contacts between the sensor and the link it is mounted on

    Returns the attached root body. Its elements are all named prefix + original name."""
    child = _as_spec(sensor)
    root_src = child.worldbody.first_body()
    if tip_at is not None:
        pos, quat = mount_pose(sensor_info(child, axis=axis), tip_at,
                               [0, 0, -1] if axis_to is None else axis_to)
    elif quat is None:
        if axis_to is None:
            quat = [1, 0, 0, 0]
        else:
            ax = sensor_info(child)["axis"] if axis is None else np.asarray(axis, float)
            quat = quat_between(ax, axis_to)
    match_options(child, spec)          # the robot's timestep etc. (no "Attach conflict" warning)
    frame = parent.add_frame(pos=list(pos), quat=list(quat))
    root = frame.attach_body(root_src, prefix, "")

    bodies = _subtree_bodies(root)
    if stiffness_scale != 1.0:
        k = float(stiffness_scale)
        for b in bodies:
            for j in b.joints:
                scale_coef(j, "stiffness", k)
                scale_coef(j, "damping", k)
                j.armature *= k
        for t in spec.tendons:
            if t.name.startswith(prefix):
                scale_coef(t, "stiffness", k)
                scale_coef(t, "damping", k)
    if exclude_parent:
        for b in bodies:
            spec.add_exclude(bodyname1=parent.name, bodyname2=b.name)
    return root


# ----------------------------------------------------------------------------- scene surgery
# Used to swap the sensor on robots whose scene XML has the TacTip written in (the arm): the
# built-in sensor is cut out of the scene as plain XML (with everything that refers to it),
# then a different sensor is attached with attach_sensor().

# attribute -> kind of element it names ("*" = any kind)
_REF_KIND = {
    "body": "body", "body1": "body", "body2": "body",
    "joint": "joint", "joint1": "joint", "joint2": "joint", "jointinparent": "joint",
    "site": "site", "site1": "site", "site2": "site", "sidesite": "site", "cranksite": "site",
    "slidersite": "site", "refsite": "site",
    "geom": "geom", "geom1": "geom", "geom2": "geom",
    "tendon": "tendon", "tendon1": "tendon", "tendon2": "tendon",
    "actuator": "actuator", "camera": "camera",
    "objname": "*", "refname": "*",
}
_DEF_KIND = {"body": "body", "joint": "joint", "freejoint": "joint", "geom": "geom",
             "site": "site", "camera": "camera", "light": "light"}
# top-level sections whose children may refer to worldbody elements, and the kind each
# child's own name has
_SECTIONS = {"tendon": "tendon", "actuator": "actuator", "equality": "equality",
             "sensor": "sensor", "contact": "contact", "deformable": "deformable"}


def expand_includes(xml_path):
    """Parse an MJCF file with every <include> expanded in place, following MuJoCo's rules:
    file paths are relative to the main file's directory, and the included file's top-level
    element is dropped (its children take the include's place). Returns the root Element."""
    import xml.etree.ElementTree as ET
    xml_path = os.path.abspath(str(xml_path))
    base = os.path.dirname(xml_path)
    root = ET.parse(xml_path).getroot()

    def expand(elem, depth=0):
        if depth > 50:
            raise RecursionError(f"<include> nesting too deep in {xml_path}")
        i = 0
        while i < len(elem):
            child = elem[i]
            if child.tag == "include":
                path = os.path.join(base, child.attrib["file"])
                inner = ET.parse(path).getroot()
                expand(inner, depth + 1)
                elem.remove(child)
                for k, sub in enumerate(list(inner)):
                    elem.insert(i + k, sub)
                i += len(inner)
            else:
                expand(child, depth + 1)
                i += 1
    expand(root)
    return root


def _refs(elem):
    """(kind, name) pairs that `elem` and its descendants refer to."""
    out = []
    for e in elem.iter():
        for attr, kind in _REF_KIND.items():
            v = e.attrib.get(attr)
            if v:
                out.append((kind, v))
    return out


def remove_bodies(root, body_names):
    """Delete the named bodies (with their subtrees) from an expanded MJCF tree, then every
    tendon, actuator, equality, sensor, contact pair / exclude and skin that refers to
    anything deleted (repeatedly, so a sensor on a deleted tendon goes too). Keyframes are
    dropped if any joint or actuator went (their qpos / ctrl lengths no longer match).
    Returns {kind: set of deleted names}."""
    removed = {k: set() for k in ("body", "joint", "geom", "site", "camera", "light",
                                  "tendon", "actuator", "equality", "sensor", "contact",
                                  "deformable")}
    parents = {c: p for p in root.iter() for c in p}
    wanted = set(body_names)
    for body in [b for b in root.iter("body") if b.attrib.get("name") in wanted]:
        for e in body.iter():
            kind = _DEF_KIND.get(e.tag)
            if kind and e.attrib.get("name"):
                removed[kind].add(e.attrib["name"])
        parents[body].remove(body)
        wanted.discard(body.attrib.get("name"))
    if wanted:
        raise ValueError(f"no body named {sorted(wanted)} in the scene")

    def hit(kind, name):
        if kind == "*":
            return any(name in s for s in removed.values())
        if kind == "body" and " " in name:                     # flex / skin body lists
            return any(n in removed["body"] for n in name.split())
        return name in removed[kind]

    changed = True
    while changed:
        changed = False
        for section in [e for e in root if e.tag in _SECTIONS]:
            kind = _SECTIONS[section.tag]
            for child in list(section):
                if any(hit(k, n) for k, n in _refs(child)):
                    section.remove(child)
                    unnamed = f"<{child.tag} {len(removed[kind])}>"     # (can't match a name)
                    removed[kind].add(child.attrib.get("name") or unnamed)
                    changed = True
    if removed["joint"] or removed["actuator"]:
        for kf in [e for e in root if e.tag == "keyframe"]:
            root.remove(kf)
    return removed


def scene_without_bodies(scene_xml, body_names):
    """The scene as one MJCF string with the named bodies (and everything referring to them)
    taken out; mesh / texture directories made absolute so it loads from a string.
    Returns (xml string, {kind: deleted names})."""
    import xml.etree.ElementTree as ET
    scene_xml = os.path.abspath(str(scene_xml))
    base = os.path.dirname(scene_xml)
    root = expand_includes(scene_xml)
    removed = remove_bodies(root, body_names)
    compilers = root.findall("compiler")
    if not compilers:
        compilers = [ET.Element("compiler")]
        root.insert(0, compilers[0])
    for c in compilers:
        asset = c.attrib.pop("assetdir", None)
        for attr in ("meshdir", "texturedir"):
            d = c.attrib.get(attr, asset if asset is not None else "")
            c.attrib[attr] = d if os.path.isabs(d) else os.path.normpath(os.path.join(base, d))
    return ET.tostring(root, encoding="unicode"), removed


def child_bodies(scene_xml, parent):
    """Names of the bodies directly inside body `parent` of a scene (includes expanded)."""
    root = expand_includes(scene_xml)
    for b in root.iter("body"):
        if b.attrib.get("name") == parent:
            return [c.attrib.get("name") for c in b if c.tag == "body"]
    raise ValueError(f"no body named {parent!r} in {scene_xml}")


# ----------------------------------------------------------------------------- CLI
def _main():
    ap = argparse.ArgumentParser(description="Extract a TacTip from a scene into its own XML.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("extract", help="write the sensor as a standalone model")
    ex.add_argument("scene", help="scene XML that contains the sensor (e.g. tactip_arm.xml)")
    ex.add_argument("--body", default="auto",
                    help="'auto' (default): find the sensor from its markers; or one body to keep")
    ex.add_argument("--out", required=True, help="output XML, e.g. assets/sensor/tactip.xml")
    tr = sub.add_parser("tree", help="show where the sensor's markers are in a scene")
    tr.add_argument("scene")
    info = sub.add_parser("info", help="print the geometry of a standalone sensor")
    info.add_argument("sensor")
    args = ap.parse_args()
    if args.cmd == "tree":
        describe(args.scene)
        return
    if args.cmd == "extract":
        extract_sensor(args.scene, args.body, args.out)
        path = args.out
    else:
        path = args.sensor
    i = sensor_info(path)
    print(f"{path}: root body '{i['root']}', {i['n_markers']} markers, mass {1e3 * i['mass']:.0f} g")
    print(f"  tip axis {np.round(i['axis'], 3)} and tip point {np.round(1e3 * i['tip'], 1)} mm "
          f"(root frame), {1e3 * i['reach']:.1f} mm from the root, dome radius "
          f"{1e3 * i['radius']:.1f} mm, ee_site {'found' if i['has_site'] else 'missing'}")


if __name__ == "__main__":
    _main()
