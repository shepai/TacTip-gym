"""
add_taxels.py - cover every geom in a MuJoCo worldbody file with a grid of touch sensors ("taxels").

For each supported geom it adds thin box <site> elements across the surface, and writes a second
file with one <touch> sensor per site.

    python add_taxels.py shape_worldbody.xml --assets shape_assets.xml \
           --out shape_worldbody_tx.xml --sensors shape_sensors.xml

Input files must have a <mujoco> root with <body> elements inside (the include files used by the
gyms). Supported geoms: box, cylinder, sphere, and 6-vertex triangular-prism meshes (the triangle
in ShapeInsertion). Geoms with a rotation (quat/euler/...) are skipped with a warning.
"""
import argparse
import math
import xml.etree.ElementTree as ET

import numpy as np

DEPTH = 0.004          # half-thickness of each taxel zone (m). Contacts must fall inside it.

# First matching prefix wins. res = taxel spacing (m). faces = which box faces / cylinder caps
# to cover ("+z", "-x", ...), or None for the whole surface.
RULES = [
    ("plat_",        dict(res=0.010, faces=["+z"])),   # platform slabs: top surface only
    ("ramp_surface", dict(res=0.010, faces=["+z"])),
    ("ramp_",        dict(res=0.005, faces=None)),     # rails and end stop
    ("",             dict(res=0.004, faces=None)),     # all objects: full surface
]


def rule_for(name):
    for prefix, rule in RULES:
        if name.startswith(prefix):
            return rule


def vec(s, default=None):
    return np.array([float(x) for x in s.split()]) if s else default


def fmt(a):
    return " ".join(f"{x:.5f}" for x in a)


# ---------------------------------------------------------------- sampling
# A "cell" is (centre, x_axis, y_axis, size_x, size_y), all in the geom's local frame.

def _inside(poly2, p):
    """Point in convex polygon (2D), either winding."""
    sign = None
    for i in range(len(poly2)):
        a, b = poly2[i], poly2[(i + 1) % len(poly2)]
        c = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
        if abs(c) < 1e-12:
            continue
        if sign is None:
            sign = c > 0
        elif sign != (c > 0):
            return False
    return True


def sample_polygon(poly, res):
    """Grid of cells over a planar convex polygon given by 3D vertices in cyclic order."""
    poly = np.asarray(poly, float)
    u = poly[1] - poly[0]
    u /= np.linalg.norm(u)
    n = np.cross(poly[1] - poly[0], poly[2] - poly[0])
    n /= np.linalg.norm(n)
    v = np.cross(n, u)
    rel = poly - poly[0]
    p2 = np.stack([rel @ u, rel @ v], axis=1)
    lo, hi = p2.min(axis=0), p2.max(axis=0)
    nu = max(1, int(round(float(hi[0] - lo[0]) / res)))
    nv = max(1, int(round(float(hi[1] - lo[1]) / res)))
    cu, cv = (hi[0] - lo[0]) / nu, (hi[1] - lo[1]) / nv
    cells = []
    for i in range(nu):
        for j in range(nv):
            c2 = lo + np.array([(i + 0.5) * cu, (j + 0.5) * cv])
            if _inside(p2, c2):
                cells.append((poly[0] + c2[0] * u + c2[1] * v, u, v, cu, cv))
    return cells


def box_cells(h, res, faces):
    cells = []
    for ax in range(3):
        for sgn in (1, -1):
            tag = ("+" if sgn > 0 else "-") + "xyz"[ax]
            if faces is not None and tag not in faces:
                continue
            a, b = [k for k in range(3) if k != ax]
            poly = []
            for sa, sb in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = np.zeros(3)
                p[ax], p[a], p[b] = sgn * h[ax], sa * h[a], sb * h[b]
                poly.append(p)
            cells += sample_polygon(poly, res)
    return cells


def cylinder_cells(r, h, res, faces):
    cells = []
    if faces is None:                                   # curved side
        nt = max(3, int(round(2 * math.pi * r / res)))
        nz = max(1, int(round(2 * h / res)))
        for i in range(nt):
            th = (i + 0.5) * 2 * math.pi / nt
            tangent = np.array([-math.sin(th), math.cos(th), 0.0])
            up = np.array([0.0, 0.0, 1.0])
            for k in range(nz):
                z = -h + (k + 0.5) * 2 * h / nz
                cells.append((np.array([r * math.cos(th), r * math.sin(th), z]),
                              tangent, up, 2 * math.pi * r / nt, 2 * h / nz))
    circle = [np.array([r * math.cos(a), r * math.sin(a), 0.0])
              for a in np.linspace(0, 2 * math.pi, 32, endpoint=False)]
    for sgn, tag in ((1, "+z"), (-1, "-z")):            # flat caps
        if faces is None or tag in faces:
            cells += sample_polygon([p + np.array([0, 0, sgn * h]) for p in circle], res)
    return cells


def sphere_cells(r, res):
    cells = []
    nlat = max(2, int(round(math.pi * r / res)))
    for i in range(nlat):
        th = (i + 0.5) * math.pi / nlat
        nlon = max(1, int(round(2 * math.pi * r * math.sin(th) / res)))
        for j in range(nlon):
            ph = (j + 0.5) * 2 * math.pi / nlon
            n = np.array([math.sin(th) * math.cos(ph), math.sin(th) * math.sin(ph), math.cos(th)])
            east = np.array([-math.sin(ph), math.cos(ph), 0.0])
            cells.append((r * n, east, np.cross(n, east),
                          2 * math.pi * r * math.sin(th) / nlon, math.pi * r / nlat))
    return cells


def prism_cells(v, res):
    """v = 6 vertices: bottom triangle then top triangle."""
    b, t = v[:3], v[3:]
    faces = [list(b), list(t)] + [[b[i], b[(i + 1) % 3], t[(i + 1) % 3], t[i]] for i in range(3)]
    cells = []
    for f in faces:
        cells += sample_polygon(f, res)
    return cells


def load_meshes(paths):
    meshes = {}
    for p in paths:
        for m in ET.parse(p).getroot().iter("mesh"):
            if m.get("vertex"):
                meshes[m.get("name")] = vec(m.get("vertex")).reshape(-1, 3)
    return meshes


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("worldbody", help="include file with <body> elements")
    ap.add_argument("--assets", nargs="*", default=[], help="files containing <mesh vertex=...>")
    ap.add_argument("--out", required=True, help="worldbody file with taxel sites added")
    ap.add_argument("--sensors", required=True, help="file of <touch> sensors to include in <sensor>")
    args = ap.parse_args()

    meshes = load_meshes(args.assets)
    tree = ET.parse(args.worldbody)
    root = tree.getroot()
    names, total = [], 0

    for body in list(root.iter("body")):
        for gi, g in enumerate(list(body.findall("geom"))):
            gname = g.get("name") or f"{body.get('name', 'body')}_g{gi}"
            gtype = g.get("type", "sphere")
            if gtype == "plane":
                continue
            if any(k in g.attrib for k in ("quat", "euler", "axisangle", "xyaxes", "zaxis")):
                print(f"[skip] {gname}: rotated geoms are not supported")
                continue
            size = vec(g.get("size"))
            offset = vec(g.get("pos"), np.zeros(3))
            rule = rule_for(gname)

            if gtype == "box":
                cells = box_cells(size, rule["res"], rule["faces"])
            elif gtype == "cylinder":
                cells = cylinder_cells(size[0], size[1], rule["res"], rule["faces"])
            elif gtype == "sphere":
                cells = sphere_cells(size[0], rule["res"])
            elif gtype == "mesh" and g.get("mesh") in meshes and len(meshes[g.get("mesh")]) == 6:
                cells = prism_cells(meshes[g.get("mesh")], rule["res"])
            else:
                print(f"[skip] {gname}: unsupported geom type '{gtype}'")
                continue

            for k, (c, xa, ya, sx, sy) in enumerate(cells):
                name = f"tx_{gname}_{k}"
                ET.SubElement(body, "site", {
                    "name": name,
                    "type": "box",
                    "pos": fmt(c + offset),
                    "size": f"{sx / 2:.5f} {sy / 2:.5f} {DEPTH}",
                    "xyaxes": fmt(np.concatenate([xa, ya])),
                    "group": "4",                      # hidden in the default viewer
                    "rgba": "0 1 0 0.15",
                })
                names.append(name)
            total += len(cells)
            if len(cells) > 50:
                print(f"{gname:28s} {len(cells):6d} taxels")

    ET.indent(tree)
    tree.write(args.out)

    sroot = ET.Element("mujoco")
    for n in names:
        ET.SubElement(sroot, "touch", {"name": n, "site": n})
    stree = ET.ElementTree(sroot)
    ET.indent(stree)
    stree.write(args.sensors)
    print(f"total: {total} taxels -> {args.out}, {args.sensors}")


if __name__ == "__main__":
    main()