import itertools
import numpy as np

# ---- shared constants (the env imports these too) ----
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


def prism_polygon(R):
    """Equilateral triangle centred on the origin, one vertex pointing +x (yaw = 0)."""
    return [(R * np.cos(a), R * np.sin(a)) for a in np.deg2rad([0, 120, 240])]


def poly_interval(poly, y):
    xs = []
    for i in range(len(poly)):
        (x1, y1), (x2, y2) = poly[i], poly[(i + 1) % len(poly)]
        if (y1 <= y < y2) or (y2 <= y < y1):
            xs.append(x1 + (y - y1) / (y2 - y1) * (x2 - x1))
    return (min(xs), max(xs)) if len(xs) >= 2 else None


def hole_interval(spec, y):
    """x-extent of the hole (relative to hole centre) at height y, or None if solid."""
    c = CLEARANCE
    if spec["kind"] == "cylinder":
        rr = spec["r"] + c
        return None if abs(y) >= rr else (-np.sqrt(rr**2 - y**2), np.sqrt(rr**2 - y**2))
    if spec["kind"] == "box":
        return None if abs(y) >= spec["hy"] + c else (-(spec["hx"] + c), spec["hx"] + c)
    # triangle: offsetting by c means the circumradius grows by 2c (inradius = R/2)
    return poly_interval(prism_polygon(spec["R"] + 2 * c), y)


def _same(a, b):
    if a is None or b is None:
        return a is b
    return abs(a[0] - b[0]) < 1e-9 and abs(a[1] - b[1]) < 1e-9


def write_xml(asset_path="shape_assets.xml", world_path="shape_worldbody.xml"):
    # ---------- assets: triangle mesh ----------
    a = ["<mujoco>"]
    for name, s in SHAPES.items():
        if s["kind"] == "prism":
            poly = prism_polygon(s["R"])
            v = [c for (x, y) in poly for c in (x, y, -OBJ_HALF_H)]
            v += [c for (x, y) in poly for c in (x, y, OBJ_HALF_H)]
            a.append(f'  <mesh name="{name}_mesh" vertex="{" ".join(f"{c:.5f}" for c in v)}"/>')
    a.append("</mujoco>")
    open(asset_path, "w").write("\n".join(a))

    # ---------- worldbody: platform with holes + objects ----------
    zc = PLATE_TOP - PLATE_HALF_T
    ids = itertools.count()

    def box(x0, x1, y0, y1):
        return (f'    <geom name="plat_{next(ids)}" type="box" '
                f'pos="{(x0+x1)/2:.5f} {(y0+y1)/2:.5f} {zc:.5f}" '
                f'size="{(x1-x0)/2:.5f} {(y1-y0)/2:.5f} {PLATE_HALF_T}" {PLATE_ATTR}/>')

    y_min, y_max = HOLE_Y[0] - CELL_HALF, HOLE_Y[-1] + CELL_HALF
    bx0, bx1 = HOLE_X - CELL_HALF, HOLE_X + CELL_HALF

    w = ["<mujoco>", '  <body name="platform">']
    w.append(box(PLATE_X[0], bx0, y_min, y_max))      # solid slab before the hole band
    w.append(box(bx1, PLATE_X[1], y_min, y_max))      # solid slab after it

    n = int(round(2 * CELL_HALF / STRIP))
    for i, (name, s) in enumerate(SHAPES.items()):
        yc = HOLE_Y[i]
        merged = []                                   # [y_start, y_end, interval]
        for k in range(n):
            ys = -CELL_HALF + k * STRIP
            iv = hole_interval(s, ys + STRIP / 2)
            if merged and _same(merged[-1][2], iv):
                merged[-1][1] = ys + STRIP
            else:
                merged.append([ys, ys + STRIP, iv])
        for ys, ye, iv in merged:
            if iv is None:
                w.append(box(bx0, bx1, yc + ys, yc + ye))
            else:
                w.append(box(bx0, HOLE_X + iv[0], yc + ys, yc + ye))
                w.append(box(HOLE_X + iv[1], bx1, yc + ys, yc + ye))
        w.append(f'    <site name="hole_{name}" pos="{HOLE_X} {yc} {PLATE_TOP}" size="0.003" rgba="0 1 0 0.6"/>')
    w.append("  </body>")

    for i, (name, s) in enumerate(SHAPES.items()):
        if s["kind"] == "cylinder":
            g = f'type="cylinder" size="{s["r"]} {OBJ_HALF_H}"'
        elif s["kind"] == "box":
            g = f'type="box" size="{s["hx"]} {s["hy"]} {OBJ_HALF_H}"'
        else:
            g = f'type="mesh" mesh="{name}_mesh"'
        w.append(f'  <body name="shape_{name}" pos="0.35 {-0.09 + 0.06*i:.2f} {PLATE_TOP + OBJ_HALF_H + 0.001}">')
        w.append(f'    <freejoint name="shape_{name}_free"/>')
        w.append(f'    <geom name="shape_{name}_geom" {g} mass="0.05" rgba="{s["rgba"]}" '
                 f'friction="0.6 0.005 0.0001" condim="3"/>')
        w.append("  </body>")
    w.append("</mujoco>")
    open(world_path, "w").write("\n".join(w))


if __name__ == "__main__":
    write_xml()