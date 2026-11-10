"""
Watch what the sensors read, live: a window per input ("tactile", "tactile_image", "vision",
...) with a panel per mounted sensor, next to the MuJoCo viewer of the robot doing its
scripted task (boot_gyms.py's policies).

    python examples/demos/view_sensors.py TacRig-v0                          # the TacTip
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml presstip    # pads, piezo, accel
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml gelsight    # RGB + depth grid
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml tactip_depth
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml whiskers
    python examples/demos/view_sensors.py TacGripper-pick --sensor-xml gelsight   # two fingers
    python examples/demos/view_sensors.py TacArm-surface --sensor-xml whiskers --inputs tactile
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml gelsight \\
        --sensors '{"tactile_image": {"mode": "depth"}}'                    # change a readout
    python examples/demos/view_sensors.py TacRig-v0 --sensor-xml presstip --save presstip.png --steps 120

What each panel shows:
    tactip_markers   marker displacement (arrows, x-y; colour: z), mm
    presstip         force on each pad (N), the piezo signal, acceleration (g)
    whiskers         whiskers from above with their bend (arrows; red: touching), vibration
    contact_depth    the contact depth image (or GelSight RGB) of each sensor
    contact_map      the same on the coarse grid the policy gets (mm)
    images           as they are; anything else: a bar chart of its values

--inputs picks inputs (default: every sensor input the robot has); --one-window puts them
together; --every N redraws every N steps (drawing is slower than simulating); --no-viewer
leaves out the MuJoCo viewer; --save FILE runs --steps steps without windows and saves the
figure(s). On macOS the MuJoCo viewer needs mjpython, which matplotlib windows don't run
under: use --no-viewer there (or watch the viewer alone with boot_gyms.py --view).
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boot_gyms import env_dt, make, make_policy  # noqa: E402

from tactip_mujoco_gym.sensors import (Camera, ContactDepth, ContactMap, PressTip,  # noqa: E402
                                       TacTipCamera, TacTipMarkers, Whiskers)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"


def _style(ax, title=None, grid=False):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK2)
    ax.tick_params(labelsize=7, colors=INK2, length=2)
    if grid:
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")


def _bare(ax, title=None):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title, fontsize=8, color=INK, loc="left")


def _split(value, counts):
    return np.split(np.asarray(value), np.cumsum(counts)[:-1]) if len(counts) > 1 else [np.asarray(value)]


# ----------------------------------------------------------------------------- panels
class Panel:
    """One input: axes in a slot of a figure, redrawn from each new value."""

    def __init__(self, fig, slot, key, readout, value, mounts):
        self.fig, self.key, self.readout, self.mounts = fig, key, readout, mounts
        self.build(slot, value)

    def build(self, slot, value):
        raise NotImplementedError

    def update(self, value):
        raise NotImplementedError


class ImagePanel(Panel):
    """uint8 / float images, (H, W, C): split into RGB or single-channel images per mount."""

    def build(self, slot, value):
        v = np.asarray(value)
        if v.ndim == 2:
            v = v[..., None]
        r = self.readout
        rgb = ((isinstance(r, ContactDepth) and r.mode == "gelsight")
               or (isinstance(r, TacTipCamera) and r.mode == "rgb")
               or (isinstance(r, Camera) and getattr(r, "_mode", "rgb") == "rgb")
               or (r is None and v.shape[-1] % 3 == 0 and v.dtype == np.uint8))
        self.group = 3 if rgb and v.shape[-1] % 3 == 0 else 1
        n = v.shape[-1] // self.group
        names = [m.name for m in self.mounts] if n == len(self.mounts) and n > 1 else [""] * n
        if v.dtype == np.uint8:
            self.lim = (0, 255)
        elif isinstance(r, Camera):
            self.lim = (0.0, float(getattr(r, "_max_depth", 1.0)))
        else:
            self.lim = (float(v.min()), float(v.max()) or 1.0)
        gs = slot.subgridspec(1, n, wspace=0.08)
        self.ims = []
        for i in range(n):
            ax = self.fig.add_subplot(gs[0, i])
            _bare(ax, names[i] or None)
            img = v[..., i * self.group:(i + 1) * self.group]
            if self.group == 1:
                self.ims.append(ax.imshow(img[..., 0], cmap="viridis", vmin=self.lim[0],
                                          vmax=self.lim[1], interpolation="nearest"))
            else:
                self.ims.append(ax.imshow(img, interpolation="nearest"))

    def update(self, value):
        v = np.asarray(value)
        if v.ndim == 2:
            v = v[..., None]
        for i, im in enumerate(self.ims):
            img = v[..., i * self.group:(i + 1) * self.group]
            im.set_data(img[..., 0] if self.group == 1 else img)


class ContactMapPanel(Panel):
    def build(self, slot, value):
        r = self.readout
        n = len(self.mounts)
        gs = slot.subgridspec(1, n, wspace=0.08)
        self.ims = []
        for k, part in enumerate(_split(value, [r.grid[0] * r.grid[1]] * n)):
            ax = self.fig.add_subplot(gs[0, k])
            _bare(ax, f"{self.mounts[k].name}: contact depth (mm)" if n > 1 else "contact depth (mm)")
            self.ims.append(ax.imshow(part.reshape(r.grid), cmap="viridis", vmin=0.0,
                                      vmax=r.max_depth * 1000, interpolation="nearest"))

    def update(self, value):
        r = self.readout
        for im, part in zip(self.ims, _split(value, [r.grid[0] * r.grid[1]] * len(self.ims)), strict=True):
            im.set_data(part.reshape(r.grid))


class MarkerPanel(Panel):
    """TacTip markers: displacement arrows at their rest positions (x-y), colour = z."""

    def build(self, slot, value):
        r = self.readout
        n = len(r.counts)
        gs = slot.subgridspec(1, n, wspace=0.15)
        self.rest = _split(r.rest * 1000.0, r.counts)
        self.quivers, self.dots = [], []
        for k in range(n):
            ax = self.fig.add_subplot(gs[0, k])
            _style(ax, f"{self.mounts[k].name}: markers (mm)" if n > 1 else "marker displacement (mm)")
            p = self.rest[k]
            self.dots.append(ax.scatter(p[:, 0], p[:, 1], c=np.zeros(len(p)), cmap="coolwarm",
                                        vmin=-2, vmax=2, s=10, zorder=2))
            self.quivers.append(ax.quiver(p[:, 0], p[:, 1], np.zeros(len(p)), np.zeros(len(p)),
                                          color=INK, angles="xy", scale_units="xy", scale=0.2,
                                          width=0.006, zorder=3))
            ax.set_aspect("equal")
            pad = 0.15 * max(np.ptp(p[:, 0]), np.ptp(p[:, 1]), 1.0)
            ax.set_xlim(p[:, 0].min() - pad, p[:, 0].max() + pad)
            ax.set_ylim(p[:, 1].min() - pad, p[:, 1].max() + pad)
        self.fig.text(0.01, 0.01, "arrows x5; colour: z (mm)", fontsize=6, color=INK2)

    def update(self, value):
        r = self.readout
        disp = np.asarray(value)[:r.n_markers * 3].reshape(-1, 3)
        for k, d in enumerate(_split(disp, r.counts)):
            self.quivers[k].set_UVC(d[:, 0], d[:, 1])
            self.dots[k].set_array(d[:, 2])


class PressTipPanel(Panel):
    """Per PressTip: pad forces on the board, the piezo trace, the accelerometer."""

    def build(self, slot, value):
        from matplotlib.patches import Circle
        r = self.readout
        n = len(r.layout)
        gs = slot.subgridspec(3, n, height_ratios=[2.2, 1, 1], hspace=0.6, wspace=0.35)
        self.pads, self.traces, self.bars, self.trace_axes, self.texts = [], [], [], [], []
        top = 2 * r.f_ref
        for k in range(n):
            name = f"{self.mounts[k].name}: " if n > 1 else ""
            ax = self.fig.add_subplot(gs[0, k])
            _style(ax, f"{name}pad force (N)")
            xy = r.layout[k] * 1000.0
            self.pads.append(ax.scatter(xy[:, 0], xy[:, 1], c=np.zeros(len(xy)), cmap="Blues",
                                        vmin=0.0, vmax=top, marker="s", s=150, edgecolors=INK2,
                                        linewidths=0.5))
            self.texts.append([ax.text(x, y, "", ha="center", va="center", fontsize=5.5, color=INK)
                               for x, y in xy])
            ax.set_aspect("equal")
            lim = 1.25 * np.abs(xy).max() + 3
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.add_patch(Circle((0, 0), lim - 1.5, fill=False, color=GRID, linewidth=1.0))
            ax2 = self.fig.add_subplot(gs[1, k])
            _style(ax2, f"{name}piezo (N, high-passed)", grid=True)
            (line,) = ax2.plot([], [], color=ORANGE, linewidth=0.8)
            ax2.set_xlim(0, r.piezo_trace[k].maxlen)
            ax2.set_xticks([])
            self.traces.append(line)
            self.trace_axes.append(ax2)
            ax3 = self.fig.add_subplot(gs[2, k])
            _style(ax3, f"{name}acceleration (g)", grid=True)
            self.bars.append(ax3.bar(["x", "y", "z"], [0, 0, 1], color=[BLUE, GREEN, ORANGE]))
            ax3.set_ylim(-2, 2)
            ax3.axhline(0, color=INK2, linewidth=0.6)
        self.update(value)

    def update(self, value):
        r = self.readout
        for k in range(len(r.layout)):
            f = np.asarray(r.pads[k])
            self.pads[k].set_array(f)
            for t, x in zip(self.texts[k], f, strict=True):
                t.set_text(f"{x:.1f}" if x >= 0.05 else "")
            tr = np.asarray(r.piezo_trace[k])
            self.traces[k].set_data(np.arange(len(tr)), tr)
            m = float(np.abs(tr).max()) if len(tr) else 0.0
            self.trace_axes[k].set_ylim(-max(m, 0.01) * 1.1, max(m, 0.01) * 1.1)
            for b, a in zip(self.bars[k], r.accel[k], strict=True):
                b.set_height(float(a))


class WhiskerPanel(Panel):
    """Per whisker pad: the whiskers from above (rest, light) with their tip deflection
    from the base bend (arrows; red while touching), and each whisker's vibration."""

    def build(self, slot, value):
        import matplotlib.collections as mc
        r = self.readout
        n = len(r.layout)
        gs = slot.subgridspec(2, n, height_ratios=[2.2, 1], hspace=0.45, wspace=0.3)
        self.arrows, self.vib, self.vib_axes, self.idx = [], [], [], []
        start = 0
        for k in range(n):
            lay = r.layout[k]
            b, t = lay["base"] * 1000.0, lay["tip"] * 1000.0
            self.idx.append(np.arange(start, start + len(b)))
            start += len(b)
            name = f"{self.mounts[k].name}: " if n > 1 else ""
            ax = self.fig.add_subplot(gs[0, k])
            _style(ax, f"{name}whiskers from above (mm)")
            ax.add_collection(mc.LineCollection(np.stack([b[:, :2], t[:, :2]], axis=1), colors=GRID,
                                                linewidths=1.5))
            ax.scatter(b[:, 0], b[:, 1], s=8, color=INK2, zorder=3)
            self.arrows.append(ax.quiver(t[:, 0], t[:, 1], np.zeros(len(t)), np.zeros(len(t)),
                                         color=[BLUE] * len(t), angles="xy", scale_units="xy",
                                         scale=1.0, width=0.008, zorder=4))
            pts = np.vstack([b[:, :2], t[:, :2]])
            pad = 8.0
            ax.set_xlim(pts[:, 0].min() - pad, pts[:, 0].max() + pad)
            ax.set_ylim(pts[:, 1].min() - pad, pts[:, 1].max() + pad)
            ax.set_aspect("equal")
            ax2 = self.fig.add_subplot(gs[1, k])
            _style(ax2, f"{name}vibration (rad/s)", grid=True)
            self.vib.append(ax2.bar(np.arange(len(b)), np.zeros(len(b)), color=ORANGE))
            ax2.set_xticks(np.arange(len(b)))
            ax2.set_xticklabels([str(i) for i in range(len(b))], fontsize=6)
            ax2.set_ylim(0, 1)
            self.vib_axes.append(ax2)
        self.lengths = [np.linalg.norm(lay["tip"] - lay["base"], axis=1) * 1000.0 for lay in r.layout]
        self.update(value)

    def update(self, value):
        r = self.readout
        for k, sel in enumerate(self.idx):
            theta = r.bend[sel] * r.angle_ref                  # rad about the base x / y axes
            L = self.lengths[k]
            # a turn about x swings the whisker towards -y, about y towards +x
            self.arrows[k].set_UVC(theta[:, 1] * L, -theta[:, 0] * L)
            self.arrows[k].set_color([RED if c else BLUE for c in r.contact[sel]])
            v = r.vibration[sel]
            for bar, x in zip(self.vib[k], v, strict=True):
                bar.set_height(float(x))
            self.vib_axes[k].set_ylim(0, max(1.0, float(v.max()) * 1.2))


class VectorPanel(Panel):
    """Anything else: a bar chart of the values."""

    def build(self, slot, value):
        v = np.asarray(value, dtype=float).ravel()
        ax = self.fig.add_subplot(slot)
        _style(ax, f"{self.key} ({v.size} values)", grid=True)
        self.bars = ax.bar(np.arange(v.size), v, color=BLUE, width=0.85)
        self.ax = ax
        self.top = max(1e-3, float(np.abs(v).max()))

    def update(self, value):
        v = np.asarray(value, dtype=float).ravel()
        for b, x in zip(self.bars, v, strict=True):
            b.set_height(x)
        self.top = max(self.top * 0.98, float(np.abs(v).max()), 1e-3)
        self.ax.set_ylim(-1.1 * self.top if v.min() < 0 else 0, 1.1 * self.top)


def panel_for(readout, value):
    if isinstance(readout, PressTip):
        return PressTipPanel
    if isinstance(readout, Whiskers):
        return WhiskerPanel
    if isinstance(readout, ContactMap):
        return ContactMapPanel
    if isinstance(readout, TacTipMarkers) and readout.n_markers:
        return MarkerPanel
    v = np.asarray(value)
    if v.ndim == 3 or isinstance(readout, (ContactDepth, TacTipCamera, Camera)):
        return ImagePanel
    return VectorPanel


# ----------------------------------------------------------------------------- windows
def describe(readout, key):
    if readout is None:
        return key
    return f"{key}: {type(readout).__name__}"


def build_windows(raw, obs, keys, one_window):
    import matplotlib.pyplot as plt
    groups = [keys] if one_window else [[k] for k in keys]
    windows = []
    for group in groups:
        heights = [_height(raw.sensors.get(k), obs[k]) for k in group]
        fig = plt.figure(figsize=(_width(raw), sum(heights) + 0.4), facecolor=SURFACE)
        title = " + ".join(describe(raw.sensors.get(k), k) for k in group)
        try:
            fig.canvas.manager.set_window_title(title)
        except Exception:                                           # noqa: BLE001
            pass
        gs = fig.add_gridspec(len(group), 1, height_ratios=heights, hspace=0.35,
                              left=0.06, right=0.98, top=0.9, bottom=0.07)
        status = fig.suptitle(title, fontsize=9, color=INK, x=0.01, ha="left")
        panels = [panel_for(raw.sensors.get(k), obs[k])(fig, gs[i], k, raw.sensors.get(k), obs[k],
                                                        raw.sensor_mounts)
                  for i, k in enumerate(group)]
        windows.append((fig, panels, status, title))
    return windows


def _height(readout, value):
    if isinstance(readout, (PressTip, Whiskers)):
        return 5.0
    return 2.8 if np.asarray(value).ndim >= 2 or isinstance(readout, (TacTipMarkers, ContactMap)) else 2.0


def _width(raw):
    n = max(1, len(raw.sensor_mounts))
    return min(16.0, max(5.5, 3.2 * n + 1.0))


def choose_inputs(raw, wanted):
    avail = {k: ok for k, (ok, _) in raw.describe_inputs().items()}
    if wanted:
        bad = [k for k in wanted if not avail.get(k, False)]
        if bad:
            why = {k: raw.describe_inputs().get(k, (False, "no such input"))[1] for k in bad}
            raise SystemExit(f"unavailable inputs: {why}\navailable: {[k for k, ok in avail.items() if ok]}")
        return list(wanted)
    keys = [k for k in raw.sensors if avail.get(k)]
    if not keys:
        raise SystemExit("this robot has no sensor inputs available")
    return keys


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task", nargs="?", default="TacRig-v0")
    ap.add_argument("--sensor-xml", default=None, help="sensor hardware: tactip, presstip, gelsight, "
                    "tactip_depth, whiskers, or an MJCF file")
    ap.add_argument("--sensor-stiffness", type=float, default=None)
    ap.add_argument("--sensors", default=None, help="readouts as JSON, e.g. '{\"tactile_image\": {\"mode\": \"depth\"}}'")
    ap.add_argument("--inputs", nargs="*", default=None, help="inputs to show (default: every sensor input)")
    ap.add_argument("--kwargs", default="{}", help="task options as JSON")
    ap.add_argument("--policy", default="scripted", choices=("scripted", "random", "hold"))
    ap.add_argument("--one-window", action="store_true", help="every input in one window")
    ap.add_argument("--every", type=int, default=2, help="redraw every N env steps")
    ap.add_argument("--no-viewer", action="store_true", help="no MuJoCo viewer")
    ap.add_argument("--steps", type=int, default=0, help="stop after this many steps (0: run until closed)")
    ap.add_argument("--save", default=None, help="no windows: run --steps (default 100), save the figure(s)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    import matplotlib
    if a.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kwargs = json.loads(a.kwargs)
    if a.sensor_xml is not None:
        kwargs["sensor_xml"] = a.sensor_xml
    if a.sensor_stiffness is not None:
        kwargs["sensor_stiffness"] = a.sensor_stiffness
    if a.sensors:
        kwargs["sensors"] = json.loads(a.sensors)
    env = make(a.task, None, kwargs)
    raw = env.unwrapped
    keys = choose_inputs(raw, a.inputs)
    raw.set_obs_format("dict")
    raw.set_obs_modes(keys)
    obs, _ = env.reset(seed=a.seed)
    print(f"{a.task} with {a.sensor_xml or 'tactip'}: " + ", ".join(
        f"{k} {np.shape(obs[k])} ({type(raw.sensors[k]).__name__})" for k in keys))
    act = make_policy(a.task, env, a.policy)
    windows = build_windows(raw, obs, keys, a.one_window)
    steps = a.steps or (100 if a.save else 0)

    viewer = None
    if not (a.no_viewer or a.save):
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(raw.model, raw.data)
    sync = getattr(raw, "sync_renderer", None)
    if not a.save:
        plt.ion()
        plt.show(block=False)
    dt = env_dt(raw)
    step, episode, R = 0, 0, 0.0
    try:
        while True:
            t0 = time.time()
            obs, r, term, trunc, info = env.step(act())
            step += 1
            R += float(r)
            if step % max(1, a.every) == 0 or (a.save and step == steps):
                for fig, panels, status, title in windows:
                    for p in panels:
                        p.update(obs[p.key])
                    status.set_text(f"{title}    step {raw.step_count}  return {R:.1f}")
                    if not a.save:
                        fig.canvas.draw_idle()
                        fig.canvas.flush_events()
                if not a.save:
                    plt.pause(0.001)
            if term or trunc:
                episode += 1
                why = "success" if info.get("is_success") else (info.get("fail_reason") or "time limit")
                print(f"episode {episode}: return {R:.1f}, {why}")
                R = 0.0
                env.reset()
            if viewer is not None:
                if not viewer.is_running():
                    break
                if sync is not None:
                    sync(viewer)
                viewer.sync()
            if steps and step >= steps:
                break
            if not a.save and not any(plt.fignum_exists(w[0].number) for w in windows):
                break
            if viewer is not None:
                time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        if a.save:
            root, ext = os.path.splitext(a.save)
            for i, (fig, _, _, _) in enumerate(windows):
                path = a.save if len(windows) == 1 else f"{root}_{keys[i] if not a.one_window else i}{ext or '.png'}"
                fig.savefig(path, dpi=150, facecolor=SURFACE)
                print(f"saved {path}")
        if viewer is not None:
            viewer.close()
        env.close()


if __name__ == "__main__":
    main()
