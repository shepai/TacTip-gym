"""
Evaluation results of an experiment config: one row per test episode ("trial") in
results/<name>/episodes.csv, and from those a summary per task and variant (success rate with
a 95% CI, return, length) in summary.csv, a table, and a chart (success.png).

    python -m tactip_mujoco_gym evaluate configs/sensing_trials.yaml   # writes all of it
    python -m tactip_mujoco_gym summary configs/sensing_trials.yaml    # table + chart again

95% CIs: with one training seed, the Wilson interval of the success rate over the trials;
with several, a t-interval over the per-seed success rates (training runs usually differ more
than episodes do).
"""
import csv
import math
import os

FIELDS = ["run", "task", "variant", "seed", "trial", "success", "return", "length", "fail_reason"]
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")


def results_dir(name):
    path = os.path.join("results", name)
    os.makedirs(path, exist_ok=True)
    return path


def read_episodes(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def replace_episodes(path, run, rows):
    """Write `rows` for `run` into the CSV at `path`, dropping that run's earlier rows."""
    keep = [r for r in read_episodes(path) if r["run"] != run]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(keep + rows)


# ----------------------------------------------------------------------------- statistics
def wilson(k, n, z=1.96):
    """95% Wilson score interval of k successes in n trials."""
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
         9: 2.262, 10: 2.228, 12: 2.179, 15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042}


def t975(df):
    """Two-sided 95% t critical value (conservative between tabulated df)."""
    return _T975[max(k for k in _T975 if k <= df)] if df <= 30 else 1.96


def summarise(rows):
    """[{task, variant, seeds, trials, success, ci_low, ci_high, return_mean, return_sd,
    length}] per (task, variant), in the order they first appear."""
    groups = {}
    for r in rows:
        groups.setdefault((r["task"], r["variant"]), []).append(r)
    out = []
    for (task, variant), sel in groups.items():
        seeds = sorted({int(r["seed"]) for r in sel})
        succ = [int(r["success"]) for r in sel]
        rets = [float(r["return"]) for r in sel]
        if len(seeds) > 1:
            per = []
            for s in seeds:
                x = [int(r["success"]) for r in sel if int(r["seed"]) == s]
                per.append(sum(x) / len(x))
            p = sum(per) / len(per)
            sd = math.sqrt(sum((v - p) ** 2 for v in per) / (len(per) - 1))
            h = t975(len(per) - 1) * sd / math.sqrt(len(per))
            lo, hi = max(0.0, p - h), min(1.0, p + h)
        else:
            p = sum(succ) / len(succ)
            lo, hi = wilson(sum(succ), len(succ))
        mr = sum(rets) / len(rets)
        sr = math.sqrt(sum((v - mr) ** 2 for v in rets) / max(1, len(rets) - 1))
        out.append(dict(task=task, variant=variant, seeds=len(seeds), trials=len(sel),
                        success=round(p, 4), ci_low=round(lo, 4), ci_high=round(hi, 4),
                        return_mean=round(mr, 3), return_sd=round(sr, 3),
                        length=round(sum(int(r["length"]) for r in sel) / len(sel), 1)))
    return out


# ----------------------------------------------------------------------------- output
def print_table(summary):
    tasks = list(dict.fromkeys(s["task"] for s in summary))
    variants = list(dict.fromkeys(s["variant"] for s in summary))
    cells = {(s["task"], s["variant"]): s for s in summary}
    w = max([16] + [len(t) for t in tasks])
    print("\n" + "success [95% CI]".ljust(w) + "".join(f"{v:>20}" for v in variants))
    for t in tasks:
        line = t.ljust(w)
        for v in variants:
            s = cells.get((t, v))
            cell = "-" if s is None else "{:.0%} [{:.0%}-{:.0%}]".format(
                s["success"], s["ci_low"], s["ci_high"])
            line += f"{cell:>20}"
        print(line)


def plot(summary, path, title=None):
    """Grouped bars: success rate per task (groups) and variant (bars), 95% CI whiskers.
    Variants keep their colour (and order) from the config."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("(pip install matplotlib for the chart)")
        return None
    surface, ink, ink2, grid = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    tasks = list(dict.fromkeys(s["task"] for s in summary))
    variants = list(dict.fromkeys(s["variant"] for s in summary))
    cells = {(s["task"], s["variant"]): s for s in summary}
    n = len(variants)
    slot = 0.8 / n
    fig, ax = plt.subplots(figsize=(max(6.0, 1.15 * len(tasks) + 1.5), 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    for j, v in enumerate(variants):
        xs, ys, lo, hi = [], [], [], []
        for i, t in enumerate(tasks):
            s = cells.get((t, v))
            if s is None:
                continue
            xs.append(i - 0.4 + slot * (j + 0.5))
            ys.append(s["success"])
            lo.append(s["success"] - s["ci_low"])
            hi.append(s["ci_high"] - s["success"])
        ax.bar(xs, ys, width=slot * 0.72, color=COLORS[j % len(COLORS)], label=v, zorder=2)
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=ink2, elinewidth=0.8, capsize=2,
                    capthick=0.8, zorder=3)
        for x, y, h in zip(xs, ys, hi, strict=True):
            ax.text(x, min(1.0, y + h) + 0.02, f"{y:.0%}", ha="center", va="bottom", fontsize=6,
                    color=ink)
    slanted = len(tasks) > 4
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=20 if slanted else 0, rotation_mode="anchor",
                       ha="right" if slanted else "center", fontsize=8, color=ink)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8, color=ink2)
    ax.yaxis.grid(True, color=grid, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(ink2)
    ax.tick_params(axis="both", length=0)
    ax.set_ylabel("success rate", fontsize=8, color=ink2)
    trials = sorted({s["trials"] // max(1, s["seeds"]) for s in summary})
    ax.set_title(title or f"Success over {'/'.join(map(str, trials))} trials per policy (95% CI)",
                 fontsize=10, color=ink, loc="left")
    ax.legend(frameon=False, ncol=n, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 1.12),
              labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)
    return path


def write_summary(out_dir, rows=None):
    """summary.csv, the table and success.png from out_dir/episodes.csv (or `rows`)."""
    rows = read_episodes(os.path.join(out_dir, "episodes.csv")) if rows is None else rows
    if not rows:
        print(f"no results in {out_dir} yet")
        return []
    summary = summarise(rows)
    with open(os.path.join(out_dir, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print_table(summary)
    png = plot(summary, os.path.join(out_dir, "success.png"))
    print(f"\nresults: {os.path.join(out_dir, 'episodes.csv')}, summary.csv"
          + (", success.png" if png else ""))
    return summary
