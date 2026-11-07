"""
Pretrain the tactile and vision encoders on datasets from collect_dataset.py.

    python train_encoders.py --data datasets/TacArm-peg --out encoders/peg_vt.pt
    python train_encoders.py --data datasets/TacArm-peg datasets/TacArm-hill datasets/TacArm-shape \
                             datasets/TacArm-edge --out encoders/all_vt.pt --align 0.1

One tactile encoder and one vision encoder are shared across every dataset given (same sensor,
same camera); each task gets its own small prediction heads. Objective, per task:

    tactile latent + proprio  -> privileged state    what can touch tell you?
    vision  latent + proprio  -> privileged state    what can the camera tell you?
    both latents   + proprio  -> privileged state    together
    + --align w: CLIP-style contrastive alignment of the tactile and vision latents of the same
      frame, on frames where the sensor is touching (no-contact tactile frames all look alike)

The privileged state is standardised per task, so every target dimension counts equally.
Validation is on held-out episodes. The R^2 table it prints (1 = predicts the privileged state
perfectly, 0 = no better than the mean) shows what each modality knows about each task.

Output: one checkpoint with the encoder weights + config (used by vt_encoders.py / RL_BASE.run),
the prediction heads, normalisation statistics and the validation metrics, plus a CSV log.
"""
import argparse
import csv
import json
import os
import time

import numpy as np

KEYS = ("proprio", "privileged", "tactile", "vision", "touching", "episode")


# ----------------------------------------------------------------------------- data (NumPy only)
def load_dataset(path, max_frames=None):
    shards = sorted(f for f in os.listdir(path) if f.endswith(".npz"))
    if not shards:
        raise FileNotFoundError(f"no .npz shards in {path} (run collect_dataset.py first)")
    parts = {k: [] for k in KEYS}
    n = 0
    for f in shards:
        d = np.load(os.path.join(path, f))
        for k in KEYS:
            parts[k].append(d[k])
        n += len(d["t"])
        if max_frames and n >= max_frames:
            break
    data = {k: np.concatenate(v)[:max_frames] if max_frames else np.concatenate(v)
            for k, v in parts.items()}
    meta_path = os.path.join(path, "meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    data["name"] = meta.get("env_name", os.path.basename(os.path.normpath(path)))
    return data


def split_by_episode(data, val_frac, rng):
    """Hold out whole episodes, so validation frames are not near-copies of training frames."""
    eps = np.unique(data["episode"])
    n_val = max(1, int(round(val_frac * len(eps)))) if len(eps) > 1 else 0
    val_eps = rng.choice(eps, size=n_val, replace=False) if n_val else np.array([], dtype=eps.dtype)
    is_val = np.isin(data["episode"], val_eps)
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def standardiser(x):
    mean = x.mean(0)
    std = np.maximum(x.std(0), 1e-3)
    return mean.astype(np.float32), std.astype(np.float32)


def random_shift_indices(n, h, w, pad, rng):
    """Row / column indices for a random crop of a (h + 2 pad, w + 2 pad) padded image."""
    i = rng.integers(0, 2 * pad + 1, size=n)
    j = rng.integers(0, 2 * pad + 1, size=n)
    return i[:, None] + np.arange(h)[None], j[:, None] + np.arange(w)[None]


def r2_score(pred, target):
    """Mean R^2 over target dimensions (targets standardised on the training set)."""
    ss_res = ((pred - target) ** 2).sum(0)
    ss_tot = ((target - target.mean(0)) ** 2).sum(0)
    keep = ss_tot > 1e-8
    return float(np.mean(1.0 - ss_res[keep] / ss_tot[keep])) if keep.any() else float("nan")


# ----------------------------------------------------------------------------- training (PyTorch)
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", nargs="+", required=True, help="dataset folders from collect_dataset.py")
    ap.add_argument("--out", default="encoders/visuotactile.pt")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--latent", type=int, default=64)
    ap.add_argument("--align", type=float, default=0.0, help="weight of the contrastive alignment loss")
    ap.add_argument("--temperature", type=float, default=0.1)
    ap.add_argument("--shift", type=int, default=4, help="random-shift augmentation (pixels, 0 = off)")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--max-frames", type=int, default=None, help="per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = ap.parse_args()

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from tactip_mujoco_gym.vt_encoders import DEFAULT_CONFIG, build_encoders

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)

    # ---------- data
    tasks = [load_dataset(p, args.max_frames) for p in args.data]
    tactile_dim = tasks[0]["tactile"].shape[1]
    img_shape = tasks[0]["vision"].shape[1:]                    # (H, W, C)
    vision_uint8 = tasks[0]["vision"].dtype == np.uint8
    for d in tasks:
        if d["tactile"].shape[1] != tactile_dim or d["vision"].shape[1:] != img_shape:
            raise ValueError("all datasets must come from the same sensor and camera settings")
        d["train"], d["val"] = split_by_episode(d, args.val_frac, rng)
        d["target_mean"], d["target_std"] = standardiser(d["privileged"][d["train"]])
        d["target"] = ((d["privileged"] - d["target_mean"]) / d["target_std"]).astype(np.float32)
        print(f"{d['name']:13s} {len(d['train']):7d} train / {len(d['val']):6d} val frames, "
              f"privileged dim {d['privileged'].shape[1]}, touching {100 * d['touching'].mean():.0f}%")
    tac_mean, tac_std = standardiser(np.concatenate([d["tactile"][d["train"]] for d in tasks]))

    # ---------- model
    cfg = dict(DEFAULT_CONFIG, latent_dim=args.latent)
    chw = (img_shape[2], img_shape[0], img_shape[1])
    tactile_enc, vision_enc = build_encoders(cfg, tactile_dim=tactile_dim, vision_shape=chw)
    tactile_enc.set_input_stats(tac_mean, tac_std)
    L = cfg["latent_dim"]

    def head(n_in, n_out):
        return nn.Sequential(nn.Linear(n_in, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                             nn.Linear(256, n_out))

    heads = nn.ModuleDict()
    for i, d in enumerate(tasks):
        p, k = d["proprio"].shape[1], d["privileged"].shape[1]
        heads[str(i)] = nn.ModuleDict({"tactile": head(L + p, k), "vision": head(L + p, k),
                                       "fused": head(2 * L + p, k)})
    proj = nn.ModuleDict({"tactile": nn.Linear(L, L), "vision": nn.Linear(L, L)})   # alignment space
    model = nn.ModuleDict({"tactile": tactile_enc, "vision": vision_enc, "heads": heads, "proj": proj}).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)

    def batch_tensors(d, idx, augment):
        t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(dev)  # noqa: E731
        img = d["vision"][idx]
        if augment and args.shift > 0:
            pad = args.shift
            img = np.pad(img, ((0, 0), (pad, pad), (pad, pad), (0, 0)), mode="edge")
            rows, cols = random_shift_indices(len(idx), img_shape[0], img_shape[1], pad, rng)
            img = img[np.arange(len(idx))[:, None, None], rows[:, :, None], cols[:, None, :]]
        x_img = t(img).permute(0, 3, 1, 2).float()
        if vision_uint8:
            x_img = x_img / 255.0                               # same as SB3's image preprocessing
        return (t(d["proprio"][idx]).float(), t(d["tactile"][idx]).float(), x_img,
                t(d["target"][idx]), t(d["touching"][idx]).bool())

    def forward(i, d, idx, augment):
        prop, tac, img, target, touch = batch_tensors(d, idx, augment)
        zt, zv = model["tactile"](tac), model["vision"](img)
        h = model["heads"][str(i)]
        preds = {"tactile": h["tactile"](torch.cat([zt, prop], 1)),
                 "vision": h["vision"](torch.cat([zv, prop], 1)),
                 "fused": h["fused"](torch.cat([zt, zv, prop], 1))}
        return preds, target, zt, zv, touch

    def info_nce(a, b):
        a, b = F.normalize(a, dim=1), F.normalize(b, dim=1)
        logits = a @ b.t() / args.temperature
        labels = torch.arange(a.shape[0], device=a.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))

    # ---------- train
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    log_path = os.path.splitext(args.out)[0] + "_log.csv"
    per_task = max(8, args.batch // len(tasks))
    steps = max(1, sum(len(d["train"]) for d in tasks) // (per_task * len(tasks)))
    best, best_metrics, rows = float("inf"), None, []
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, tot, tot_align = time.time(), 0.0, 0.0
        for _ in range(steps):
            loss = 0.0
            for i, d in enumerate(tasks):
                idx = rng.choice(d["train"], size=per_task)
                preds, target, zt, zv, touch = forward(i, d, idx, augment=True)
                loss = loss + sum(F.mse_loss(p, target) for p in preds.values())
                if args.align > 0 and int(touch.sum()) >= 8:
                    la = info_nce(model["proj"]["tactile"](zt[touch]), model["proj"]["vision"](zv[touch]))
                    loss = loss + args.align * la
                    tot_align += float(la)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss)

        # ---------- validate on held-out episodes
        model.eval()
        metrics, val_loss = {}, 0.0
        with torch.no_grad():
            for i, d in enumerate(tasks):
                if len(d["val"]) == 0:
                    continue
                outs = {k: [] for k in ("tactile", "vision", "fused")}
                targets = []
                for s in range(0, len(d["val"]), 512):
                    preds, target, *_ = forward(i, d, d["val"][s:s + 512], augment=False)
                    for k in outs:
                        outs[k].append(preds[k].cpu().numpy())
                    targets.append(target.cpu().numpy())
                y = np.concatenate(targets)
                for k in outs:
                    p = np.concatenate(outs[k])
                    metrics[f"{d['name']}/{k}_r2"] = r2_score(p, y)
                    val_loss += float(((p - y) ** 2).mean())
        line = " | ".join(f"{d['name']}: T {metrics.get(d['name'] + '/tactile_r2', np.nan):.2f} "
                          f"V {metrics.get(d['name'] + '/vision_r2', np.nan):.2f} "
                          f"T+V {metrics.get(d['name'] + '/fused_r2', np.nan):.2f}" for d in tasks)
        print(f"epoch {epoch:3d}  train {tot / steps:.3f}" + (f" (align {tot_align / steps:.3f})" if args.align else "")
              + f"  val {val_loss:.3f}  {time.time() - t0:.0f}s   R^2  {line}", flush=True)
        rows.append(dict(epoch=epoch, train_loss=tot / steps, val_loss=val_loss, **metrics))
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[-1].keys()))
            w.writeheader()
            w.writerows(rows)

        if val_loss < best or epoch == 1:
            best, best_metrics = val_loss, dict(metrics)
            ckpt = {
                # read by vt_encoders.py
                "config": dict(cfg, tactile_dim=int(tactile_dim), vision_shape=[int(s) for s in chw],
                               vision_uint8=bool(vision_uint8)),
                "tactile": {k: v.cpu() for k, v in model["tactile"].state_dict().items()},
                "vision": {k: v.cpu() for k, v in model["vision"].state_dict().items()},
                # for analysis / later use
                "heads": {k: v.cpu() for k, v in model["heads"].state_dict().items()},
                "proj": {k: v.cpu() for k, v in model["proj"].state_dict().items()},
                "tasks": [dict(name=d["name"], proprio_dim=int(d["proprio"].shape[1]),
                               privileged_dim=int(d["privileged"].shape[1]),
                               target_mean=d["target_mean"].tolist(),
                               target_std=d["target_std"].tolist()) for d in tasks],
                "metrics": {k: float(v) for k, v in best_metrics.items()},
                "epoch": int(epoch),
                "args": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                         for k, v in vars(args).items()},
            }
            torch.save(ckpt, args.out)
    print(f"best validation loss {best:.3f}; encoders saved to {args.out}, log in {log_path}")


def _cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
