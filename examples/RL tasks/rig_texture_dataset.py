"""
The 3D Printable Tactile Dataset's benchmark on the simulated rig: press the sensor onto each
texture block on a 5 x 5 grid (5 x 5 mm, two presses per point, 1.9 N), then train and test a
classifier on the recordings - the paper's protocol, with any simulated sensor.

    python "examples/RL tasks/rig_texture_dataset.py"                       # TacTip markers
    python "examples/RL tasks/rig_texture_dataset.py" --readout tactile_image
    python "examples/RL tasks/rig_texture_dataset.py" --sensor-xml presstip           # pads, piezo, accel
    python "examples/RL tasks/rig_texture_dataset.py" --sensor-xml gelsight --readout tactile_image
    python "examples/RL tasks/rig_texture_dataset.py" --sensor-xml tactip_depth       # contact depth grid
    python "examples/RL tasks/rig_texture_dataset.py" --sensor-xml my_sensor.xml --readout contact \\
        --sensors '{"contact": "examples.custom_sensor.pad_contact:PadContact"}'
    python "examples/RL tasks/rig_texture_dataset.py" --print-blur 0.2 --print-noise 0.03

The classifier is the paper's (PCA, then a 50-tree random forest; scikit-learn) or, without
scikit-learn, a nearest-centroid one. The data is saved to --out (.npz) for other classifiers.
"""
import argparse
import json

import gymnasium as gym
import numpy as np

import tactip_mujoco_gym  # noqa: F401
from tactip_mujoco_gym.envs.tactile_rig import press_grid


def classify(X, y, test_fraction=0.3, seed=0):
    """Accuracy on a held-out split, and the classifier's name."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    n_test = max(1, int(round(test_fraction * len(y))))
    te, tr = idx[:n_test], idx[n_test:]
    X = X.reshape(len(X), -1).astype(np.float64)
    try:
        from sklearn.decomposition import PCA
        from sklearn.ensemble import RandomForestClassifier
        k = int(min(300, len(tr) - 1, X.shape[1]))
        pca = PCA(n_components=k).fit(X[tr])
        clf = RandomForestClassifier(n_estimators=50, random_state=seed).fit(pca.transform(X[tr]), y[tr])
        return float(clf.score(pca.transform(X[te]), y[te])), f"PCA ({k}) + random forest"
    except ImportError:
        classes = np.unique(y[tr])
        mu = np.stack([X[tr][y[tr] == c].mean(axis=0) for c in classes])
        pred = classes[np.argmin(((X[te][:, None, :] - mu[None]) ** 2).sum(-1), axis=1)]
        return float((pred == y[te]).mean()), "nearest centroid (pip install scikit-learn for the paper's)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--textures", default="1,2,3,4,5,6", help="dataset textures (1..20) or STL files")
    ap.add_argument("--readout", default="tactile", help="the input recorded at each press")
    ap.add_argument("--force", type=float, default=1.9, help="press force, N (paper: 1.9)")
    ap.add_argument("--points", type=int, default=5)
    ap.add_argument("--span", type=float, default=5.0, help="grid width, mm")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--tilt", type=float, default=0.0, help="plate tilt, rad")
    ap.add_argument("--sensor-xml", default=None)
    ap.add_argument("--sensor-stiffness", type=float, default=1.0)
    ap.add_argument("--sensors", default=None, help="readouts as JSON {key: spec}")
    ap.add_argument("--print-blur", type=float, default=0.0, help="print smoothing sigma, mm")
    ap.add_argument("--print-noise", type=float, default=0.0, help="print roughness sd, mm")
    ap.add_argument("--out", default="rig_texture_dataset.npz")
    a = ap.parse_args()

    textures = [int(t) if t.strip().isdigit() else t.strip() for t in a.textures.split(",")]
    kw = dict(textures=textures, sensor_xml=a.sensor_xml, sensor_stiffness=a.sensor_stiffness,
              overrides={"PRINT_BLUR_MM": a.print_blur, "PRINT_NOISE_MM": a.print_noise})
    if a.sensors:
        kw["sensors"] = json.loads(a.sensors)
    env = gym.make("TacRig-v0", **kw)
    data = press_grid(env, points=a.points, span_mm=(a.span, a.span), repeats=a.repeats,
                      force=a.force, readout=a.readout, tilt=a.tilt)
    env.close()
    np.savez_compressed(a.out, X=data["X"], y=data["y"], pos=data["pos"], force=data["force"],
                        textures=np.array([str(t) for t in data["textures"]]))
    acc, name = classify(data["X"], data["y"])
    print(f"\n{len(data['y'])} presses ({a.readout}), force {data['force'].mean():.2f} "
          f"+- {data['force'].std():.2f} N; saved to {a.out}")
    print(f"{name}: {acc:.1%} on a held-out 30% (chance {1 / len(textures):.1%})")


if __name__ == "__main__":
    main()
