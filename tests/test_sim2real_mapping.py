"""sim2real image mapping: paired sessions (as rig_dataset records them) of a synthetic TacTip,
the linear and torch backends, saving / loading, the CLI, and the "real_tactip" gym readout
(when MuJoCo is installed)."""
import json
import os

import numpy as np
import pytest

from tactip_mujoco_gym.sim2real import PairedData, Preprocess, load_mapping, train_mapping
from tactip_mujoco_gym.sim2real.paired import crop_check, resize_area


# ============================================================================ a synthetic sensor
def hex_layout(rings=3, pitch=2.0):
    pts = [(0.0, 0.0)]
    for r in range(1, rings + 1):
        for k in range(6 * r):
            a = 2 * np.pi * k / (6 * r)
            pts.append((r * pitch * np.cos(a), r * pitch * np.sin(a)))
    return np.array(pts)


def press(rest_xy, centre, depth):
    """Marker displacements (n, 3) mm for a press at `centre` (mm), `depth` mm."""
    d = rest_xy - centre
    r2 = (d ** 2).sum(axis=1)
    w = np.exp(-r2 / 8.0)
    out = np.zeros((len(rest_xy), 3))
    out[:, :2] = 0.4 * depth * d / np.sqrt(r2 + 1.0)[:, None] * w[:, None]
    out[:, 2] = -depth * w
    return out


def blobs(xy_mm, size, span=8.5, sigma=1.1):
    """A grey (size, size) picture of markers at xy_mm."""
    px = (xy_mm / span * 0.5 + 0.5) * (size - 1)
    g = np.arange(size, dtype=np.float32)
    gx = np.exp(-(g[None, :] - px[:, :1]) ** 2 / (2 * sigma ** 2))       # (n, size)
    gy = np.exp(-(g[None, :] - px[:, 1:2]) ** 2 / (2 * sigma ** 2))
    return np.clip(np.einsum("ny,nx->yx", gy, gx), 0, 1)


def fake_sessions(root, n_markers=37, sim_size=48, frame=(160, 120), motions=5, repeats=2,
                  steps=10, textures=(1, 2), seed=0, names=("session_a",)):
    """rig_dataset-style sessions: the sim image shows the markers; the real frame shows the
    same markers mirrored, recoloured, in the centre square of a wider BGR frame with noise
    either side. Returns the session folders."""
    rng = np.random.default_rng(seed)
    rings = 1
    while 1 + 3 * rings * (rings + 1) < n_markers:
        rings += 1
    rest_xy = hex_layout(rings)[:n_markers]
    rest = np.c_[rest_xy, np.zeros(n_markers)]
    W, H = frame
    out = []
    for name in names:
        sdir = os.path.join(root, name)
        rows = []
        for tex in textures:
            os.makedirs(os.path.join(sdir, f"texture_{tex:02d}"), exist_ok=True)
            for m in range(motions):
                c = rng.uniform(-4, 4, 2)
                depth = np.r_[np.zeros(2), np.linspace(0.3, 2.0, steps - 4), np.zeros(2)]
                phase = np.r_[np.zeros(2), np.ones(steps - 4), np.full(2, 2)].astype(np.int8)
                for r in range(repeats):
                    disp = np.stack([press(rest_xy, c, dd) for dd in depth]).astype(np.float32)
                    sim, real = [], []
                    for d in disp:
                        pos = rest_xy + d[:, :2]
                        g = blobs(pos, sim_size)
                        sim.append((np.stack([g, g, 0.6 * g], -1) * 230 + 15).astype(np.uint8))
                        gr = blobs(pos * [-1, 1], H)                       # mirrored, H x H
                        rgb = np.stack([0.8 * gr, 0.5 * gr, 0.2 * gr + 0.1], -1) * 255 + 20
                        f = rng.integers(0, 255, (H, W, 3)).astype(np.float32)
                        x0 = (W - H) // 2
                        f[:, x0:x0 + H] = rgb
                        real.append(np.clip(f, 0, 255).astype(np.uint8)[..., ::-1])   # BGR
                    k = len(rows)
                    fname = f"texture_{tex:02d}/traj_{k:04d}_m{m:03d}_r{r}_press.npz"
                    np.savez_compressed(os.path.join(sdir, fname), sim_tactip=np.array(sim),
                                        sim_markers=disp, real_tactip=np.array(real), phase=phase,
                                        waypoint=np.zeros((steps, 3), np.float32))
                    rows.append(dict(file=fname, texture=tex, number=k, kind="press",
                                     samples=steps, motion=m, repeat=r, params="{}"))
        meta = dict(sim=dict(image_size=sim_size, marker_rest_mm=rest.tolist()), trajectories=rows)
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump(meta, f)
        out.append(sdir)
    return out


SMALL = dict(sim_size=32, real_size=(32, 32))


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    return fake_sessions(str(tmp_path_factory.mktemp("data")), names=("session_a", "session_b"))


# ============================================================================ data
def test_paired_load_crop_and_split(sessions):
    data = PairedData.load(os.path.dirname(sessions[0]), **SMALL, log=lambda *a: None)
    assert len(data.sessions) == 2 and len(data) == 2 * 2 * 5 * 2 * 10
    assert data.sim.shape == (len(data), 32, 32, 3) and data.sim.dtype == np.uint8
    assert data.real.shape == (len(data), 32, 32, 3) and data.real.dtype == np.uint8
    assert data.prep.real_crop == (20, 0, 120, 120) and data.prep.real_frame == (160, 120)
    assert data.prep.sim_render == 48
    # the crop keeps only the sensor (no noise) and the frames come out RGB: red > blue
    assert data.real[..., 0].max() > data.real[..., 2].max()
    assert data.real[..., 2].max() <= 100                  # no noise from outside the crop
    tr, va = data.split(0.2, by="motion", seed=1)
    assert len(np.intersect1d(tr, va)) == 0 and len(tr) + len(va) == len(data)
    g = data.groups("motion")
    assert not set(g[tr]) & set(g[va])                      # every repeat of a motion together
    t_tr, t_va = data.split(0.5, by="texture")
    assert not set(data.texture[t_tr]) & set(data.texture[t_va])
    contact = PairedData.load(sessions[0], phases=["contact"], **SMALL, log=lambda *a: None)
    assert (contact.phase == 1).all() and len(contact) == 2 * 5 * 2 * 6


def test_preprocess_round_trip_and_sizes():
    p = Preprocess(sim_input="image", sim_size=16, real_size=(24, 16), real_crop=(10, 0, 40, 30),
                   gray=True)
    sim, single = p.sim(np.zeros((40, 40, 3), np.uint8))
    assert single and sim.shape == (1, 16, 16, 1)
    frames = np.zeros((3, 30, 60, 3), np.uint8)
    frames[:, :, 10:50] = (0, 0, 255)                       # red in BGR, inside the crop
    real, single = p.real(frames)
    assert not single and real.shape == (3, 16, 24, 1)
    assert abs(int(real.max()) - int(round(0.299 * 255))) <= 1
    q = Preprocess.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q.to_dict() == p.to_dict()
    assert q.crop_box((60, 120)) == (20, 0, 80, 60)         # a frame twice the size: scaled
    full = Preprocess(real_crop="full")
    full.real(np.zeros((10, 20, 3), np.uint8))
    assert full.real_crop == (0, 0, 20, 10)
    a = np.arange(64, dtype=np.float32).reshape(8, 8)
    assert np.allclose(resize_area(a, (4, 4)), a.reshape(4, 2, 4, 2).mean(axis=(1, 3)))


# ============================================================================ linear backend
def test_linear_mapping_images(sessions, tmp_path):
    data = PairedData.load(sessions, **SMALL, log=lambda *a: None)
    out = str(tmp_path / "lin")
    m = train_mapping(data, out=out, backend="linear", latent_sim=24, latent_real=24,
                      log=lambda *a: None)
    mt = m.info["metrics"]
    assert mt["sim2real_nmse"] < 0.35, mt                   # well under "the average image"
    assert mt["real2sim_nmse"] < 0.35 and mt["real_rec_nmse"] < 0.3, mt
    for f in ("mapping.json", "weights.npz", "preview.png", "history.csv"):
        assert os.path.exists(os.path.join(out, f))
    again = load_mapping(out)
    raw = np.load(os.path.join(sessions[0], json.load(open(os.path.join(sessions[0], "meta.json")))
                               ["trajectories"][0]["file"]))
    one = again.sim_to_real(raw["sim_tactip"][5])           # sim camera image as rendered (48 px)
    assert one.shape == (32, 32, 3) and one.dtype == np.uint8
    assert np.array_equal(one, m.sim_to_real(raw["sim_tactip"][5]))
    batch = again(raw["sim_tactip"])
    assert batch.shape == (10, 32, 32, 3) and np.array_equal(batch[5], one)
    back = again.real_to_sim(raw["real_tactip"][5])         # a camera frame (BGR, full size)
    assert back.shape == (32, 32, 3) and back.dtype == np.uint8
    assert again.encode_sim(raw["sim_tactip"]).shape == (10, 24)
    assert again.reconstruct_real(raw["real_tactip"]).shape == (10, 32, 32, 3)
    # the mapped image is closer to the real one than the average real image is
    target = again.prep.real(raw["real_tactip"])[0].astype(float)
    err = np.mean((batch.astype(float) - target) ** 2)
    assert err < np.mean((data.real.mean(axis=0) - target) ** 2)


def test_linear_mapping_markers(sessions, tmp_path):
    data = PairedData.load(sessions[0], sim_input="markers", **SMALL, log=lambda *a: None)
    assert data.sim.shape == (len(data), 37 * 3) and data.prep.n_markers == 37
    m = train_mapping(data, out=str(tmp_path / "mk"), backend="linear", latent_sim=12,
                      latent_real=24, log=lambda *a: None)
    assert m.info["metrics"]["sim2real_nmse"] < 0.5
    assert m.info["metrics"]["real2sim_rmse_mm"] < 0.3
    m2 = load_mapping(str(tmp_path / "mk"))
    disp = data.sim[3].reshape(37, 3)
    assert m2.sim_to_real(disp).shape == (32, 32, 3)
    assert m2.sim_to_real(np.stack([disp, disp])).shape == (2, 32, 32, 3)
    assert m2.real_to_sim(data.real[3], raw=False).shape == (37, 3)


def test_crop_check_and_cli(sessions, tmp_path, capsys):
    from tactip_mujoco_gym.sim2real.__main__ import main
    data = PairedData.load(sessions[0], **SMALL, log=lambda *a: None)
    png = crop_check(data, str(tmp_path / "crop.png"))
    assert open(png, "rb").read(4) == b"\x89PNG"
    out = str(tmp_path / "cli")
    main(["train-mapping", sessions[0], "--backend", "linear", "--out", out, "--sim-size", "32",
          "--real-size", "32", "32", "--latent-real", "16", "--gray"])
    assert os.path.exists(os.path.join(out, "crop_check.png"))
    m = load_mapping(out)
    assert m.prep.gray and m.real_shape == (32, 32, 1)
    main(["mapping-preview", out, sessions[1], "--out", str(tmp_path / "pv.png"), "--n", "3"])
    text = capsys.readouterr().out
    assert "sim->real" in text and os.path.exists(tmp_path / "pv.png")


def test_settings_are_checked(sessions):
    data = PairedData.load(sessions[0], **SMALL, log=lambda *a: None)
    with pytest.raises(TypeError, match="unknown training settings"):
        train_mapping(data, backend="linear", latnet_sim=3, log=lambda *a: None)
    with pytest.raises(FileNotFoundError):
        load_mapping(os.path.join(sessions[0], "nothing"))


# ============================================================================ torch backend
def test_torch_mapping(sessions, tmp_path):
    pytest.importorskip("torch")
    data = PairedData.load(sessions[0], **SMALL, log=lambda *a: None)
    out = str(tmp_path / "torch")
    m = train_mapping(data, out=out, backend="torch", device="cpu", latent_sim=8, latent_real=8,
                      base=8, max_ch=32, map_hidden=16, epochs_ae=2, epochs_map=2,
                      epochs_joint=1, batch=32, log=lambda *a: None)
    hist = open(os.path.join(out, "history.csv")).read()
    for stage in ("sim_ae", "real_ae", "map", "joint"):
        assert stage in hist
    assert np.isfinite(m.info["metrics"]["sim2real_nmse"])
    again = load_mapping(out, device="cpu")
    x = data.sim[:4]
    assert np.abs(again.sim_to_real(x).astype(int) - m.sim_to_real(x).astype(int)).max() <= 1
    assert again.real_to_sim(data.real[:4], raw=False).shape == (4, 32, 32, 3)
    mk = PairedData.load(sessions[0], sim_input="markers", **SMALL, log=lambda *a: None)
    m2 = train_mapping(mk, backend="torch", device="cpu", latent_sim=4, latent_real=8, base=8,
                       max_ch=16, map_hidden=8, epochs_ae=1, epochs_map=1, epochs_joint=1,
                       log=lambda *a: None)
    assert m2.sim_to_real(mk.sim[0].reshape(-1, 3)).shape == (32, 32, 3)


def test_torch_rejects_odd_sizes():
    pytest.importorskip("torch")
    from tactip_mujoco_gym.sim2real.mapping import check_image_size
    with pytest.raises(ValueError, match="divisible by 8"):
        check_image_size(30, 30)


# ============================================================================ in the gym
def _gym():
    pytest.importorskip("mujoco")
    gym = pytest.importorskip("gymnasium")
    import tactip_mujoco_gym  # noqa: F401
    return gym


def test_real_tactip_readout(sessions, tmp_path, monkeypatch):
    gym = _gym()
    from tactip_mujoco_gym.sensors.real_tactip import ENV_VAR
    monkeypatch.delenv(ENV_VAR, raising=False)
    env = gym.make("TacRig-v0")
    ok, why = env.unwrapped.describe_inputs()["real_tactip"]
    assert not ok and "train-mapping" in why                 # there, but no mapping yet
    env.close()

    data = PairedData.load(sessions[0], **SMALL, log=lambda *a: None)
    out = str(tmp_path / "lin")
    train_mapping(data, out=out, backend="linear", latent_sim=8, latent_real=8, log=lambda *a: None)
    env = gym.make("TacRig-v0", sensors={"real_tactip": {"model": out}},
                   obs_modes=["real_tactip"], obs_format="dict")
    obs, _ = env.reset(seed=0)
    assert obs["real_tactip"].shape == (32, 32, 3) and obs["real_tactip"].dtype == np.uint8
    assert env.observation_space["real_tactip"].contains(obs["real_tactip"])
    obs, *_ = env.step(env.action_space.sample())
    assert obs["real_tactip"].shape == (32, 32, 3)
    s = env.unwrapped.sensors["real_tactip"]
    assert s.sim_input(env.unwrapped).shape == (1, 48, 48, 3)        # rendered at the data's size
    env.close()

    # in place of the simulated camera, and from the environment variable
    env = gym.make("TacRig-v0", obs_modes=["tactile_image"], obs_format="dict",
                   sensors={"tactile_image": {"type": "real_tactip", "model": out}})
    assert env.reset(seed=0)[0]["tactile_image"].shape == (32, 32, 3)
    env.close()
    monkeypatch.setenv(ENV_VAR, out)
    env = gym.make("TacRig-v0", obs_modes=["real_tactip"])           # flat observations
    obs, _ = env.reset(seed=0)
    assert np.isfinite(obs).all()
    env.close()


def test_real_tactip_readout_markers(tmp_path):
    gym = _gym()
    env = gym.make("TacRig-v0")
    n = env.unwrapped.sensors["tactile"].n_markers
    env.close()
    sdir = fake_sessions(str(tmp_path), n_markers=n, motions=3, textures=(1,))
    data = PairedData.load(sdir, sim_input="markers", **SMALL, log=lambda *a: None)
    out = str(tmp_path / "mk")
    train_mapping(data, out=out, backend="linear", latent_sim=8, latent_real=8, log=lambda *a: None)
    env = gym.make("TacRig-v0", sensors={"real_tactip": {"model": out}},
                   obs_modes=["real_tactip"], obs_format="dict")
    obs, _ = env.reset(seed=0)
    assert obs["real_tactip"].shape == (32, 32, 3)
    env.close()
