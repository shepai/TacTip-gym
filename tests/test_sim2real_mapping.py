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
                  steps=10, textures=(1, 2), seed=0, names=("session_a",), issues=None, noise=2.0,
                  pin_sigma=1.1, depth_frames=False, approach=3):
    """rig_dataset-style sessions: the sim image shows the markers; the real frame shows the
    same markers mirrored, recoloured, in the centre square of a wider BGR frame with noise
    either side. Each trajectory: 3 approach samples (2, 1 and 0.2 mm above the block), a
    press 0.3 -> 2 mm deep, 2 retract samples.

    issues: {session name: {...}} to make a session imperfect:
        offset        grey levels added to its real frames (exposure / light)
        drift         std of a further offset per trajectory
        lag           real frame k shows the press of sim sample k - lag
        depth_offset  the real TacTip is this many mm deeper than the sim at every sample
        shuffle       the real frames are shuffled across the session (no pairing at all)
    depth_frames: also a RealSense-like depth image registered with the colour (0.1 mm
    units, the skin 40 mm away, pushed closer where pressed, nothing outside the sensor).
    Returns the session folders."""
    rng = np.random.default_rng(seed)                     # the motions
    pix = np.random.default_rng(seed + 1)                 # camera noise and the frame's edges
    rings = 1
    while 1 + 3 * rings * (rings + 1) < n_markers:
        rings += 1
    rest_xy = hex_layout(rings)[:n_markers]
    rest = np.c_[rest_xy, np.zeros(n_markers)]
    W, H = frame
    down = [-2.0, -1.0, -0.2] if approach == 3 else list(np.linspace(-2.0, -0.1, approach))
    depth = np.r_[down, np.linspace(0.3, 2.0, steps - approach - 2), -1.0, -2.0].astype(np.float32)
    phase = np.r_[np.zeros(approach), np.ones(steps - approach - 2), np.full(2, 2)].astype(np.int8)
    out = []
    for name in names:
        iss = dict(offset=0.0, drift=0.0, lag=0, depth_offset=0.0, shuffle=False)
        iss.update((issues or {}).get(name, {}))
        sdir = os.path.join(root, name)
        rows, trajs = [], []
        for tex in textures:
            os.makedirs(os.path.join(sdir, f"texture_{tex:02d}"), exist_ok=True)
            for m in range(motions):
                c = rng.uniform(-4, 4, 2)
                for r in range(repeats):
                    disp = np.stack([press(rest_xy, c, max(dd, 0.0)) for dd in depth]).astype(np.float32)
                    shift = iss["offset"] + (pix.normal(0.0, iss["drift"]) if iss["drift"] else 0.0)
                    sim, real, dep = [], [], []
                    for i, d in enumerate(disp):
                        pos = rest_xy + d[:, :2]
                        g = blobs(pos, sim_size)
                        sim.append((np.stack([g, g, 0.6 * g], -1) * 230 + 15).astype(np.uint8))
                        j = max(i - int(iss["lag"]), 0)
                        dr = press(rest_xy, c, max(depth[j] + iss["depth_offset"], 0.0))
                        gr = blobs((rest_xy + dr[:, :2]) * [-1, 1], H, sigma=pin_sigma)  # mirrored
                        rgb = np.stack([0.8 * gr, 0.5 * gr, 0.2 * gr + 0.1], -1) * 255 + 20 + shift
                        rgb = rgb + pix.normal(0.0, noise, rgb.shape) if noise else rgb
                        f = pix.integers(0, 255, (H, W, 3)).astype(np.float32)
                        x0 = (W - H) // 2
                        f[:, x0:x0 + H] = rgb
                        real.append(np.clip(f, 0, 255).astype(np.uint8)[..., ::-1])   # BGR
                        if depth_frames:
                            px = ((rest_xy + dr[:, :2]) * [-1, 1] / 8.5 * 0.5 + 0.5) * (H - 1)
                            yy, xx = np.mgrid[0:H, 0:H]
                            k = np.exp(-((xx[..., None] - px[:, 0]) ** 2 + (yy[..., None] - px[:, 1]) ** 2) / 50.0)
                            mm = 40.0 + (k * dr[:, 2]).sum(-1) / (k.sum(-1) + 1e-6)
                            mm = mm + pix.normal(0.0, 0.02, mm.shape)
                            dimg = np.zeros((H, W), np.uint16)
                            dimg[:, x0:x0 + H] = np.rint(mm * 10).astype(np.uint16)
                            dep.append(dimg)
                    k = len(rows)
                    fname = f"texture_{tex:02d}/traj_{k:04d}_m{m:03d}_r{r}_press.npz"
                    wp = np.zeros((steps, 3), np.float32)
                    wp[:, :2], wp[:, 2] = c, depth
                    trajs.append(dict(sim_tactip=np.array(sim), sim_markers=disp,
                                      real_tactip=np.array(real), phase=phase, waypoint=wp))
                    if depth_frames:
                        trajs[-1]["depth"] = np.array(dep)
                    rows.append(dict(file=fname, texture=tex, number=k, kind="press",
                                     samples=steps, motion=m, repeat=r, params="{}"))
        if iss["shuffle"]:
            allreal = np.concatenate([t["real_tactip"] for t in trajs])[pix.permutation(len(trajs) * steps)]
            for k, t in enumerate(trajs):
                t["real_tactip"] = allreal[k * steps:(k + 1) * steps]
        for row, t in zip(rows, trajs):
            np.savez_compressed(os.path.join(sdir, row["file"]), **t)
        meta = dict(sim=dict(image_size=sim_size, marker_rest_mm=rest.tolist()), trajectories=rows,
                    d401=dict(name="Intel RealSense D405", depth_scale_m=1e-4),
                    config=dict(sim_settle_steps=2),
                    calibration=dict(pose_map=dict(kind="offset", known_spm=[50, 50, 50],
                                                   created="2026-11-17T10:00:00"), trim_steps=[0, 0, 0]))
        with open(os.path.join(sdir, "meta.json"), "w") as f:
            json.dump(meta, f)
        out.append(sdir)
    return out


SMALL = dict(sim_size=32, real_size=(32, 32))
MOTIONS = 8


@pytest.fixture(scope="module")
def sessions(tmp_path_factory):
    return fake_sessions(str(tmp_path_factory.mktemp("data")), names=("session_a", "session_b"),
                         motions=MOTIONS)


# ============================================================================ data
def test_paired_load_crop_and_split(sessions):
    data = PairedData.load(os.path.dirname(sessions[0]), **SMALL, log=lambda *a: None)
    assert len(data.sessions) == 2 and len(data) == 2 * 2 * MOTIONS * 2 * 10
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
    assert (contact.phase == 1).all() and len(contact) == 2 * MOTIONS * 2 * 5


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


def test_pairing_check_and_min_corr(tmp_path):
    sd = fake_sessions(str(tmp_path), names=("good", "lag", "deep", "shuffled"), motions=6,
                       textures=(1,), issues={"lag": {"lag": 1}, "deep": {"depth_offset": 0.8},
                                              "shuffled": {"shuffle": True}})
    lines = []
    data = PairedData.load(sd, **SMALL, log=lines.append)
    rep = {r["session"]: r for r in data.pairing}
    assert rep["good"]["verdict"] == "good" and rep["good"]["lag"] == 0, rep["good"]
    assert abs(rep["good"]["onset_mm"]) < 0.3, rep["good"]
    assert rep["lag"]["lag"] == 1 and rep["lag"]["lag_corr"] > rep["lag"]["corr"] + 0.2, rep["lag"]
    assert rep["deep"]["onset_mm"] < -0.3, rep["deep"]          # the real TacTip touches first
    assert rep["shuffled"]["verdict"] == "poor", rep["shuffled"]
    text = "\n".join(lines)
    assert "shuffled: the real images hardly follow" in text and "--settle" in text
    assert "deep (" in text and "real lower" in text
    kept = PairedData.load(sd, **SMALL, min_corr=0.4, log=lambda *a: None)
    assert set(np.unique(kept.session)) == {0, 1, 2}


def test_reference_takes_out_camera_drift(tmp_path):
    """Session b's camera is brighter and drifts from trajectory to trajectory: the simulator
    can't predict that, so without a reference the mapping can't either."""
    sd = fake_sessions(str(tmp_path / "data"), names=("a", "b"), motions=MOTIONS,
                       issues={"b": {"offset": 30, "drift": 10}})
    quiet = lambda *a: None  # noqa: E731
    scores = {}
    for ref in ("none", "trajectory"):
        d = PairedData.load(sd, **SMALL, reference=ref, log=quiet)
        m = train_mapping(d, out=str(tmp_path / ref), backend="linear", latent_sim=16,
                          latent_real=16, log=quiet)
        scores[ref] = m.info["metrics"]["sim2real_nmse"]
    assert scores["trajectory"] < 0.6 * scores["none"], scores
    m = load_mapping(str(tmp_path / "trajectory"))
    assert m.uses_reference and m.real_rest.shape == (32, 32, 3)
    assert os.path.exists(tmp_path / "trajectory" / "real_rest.png")
    with open(os.path.join(sd[1], "meta.json")) as f:
        row = json.load(f)["trajectories"][0]
    raw = np.load(os.path.join(sd[1], row["file"]))
    # a new session whose camera is brighter still; nothing touching the TacTip
    untouched = np.clip(raw["real_tactip"][:2].astype(int) + 40, 0, 255).astype(np.uint8)
    with pytest.warns(UserWarning, match="set_reference"):
        m.real_to_sim(raw["real_tactip"][5])
    m.set_reference(untouched)
    target = m.prep.real(untouched)[0].astype(float).mean(axis=0)
    out = m.sim_to_real(raw["sim_tactip"][0])               # sim untouched -> this session's look
    assert np.abs(out.astype(float) - target).mean() < 4
    assert m.real_to_sim(raw["real_tactip"][5]).shape == (32, 32, 3)
    m.set_reference(None)
    assert np.abs(m.sim_to_real(raw["sim_tactip"][0]).astype(float) - target).mean() > 15


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
    again.set_reference(raw["real_tactip"][:2])              # this session's untouched TacTip
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


def test_sobel_filter(sessions, tmp_path):
    data = PairedData.load(sessions[0], **SMALL, filter="sobel", log=lambda *a: None)
    assert data.prep.gray and data.sim.shape[1:] == (32, 32, 1) and data.real.shape[1:] == (32, 32, 1)
    assert data.pairing[0]["corr"] > 0.7
    out = str(tmp_path / "sobel")
    train_mapping(data, out=out, backend="linear", latent_sim=16, latent_real=16, log=lambda *a: None)
    m = load_mapping(out)
    assert m.prep.filter == "sobel"
    raw = np.load(os.path.join(sessions[0], data.rows[0]["file"]))
    assert m.sim_to_real(raw["sim_tactip"][5]).shape == (32, 32, 1)


# ============================================================================ real pins, marker map
@pytest.fixture(scope="module")
def pin_sessions(tmp_path_factory):
    pytest.importorskip("cv2")
    return fake_sessions(str(tmp_path_factory.mktemp("pins")), names=("pa", "pb"), motions=6,
                         pin_sigma=2.0, depth_frames=True, noise=1.0)


def test_depth_check_and_extract_pins(pin_sessions, tmp_path):
    from tactip_mujoco_gym.sim2real import depth_check, extract_pins
    out = depth_check(pin_sessions[0], picture=str(tmp_path / "dc.png"), log=lambda *a: None)
    assert out["pins"] >= 30 and out["registered"]
    assert out["depth_valid_pins"] > 0.9 and abs(out["distance_mm"] - 40) < 1.0
    assert out["usable"], out
    assert os.path.exists(tmp_path / "dc.png")
    for sdir in pin_sessions:
        assert extract_pins(sdir, log=lambda *a: None) > 0
    with open(os.path.join(pin_sessions[0], "meta.json")) as f:
        row = json.load(f)["trajectories"][0]
    z = np.load(os.path.join(pin_sessions[0], "pins", row["file"]))
    P = json.load(open(os.path.join(pin_sessions[0], "pins", "pins.json")))["pins"]
    assert z["real_pins"].shape == (10, P, 3) and z["real_pins_ok"].mean() > 0.8
    still = np.nanmax(np.abs(z["real_pins"][:2, :, :2]))
    pressed = np.nanmax(np.abs(z["real_pins"][7, :, :2]))
    assert pressed > 3 * max(still, 0.1)                     # the press moves the pins
    assert np.nanmin(z["real_pins"][7, :, 2]) < -0.5           # and brings them closer


def test_marker_map(pin_sessions, tmp_path):
    from tactip_mujoco_gym.sim2real import MarkerPairs, extract_pins, load_marker_map, train_marker_map
    for sdir in pin_sessions:
        if not os.path.exists(os.path.join(sdir, "pins", "pins.json")):
            extract_pins(sdir, log=lambda *a: None)
    pairs = MarkerPairs.load(pin_sessions, log=lambda *a: None)
    assert pairs.use_z and pairs.real.shape[2] == 3 and pairs.sim.shape[1:] == (37, 3)
    m = train_marker_map(pairs, out=str(tmp_path / "mm"), log=lambda *a: None)
    mt = m.cfg["metrics"]
    assert mt["sim2real_nmse"] < 0.5 and mt["real2sim_nmse"] < 0.5, mt
    again = load_marker_map(str(tmp_path / "mm"))
    x = pairs.sim[5]
    assert np.allclose(again.sim_to_real(x), m.sim_to_real(x), atol=1e-4)
    assert again.sim_to_real(pairs.sim[:4]).shape == (4, pairs.real.shape[1], 3)
    real = pairs.real[5].copy()
    real[:3] = np.nan                                         # pins not found
    assert again.real_to_sim(real).shape == (37, 3)
    xy = MarkerPairs.load(pin_sessions[0], use_z=False, log=lambda *a: None)
    assert xy.real.shape[2] == 2


def test_marker_map_mlp(pin_sessions, tmp_path):
    pytest.importorskip("torch")
    from tactip_mujoco_gym.sim2real import MarkerPairs, extract_pins, load_marker_map, train_marker_map
    for sdir in pin_sessions:
        if not os.path.exists(os.path.join(sdir, "pins", "pins.json")):
            extract_pins(sdir, log=lambda *a: None)
    pairs = MarkerPairs.load(pin_sessions, log=lambda *a: None)
    m = train_marker_map(pairs, out=str(tmp_path / "mlp"), backend="mlp", epochs=5, hidden=32,
                         device="cpu", log=lambda *a: None)
    again = load_marker_map(str(tmp_path / "mlp"))
    assert np.allclose(again.sim_to_real(pairs.sim[3]), m.sim_to_real(pairs.sim[3]), atol=1e-4)


# ============================================================================ re-simulation
class FakeRig:
    """Stands in for SimRig on TacRig-v0: goes straight to each target; the markers move
    as the fake sensor's press at the commanded depth."""
    start_height_mm = 10.0

    def __init__(self, rest_xy, depth_size=16):
        self.rest_xy, self.pos, self.centre = rest_xy, np.zeros(3), np.zeros(2)
        rig = self

        class Depth:
            available = True

            def observe(self, e):
                return np.full((depth_size, depth_size, 1), 40.0 + rig.displacement()[:, 2].min(), np.float32)

        class Env:
            plate_force = 0.0
            sensors = {"tactile_depth": Depth()}
        self.e = Env()

    def reset(self, texture=None, verbose=True):
        self.pos = np.array([0.0, 0.0, 0.0])

    def step_toward(self, target):
        self.pos = np.asarray(target, float)
        return True

    def _raw(self):
        return self.pos.copy()

    def displacement(self):
        d = -self.pos[2] - self.start_height_mm
        return press(self.rest_xy, self.pos[:2], max(d, 0.0)).astype(np.float32)


def test_resimulate_fits_z_and_merges(tmp_path):
    from tactip_mujoco_gym.sim2real import resimulate
    sd = fake_sessions(str(tmp_path / "data"), names=("deep",), motions=4, textures=(1,),
                       steps=24, approach=12, issues={"deep": {"depth_offset": 0.75}})[0]
    rest = np.asarray(json.load(open(os.path.join(sd, "meta.json")))["sim"]["marker_rest_mm"])[:, :2]
    lines = []
    out = resimulate(sd, z_offset="fit", rig=FakeRig(rest), fit_values=(0.0, 0.25, 0.5, 0.75, 1.0),
                     log=lines.append)
    meta = json.load(open(os.path.join(out, "meta.json")))
    assert meta["resim"]["z_offset_mm"] in (0.5, 0.75, 1.0), lines
    assert len(meta["trajectories"]) == 8 and meta["sim"]["depth_size"] == 64
    # the re-simulated session pairs better than the recording
    q = lambda *a: None  # noqa: E731
    before = PairedData.load(sd, sim_input="markers", **SMALL, log=q).pairing[0]
    after = PairedData.load(out, sim_input="markers", **SMALL, log=q).pairing[0]
    assert abs(after["onset_mm"]) < abs(before["onset_mm"]), (before, after)
    d = PairedData.load(out, sim_input="depth", **SMALL, log=q)    # real images from the recording
    assert d.sim.shape[1:] == (32, 32, 1) and d.real.shape[1:] == (32, 32, 3)
    lines = []
    with pytest.raises(ValueError, match="no paired samples"):
        PairedData.load(out, sim_input="image", **SMALL, log=lines.append)
    assert any("--record image" in s for s in lines)


class FakeContactRig(FakeRig):
    """The depth TacTip version: no markers, a contact depth image (uint8, deeper = brighter)."""
    n = 0

    def __init__(self, rest_xy, size=16):
        super().__init__(rest_xy)
        rig = self

        class Contact:
            available = True

            def observe(self, e):
                d = max(-rig.pos[2] - rig.start_height_mm, 0.0)
                img = np.zeros((size, size, 1), np.uint8)
                r = int(min(d / 2.0, 1.0) * size / 2)
                c = size // 2
                img[c - r:c + r, c - r:c + r] = int(min(d / 4.0, 1.0) * 255)
                return img
        self.e.sensors = {"tactile_image": Contact()}


def test_resimulate_contact_depth(tmp_path):
    from tactip_mujoco_gym.sim2real import resimulate
    from tactip_mujoco_gym.sim2real.resim import dome_radius_mm
    sd = fake_sessions(str(tmp_path / "data"), names=("s",), motions=4, textures=(1,),
                       steps=24, approach=12, issues={"s": {"depth_offset": 0.5}})[0]
    rest = np.asarray(json.load(open(os.path.join(sd, "meta.json")))["sim"]["marker_rest_mm"])[:, :2]
    with pytest.raises(ValueError, match="on its own"):
        resimulate(sd, record=("contact", "markers"), rig=FakeContactRig(rest), log=lambda *a: None)
    out = resimulate(sd, record=("contact",), z_offset="fit", rig=FakeContactRig(rest),
                     fit_values=(0.0, 0.25, 0.5, 0.75, 1.0), log=lambda *a: None)
    assert out.endswith("_resim_contact")
    m = json.load(open(os.path.join(out, "meta.json")))
    assert m["resim"]["z_offset_mm"] in (0.5, 0.75, 1.0) and m["sim"]["contact_size"] == 64
    d = PairedData.load(out, sim_input="contact", **SMALL, log=lambda *a: None)
    assert d.sim.shape[1:] == (32, 32, 1) and d.sim.max() > 0
    mp = train_mapping(d, backend="linear", latent_sim=8, latent_real=8, log=lambda *a: None)
    assert mp.real_to_sim(d.real[3], raw=False).shape == (32, 32, 1)
    # a sphere's radius from points on it
    pts = np.random.default_rng(1).normal(size=(80, 3))
    pts = 20 * pts / np.linalg.norm(pts, axis=1, keepdims=True) + [1, 2, 3]
    assert abs(dome_radius_mm({"sim": {"marker_rest_mm": pts.tolist()}}) - 20) < 1e-6
