"""The dataset textures, wave surfaces and heightfield helpers (no simulation)."""
import struct

import numpy as np
import pytest

pytest.importorskip("mujoco")
from tactip_mujoco_gym import textures as tx  # noqa: E402


def write_stl(path, texture):
    """A block as the dataset's generator makes it: the texture on the generator's grid on a
    base that makes it 7 mm tall (top surface + underside; enough for stl_heights)."""
    x, y = tx.generator_grid()
    x, y = x[::3], y[::3]
    X, Y = np.meshgrid(x, y)
    Z = tx.texture_function(texture)(X, Y)
    bottom = Z.max() - 7.0
    V = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    nr, nc = X.shape
    tris = []
    for i in range(nr - 1):
        for j in range(nc - 1):
            a = i * nc + j
            tris += [(V[a], V[a + 1], V[a + nc]), (V[a + 1], V[a + nc + 1], V[a + nc])]
    c = [np.array([x[0], y[0], bottom]), np.array([x[-1], y[0], bottom]),
         np.array([x[-1], y[-1], bottom]), np.array([x[0], y[-1], bottom])]
    tris += [(c[0], c[2], c[1]), (c[0], c[3], c[2])]
    with open(path, "wb") as f:
        f.write(b"\0" * 80 + struct.pack("<I", len(tris)))
        for t in tris:
            f.write(struct.pack("<12fH", 0, 0, 0, *t[0], *t[1], *t[2], 0))


def test_texture_equations():
    assert tx.texture_function(1)(np.pi / 2, 0.0) == pytest.approx(1.0)
    assert tx.texture_function(2)(0.0, 0.0) == pytest.approx(1.0)          # sin 0 + cos 0
    assert tx.texture_function(5)(np.pi / 2, 0.0) == pytest.approx(0.984, abs=0.001)   # 13-term triangle (as printed)
    assert abs(tx.texture_function(6)(np.pi / 2, 0.0) - 1.0) < 0.05        # square wave
    assert set(tx.TEXTURE_NAMES) == set(range(1, 21))


@pytest.mark.parametrize("t", tx.DATASET)
def test_blocks_are_seven_mm_tall(t):
    H = tx.texture_heights(t, res_mm=0.25)
    xs, ys = tx.node_grid(0.25)
    assert H.shape == (len(ys), len(xs))
    assert xs[-1] == pytest.approx(29.3) and ys[-1] == pytest.approx(30.0)
    assert 6.8 < H.max() <= 7.0 and H.min() > 0.0


def test_stl_block_matches_the_equation(tmp_path):
    path = tmp_path / "t2.stl"
    write_stl(path, 2)
    eq = tx.texture_heights(2, res_mm=0.5)
    stl = tx.texture_heights(str(path), res_mm=0.5)
    assert stl.shape == eq.shape
    assert np.abs(stl - eq).mean() < 0.15                  # mm (the STL here is coarser)


def test_print_imitation_changes_the_surface():
    ideal = tx.texture_heights(4, res_mm=0.25)
    printed = tx.texture_heights(4, res_mm=0.25, blur_mm=0.3, noise_mm=0.02,
                                 rng=np.random.default_rng(0))
    assert 0.01 < np.abs(printed - ideal).mean() < 0.5


@pytest.mark.parametrize("kind", ("flat",) + tx.WAVES)
def test_waves_stay_in_range(kind):
    rng = np.random.default_rng(1)
    s = np.linspace(0, 0.3, 601)
    for _ in range(20):
        h, _ = tx.wave_profile(kind, s, rng, amp=(0.002, 0.006), wavelength=(0.08, 0.16))
        assert np.abs(h).max() <= 0.006 + 1e-9


def test_bilinear_on_a_plane():
    xs, ys = np.linspace(0, 2, 21), np.linspace(0, 1, 11)
    H = 0.5 * xs[None, :] - 0.25 * ys[:, None]
    b = tx.Bilinear(H, 0, 2, 0, 1)
    assert b(1.33, 0.71) == pytest.approx(0.5 * 1.33 - 0.25 * 0.71)
    gx, gy = b.gradient(0.4, 0.3)
    assert gx == pytest.approx(0.5) and gy == pytest.approx(-0.25)
    n = b.normal(0.4, 0.3)
    assert np.linalg.norm(n) == pytest.approx(1.0) and n[0] < 0 < n[1]
