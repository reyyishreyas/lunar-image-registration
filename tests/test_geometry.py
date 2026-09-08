"""Unit tests for the SPICE equal-GSD geometry staging (src.preprocessing.geometry).

Synthetic grids only (no data/ or SPICE kernels): exercises the NAC inverse
round-trip, the uint8 stretch, the NAC remap, and the strip-ortho builder so the
production pair-2 (phase5) path has a fast deterministic regression.
"""
import numpy as np
import pytest

from src.preprocessing.geometry import (
    NacInverse, as_u8, build_strip_orthos, remap_nac, load_ohrc_geom,
    load_nac_geom, _gsd_from_geometry,
)


@pytest.fixture
def nac_grid():
    line_ax = np.linspace(0.0, 100.0, 41)
    samp_ax = np.linspace(0.0, 50.0, 21)
    L, S = np.meshgrid(line_ax, samp_ax, indexing="ij")
    lat0, lon0 = 8.0, 297.0
    dlat, dlon = -0.02, -0.015
    lat = lat0 + dlat * L + 0.0002 * S  # tiny skew: non-pure-rectangular
    lon = lon0 + dlon * S + 0.0002 * L
    inv = NacInverse(lon.ravel(), lat.ravel(), L.ravel(), S.ravel())
    return inv, line_ax, samp_ax


def test_nac_inverse_roundtrip(nac_grid):
    inv, line_ax, samp_ax = nac_grid
    rng = np.random.default_rng(7)
    L0 = rng.uniform(5, 95, 200)
    S0 = rng.uniform(3, 47, 200)
    lon0, lat0 = inv.forward(L0, S0)
    S1, L1 = inv.apply(lon0, lat0)
    assert np.allclose(L1, L0, atol=1e-6)
    assert np.allclose(S1, S0, atol=1e-6)


def test_nac_inverse_boundaries(nac_grid):
    inv, _, _ = nac_grid
    lo_lo_bad = np.array([290.0, 5.0])   # lon far outside grid (west)
    lo_hi_bad = np.array([305.0, 5.0])   # lon far outside grid (east)
    S, L = inv.apply(lo_lo_bad, np.array([8.2, 8.2]))
    assert np.isnan(S).all() and np.isnan(L).all()
    S2, L2 = inv.apply(lo_hi_bad, np.array([6.0, 6.0]))
    assert np.isnan(S2).all() and np.isnan(L2).all()


def test_as_u8_nan_stretch():
    img = np.array([[0.0, 50.0, np.nan], [100.0, 200.0, 255.0]])
    out = as_u8(img)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.max() == 255 and out.min() == 0
    assert out[0, 2] == 0


def test_remap_nac_identity():
    src = (np.arange(40 * 30, dtype=np.float32) % 255).reshape(40, 30)
    ys, xs = np.meshgrid(np.arange(40.0), np.arange(30.0), indexing="ij")
    out = remap_nac(src, ys, xs)
    assert np.allclose(out, src, atol=0.51)


def test_gsd_from_geometry():
    scan = np.array([0.0, 100.0, 200.0])
    px = np.array([0.0, 50.0, 100.0])
    lon = np.zeros((3, 3)) + 296.0 - np.add.outer(np.zeros(3), np.arange(3) * 0.002)
    lat = np.zeros((3, 3)) + 7.0 + np.add.outer(np.arange(3) * 0.001, np.zeros(3))
    m_scan, m_px = _gsd_from_geometry(scan, px, lon, lat)
    assert m_scan == pytest.approx(0.002 * 111000.0 / 200.0, rel=1e-3)
    assert m_px == pytest.approx(
        0.004 * 111000.0 * np.cos(np.deg2rad(7.0)) / 100.0, rel=1e-3)


def test_build_strip_orthos_synthetic():
    scan_axis = np.array([0.0, 200.0, 400.0])
    px_axis = np.array([0.0, 100.0])
    lon = np.array([[297.00, 296.95], [297.00, 296.95], [297.00, 296.95]])
    lat = np.array([[8.0, 8.0], [7.9, 7.9], [7.8, 7.8]])
    line_ax = np.linspace(0.0, 300.0, 31)
    samp_ax = np.linspace(0.0, 120.0, 13)
    L, S = np.meshgrid(line_ax, samp_ax, indexing="ij")
    latn = 8.0 - 0.001 * L
    lonn = 297.0 - 0.0005 * S
    inv = NacInverse(lonn.ravel(), latn.ravel(), L.ravel(), S.ravel())
    ohrc = np.random.default_rng(3).random((500, 120))
    nac = np.random.default_rng(4).random((400, 150)).astype(np.float32)
    strip = build_strip_orthos(ohrc, scan_axis, px_axis, lon, lat, inv, nac, gsd_m=40.0)
    assert strip["src"].ndim == 2 and strip["ref"].ndim == 2
    assert strip["mask"].shape == strip["src"].shape
    assert strip["src"].shape[0] > 1 and strip["src"].shape[1] == ohrc.shape[1]
    assert strip["gsd_m"] == 40.0
    assert strip["ref"][strip["mask"]].size > 0


def test_loaders_parse_minimal(tmp_path):
    csv_p = tmp_path / "ohrc.csv"
    csv_p.write_text("Longitude,Latitude,Pixel,Scan\n"
                     "296.1,7.3,0,0\n296.2,7.3,100,0\n"
                     "296.1,7.4,0,100\n296.2,7.4,100,100\n")
    scan, px, lon, lat = load_ohrc_geom(str(csv_p))
    assert list(scan) == [0.0, 100.0]
    assert list(px) == [0.0, 100.0]
    assert lon.shape == (2, 2)
    nac_p = tmp_path / "nac.txt"
    nac_p.write_text("Longitude Latitude Radius Line Sample\n"
                     "297.0 8.0 1737400.0 0 0\n"
                     "296.5 8.0 1737400.0 0 100\n"
                     "297.0 7.0 1737400.0 200 0\n"
                     "296.5 7.0 1737400.0 200 100\n")
    lo, la, ln, sm = load_nac_geom(str(nac_p))
    assert len(lo) == 4 and len(sm) == 4
    assert ln[0] == 0.0 and sm[3] == 100.0