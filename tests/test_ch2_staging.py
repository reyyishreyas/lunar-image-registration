"""Unit tests for Phase 9 (fix) — robust dark/polar CH2 sensor registration.

Covers enhancement, low-contrast detection, ground staging, and the layered
content -> geometry-fallback -> not_registered decision on synthetic geometry.
"""
import os

import numpy as np
import pytest

from src.preprocessing.ch2_staging import (
    overlap_box,
    percentile_stretch,
    enhance_dark,
    low_contrast_score,
    stage_ch2_ground_pair,
    register_ch2_pair,
    GroundGridInverse,
    _make_original_preview,
    _draw_highlight_box,
    _outline_registered,
)
from src.preprocessing.tmc import read_ground_grid, _gsd_m  # noqa: F401


def _write_ch2_geom(tmp_path, name, width, n_scan, lon0, lat0, dlon=1e-4, dlat=1e-4):
    """Write an ISRO-style geometry CSV; plane moves lon with pixel, lat with scan."""
    path = tmp_path / name
    rows = []
    stride = 100
    pix_grid = sorted(set(list(range(0, width, stride)) + [width - 1]))
    for s in range(0, n_scan, stride):
        for p in pix_grid:
            rows.append(f"{lon0 + p * dlon:.6f},{lat0 + s * dlat:.6f},{p},{s}\n")
    path.write_text("Longitude,Latitude,Pixel,Scan\n" + "".join(rows))
    return str(path)


def _write_ch2_raw(tmp_path, name, width, n_scan, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, 256, (n_scan, width)).astype(np.uint8)
    img = tmp_path / name
    img.write_bytes(a.tobytes())
    return str(img)


def test_overlap_box_intersecting():
    A = (np.array([0., 5.]), np.array([0., 5.]))
    B = (np.array([3., 7.]), np.array([2., 6.]))
    b = overlap_box(*A, *B)
    assert b == pytest.approx((3.0, 5.0, 2.0, 5.0))


def test_ground_grid_inverse_separable_grid():
    # pure separable grid: lon is a function of pixel only, lat of scan only
    scan_ax = np.linspace(0, 5000, 51)
    pix_ax = np.linspace(0, 2000, 21)
    S, P = np.meshgrid(scan_ax, pix_ax, indexing="ij")
    lon0, lat0 = 296.0, 8.0
    dlon, dlat = -2e-5, 1e-4
    lon = lon0 + dlon * P
    lat = lat0 + dlat * S
    inv = GroundGridInverse(pix_ax, scan_ax, lon, lat)
    rng = np.random.default_rng(1)
    P0 = rng.uniform(50, 1900, 64)
    S0 = rng.uniform(100, 4800, 64)
    lo0, la0 = lon0 + dlon * P0, lat0 + dlat * S0
    S1, P1 = inv.apply(lo0, la0)
    assert np.allclose(S1, S0, atol=1e-3)
    assert np.allclose(P1, P0, atol=1e-3)


def test_ground_grid_inverse_sweeping_grid():
    # sweeping strip: lat depends on BOTH scan and pixel so the old separable
    # (median-axis) inverse is badly wrong; the 2-D Newton inverse must still
    # round-trip. This is the TMC-2026 -> NAC staging regression guard.
    scan_ax = np.linspace(0, 5000, 51)
    pix_ax = np.linspace(0, 2000, 21)
    S, P = np.meshgrid(scan_ax, pix_ax, indexing="ij")
    lon0, lat0 = 296.0, 8.0
    # lon leans on scan too, lat leans on pixel too (coupled / tilted strip)
    lon = lon0 - 2e-5 * P - 3e-6 * S
    lat = lat0 + 1e-4 * S + 8e-6 * P
    inv = GroundGridInverse(pix_ax, scan_ax, lon, lat)
    rng = np.random.default_rng(2)
    P0 = rng.uniform(50, 1900, 64)
    S0 = rng.uniform(100, 4800, 64)
    lo0, la0 = lon0 - 2e-5 * P0 - 3e-6 * S0, lat0 + 1e-4 * S0 + 8e-6 * P0
    S1, P1 = inv.apply(lo0, la0)
    assert np.allclose(S1, S0, atol=1e-2)
    assert np.allclose(P1, P0, atol=1e-2)


def test_overlap_box_disjoint_returns_none():
    A = (np.array([0., 1.]), np.array([0., 1.]))
    B = (np.array([5., 7.]), np.array([2., 6.]))
    assert overlap_box(*A, *B) is None


def test_percentile_stretch_lifts_dark():
    # realistic dark frame: small gradient on a low baseline (like OHRC-2026)
    base = np.linspace(3, 12, 64 * 64).reshape(64, 64).astype(np.uint8)
    out = percentile_stretch(base)
    assert out.dtype == np.uint8
    # original had only ~9 counts of range; stretched fully spans 0..255
    assert (out.max() - out.min()) > 200
    # monotonic order is preserved
    assert out[0, 0] <= out[-1, -1]


def test_low_contrast_score_dark_higher_than_structured():
    rng = np.random.default_rng(3)
    dark = (rng.random((64, 64)) * 8).astype(np.uint8)          # near-flat dark
    rich = (rng.random((64, 64)) * 255).astype(np.uint8)         # full range
    assert low_contrast_score(dark) > low_contrast_score(rich)


def test_enhance_dark_returns_u8_same_shape():
    img = np.full((50, 50), 4, np.uint8)
    img[10:30, 10:30] = 30
    out = enhance_dark(img)
    assert out.shape == img.shape and out.dtype == np.uint8


def test_stage_ch2_ground_pair_aligned(tmp_path):
    # both sensors cover the same small box -> aligned crops
    gA = _write_ch2_geom(tmp_path, "a.csv", 2000, 800, lon0=10.0, lat0=10.0)
    gB = _write_ch2_geom(tmp_path, "b.csv", 2000, 800, lon0=10.0, lat0=10.0)
    rA = _write_ch2_raw(tmp_path, "a.img", 2000, 800, seed=1)
    rB = _write_ch2_raw(tmp_path, "b.img", 2000, 800, seed=2)
    st = stage_ch2_ground_pair(rA, gA, rB, gB, n_along=200)
    assert st["overlap"] is True
    assert st["src"].shape == st["ref"].shape
    assert st["src"].shape[0] == 200
    assert st["gsd_m"] > 0


def test_stage_ch2_disjoint_not_overlap(tmp_path):
    gA = _write_ch2_geom(tmp_path, "c.csv", 2000, 800, lon0=0.0, lat0=0.0)
    gB = _write_ch2_geom(tmp_path, "d.csv", 2000, 800, lon0=50.0, lat0=50.0)
    rA = _write_ch2_raw(tmp_path, "c.img", 2000, 800, seed=3)
    rB = _write_ch2_raw(tmp_path, "d.img", 2000, 800, seed=4)
    st = stage_ch2_ground_pair(rA, gA, rB, gB, n_along=100)
    assert st["overlap"] is False


def test_register_ch2_pair_no_overlap_verdict(tmp_path):
    gA = _write_ch2_geom(tmp_path, "e.csv", 2000, 800, lon0=0.0, lat0=0.0)
    gB = _write_ch2_geom(tmp_path, "f.csv", 2000, 800, lon0=60.0, lat0=20.0)
    rA = _write_ch2_raw(tmp_path, "e.img", 2000, 800, seed=5)
    rB = _write_ch2_raw(tmp_path, "f.img", 2000, 800, seed=6)
    rep = register_ch2_pair(rA, gA, rB, gB, out_dir=str(tmp_path / "o1"),
                            n_along=120, verbose=False)
    assert rep["verdict"] == "not_registered"
    assert "not overlap" in rep["notes"][0].lower()


def test_register_ch2_pair_geometry_fallback_dark(tmp_path):
    # overlapping but deliberately featureless/dark ref -> geometry fallback
    gA = _write_ch2_geom(tmp_path, "g.csv", 2000, 800, lon0=10.0, lat0=10.0)
    gB = _write_ch2_geom(tmp_path, "h.csv", 2000, 800, lon0=10.0, lat0=10.0)
    rA = _write_ch2_raw(tmp_path, "g.img", 2000, 800, seed=7)
    # near-black reference
    rB = str(tmp_path / "h.img")
    (tmp_path / "h.img").write_bytes(np.zeros((800, 2000), np.uint8).tobytes())
    rep = register_ch2_pair(rA, gA, str(rB), gB, out_dir=str(tmp_path / "o2"),
                            n_along=120, verbose=False)
    # either content (if noise matched — unlikely) or geometry fallback; must
    # never be 'registration_failed' nor 'not_registered' given an overlap
    assert rep["verdict"] in ("registered", "geometry_registered")
    if rep["verdict"] == "geometry_registered":
        assert rep["method"] == "geometry"
        assert "NOT verifiable" in rep["notes"][-1]
    for k in ("original_src", "original_ref", "after_src", "after_ref",
              "change_map", "montage"):
        assert k in rep["artifacts"] and rep["artifacts"][k], k
    assert os.path.isfile(rep["artifacts"]["montage"])


def test_original_preview_box_and_highlights(tmp_path):
    # synthetic strip: 3000 rows x 200 cols, gradient so preview has content
    ll = np.tile(np.arange(200, dtype=np.uint8) * 4, (3000, 1))
    scan = np.array([400.0, 900.0, 1200.0, 2100.0])
    pix = np.array([30.0, 90.0, 130.0, 170.0])
    prev, box = _make_original_preview(ll, scan, pix)
    assert prev is not None and box is not None
    h, w = prev.shape
    assert h <= 1600 and w <= 640
    assert h / w <= 3.5, "preview must stay a wide panel, not a tall sliver"
    x0, y0, x1, y1 = box
    assert 0 <= x0 < x1 <= w and 0 <= y0 < y1 <= h

    # a colour highlight is actually drawn
    import cv2
    boxed = _draw_highlight_box(prev, box)
    outlined = _outline_registered(prev)
    assert boxed.ndim == 3 and outlined.ndim == 3
    red = ((boxed[:, :, 2].astype(int) > 150)
           & (boxed[:, :, 0].astype(int) < 60)).sum()
    assert red > 0, "highlight box must draw red pixels"
    red2 = ((outlined[:, :, 2].astype(int) > 150)
            & (outlined[:, :, 0].astype(int) < 60)).sum()
    assert red2 > 0, "registered outline must draw red pixels"
