"""Quality-metric helpers introduced in the Phase-11 'any image pair' round:
registered checkerboard, per-product GSD estimates, per-tile residuals."""

from __future__ import annotations

import numpy as np
import pytest

from src.auto_pipeline import (
    _checkerboard,
    _local_norm_residual,
    _match_histogram_2d,
    _nac_gsd_from_meta,
    _residual_structure,
    _sensor_gsd_report,
    _tile_residuals,
)
from src.evaluation.decision import best_product


def test_checkerboard_alternates_between_a_and_b():
    a = np.full((8, 8), 0, np.uint8)
    b = np.full((8, 8), 255, np.uint8)
    chk = _checkerboard(a, b, tiles=2)
    # 2x2 tiles: (0,0)=b, (0,1)=a, (1,0)=a, (1,1)=b
    assert chk[0, 0] == 255 and chk[0, 7] == 0
    assert chk[7, 0] == 0 and chk[7, 7] == 255


def test_nac_gsd_from_meta_derived_from_corners():
    meta = {
        "nac_crop_corners_native": [
            [0.0, 0.0], [0.0, 100.0], [100.0, 100.0], [100.0, 0.0]],
        "ground_corners_lon_lat": [
            [0.0, 0.0], [0.0, 0.001], [0.0009, 0.001], [0.0009, 0.0]],
    }
    # ~111 m per 0.001 deg over ~100 native px -> ~1.1 m/px
    v = _nac_gsd_from_meta(meta)
    assert v is not None and 0.8 < v < 1.4
    assert _nac_gsd_from_meta({}) is None


def test_sensor_gsd_report_tolerates_missing_grid():
    out = _sensor_gsd_report("/nonexistent/ohrc.csv", meta={}, staged_m=0.9)
    assert out["staged_m"] == 0.9
    assert out["scale_ratio"] is None


def test_tile_residuals_bucket_by_source_tile():
    H = np.eye(3, dtype=np.float32)
    pts1 = np.float32([[5, 5], [80, 5], [5, 80], [80, 80], [50, 50]])
    pts2 = pts1 + 0.0  # perfect fit except one: shift BOTH axes by 2
    pts2[0] += 2.0
    tr = _tile_residuals(H, pts1, pts2, shape=(100, 100), tiles=2)
    assert len(tr) == 2 and len(tr[0]) == 2
    assert tr[0][0] == pytest.approx(2.828, abs=1e-3)  # sqrt(8), 3-dec rounding


def test_best_product_prefers_warped_aligned_over_checkerboard():
    report = {"artifacts": {"best_aligned": "a_aligned.png",
                            "checkerboard": "a_checkerboard.png",
                            "matches": "a_matches.png"}}
    assert best_product(report) == "a_aligned.png"


def test_checkerboard_stipples_outside_mask():
    a = np.full((16, 16), 30, np.uint8)
    b = np.full((16, 16), 200, np.uint8)
    mask = np.zeros((16, 16), bool)
    mask[8:, :] = True
    chk = _checkerboard(a, b, tiles=4, mask=mask)
    top = chk[:4, :4].ravel()  # no data above -> dim stipple, never raw content
    assert (top == 64).any() and (top == 110).any()
    assert not (top == 30).any() and not (top == 200).any()


def test_local_norm_residual_removes_brightness_mismatch():
    yy, xx = np.mgrid[0:64, 0:64]
    base = (120 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 300)
            + 40 * np.sin(2 * np.pi * xx / 12)).astype(np.float32)
    a = np.clip(base, 0, 255).astype(np.uint8)
    b = np.clip(0.6 * base + 45, 0, 255).astype(np.uint8)  # same structure,
    mask = np.ones(a.shape, bool)                          # different brightness
    img, mean, std, res = _local_norm_residual(a, b, mask, k=15)
    # local normalisation cancels the global gain/offset -> near-zero residual
    assert mean < 0.15
    assert img.shape == a.shape and img.dtype == np.uint8


def test_residual_structure_reports_smooth_energy_low_for_noise():
    rng = np.random.default_rng(0)
    noise = rng.standard_normal((128, 128)).astype(np.float32)
    st = _residual_structure(noise, np.ones((128, 128), bool), mean=0.0)
    # white noise -> smooth component carries a small share of the variance
    assert st["braid_energy"] < 0.35
    assert st["std"] > 0


def test_tile_nmis_distribution_and_mask_indexing():
    from src.auto_pipeline import _tile_nmis
    yy, xx = np.mgrid[0:128, 0:128]
    a = (40 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 400)
         + np.sin(2 * np.pi * xx / 9)).astype(np.uint8)
    b = a.copy()  # identical -> every tile NMI~high
    mask = np.ones(a.shape, bool)
    r = _tile_nmis(a, b, mask[:, :], tiles=8)
    assert r["min"] > 1.0 and r["max"] > 1.0
    assert r["median"] >= r["min"] and r["mean"] <= r["max"]
    assert r["above_1_05"] == 1.0


def test_match_histogram_2d_aligns_brightness():
    yy, xx = np.mgrid[0:64, 0:64]
    base = (120 * np.exp(-((xx - 32) ** 2 + (yy - 32) ** 2) / 300)
            + 40 * np.cos(2 * np.pi * yy / 9)).astype(np.float32)
    a = np.clip(base, 0, 255).astype(np.uint8)
    b = np.clip(0.5 * base + 60, 0, 255).astype(np.uint8)
    mask = np.ones(a.shape, bool)
    m = _match_histogram_2d(a, b, mask)
    ha, _ = np.histogram(a[mask], bins=256, range=(0, 255))
    hb, _ = np.histogram(b[mask], bins=256, range=(0, 255))
    hm, _ = np.histogram(m[mask], bins=256, range=(0, 255))
    # the matched image tracks b's distribution (quantized 256-bin mapping, so
    # tail CDF may plateau; the bulk must follow)
    ca = np.cumsum(hb) / hb.sum()
    cm = np.cumsum(hm) / hm.sum()
    assert np.abs(ca - cm).max() < 0.4
    assert abs(float(m[mask].mean()) - float(b[mask].mean())) < 8
    assert m.shape == a.shape and m.dtype == a.dtype