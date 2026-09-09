"""Quality-metric helpers introduced in the Phase-11 'any image pair' round:
registered checkerboard, per-product GSD estimates, per-tile residuals."""

from __future__ import annotations

import numpy as np
import pytest

from src.auto_pipeline import (
    _checkerboard,
    _nac_gsd_from_meta,
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