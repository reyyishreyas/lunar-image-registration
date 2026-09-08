"""Unit tests for the fully-automatic pipeline (Phase 8, src/auto_pipeline.py).

Covers the pure building blocks: least-squares homography, homography apply,
iterative sub-pixel tightening, RMSE, and the JSON report round-trip. The heavy
real-data end-to-end runs are exercised via scripts/run_auto.py and recorded in
results/logs — not in fast unit tests.
"""
import json

import numpy as np
import pytest

from src.auto_pipeline import (
    _lsq_homography,
    _apply_homography,
    _tighten,
    _rmse,
    run_auto,
    summarize,
    write_json_report,
    load_report,
)


def _make_target(n=40, seed=0):
    rng = np.random.default_rng(seed)
    pts1 = rng.uniform(0, 512, (n, 2))
    H = np.array([[1.001, 0.002, 3.0],
                  [0.001, 1.0005, -2.0],
                  [1e-5, 2e-5, 1.0]])
    pts2 = _apply_homography(H, pts1)
    return pts1, pts2, H


def test_lsq_homography_identity():
    H = _lsq_homography(np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]),
                        np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]]))
    assert np.allclose(H, np.eye(3), atol=1e-9)


def test_lsq_homography_recovers_transform():
    pts1, pts2, H = _make_target()
    Hr = _lsq_homography(pts1, pts2)
    # compare up to scale
    assert np.allclose(Hr / Hr[2, 2], H / H[2, 2], atol=1e-6)


def test_apply_homography_projective():
    pts1, pts2, H = _make_target()
    out = _apply_homography(H, pts1)
    assert np.allclose(out, pts2, atol=1e-6)


def test_tighten_recovers_exact_model_on_clean_data():
    pts1, pts2, _ = _make_target()
    H, keep, r = _tighten(pts1, pts2)
    assert keep.sum() == len(pts1)
    assert np.abs(r).max() < 1e-3


def test_tighten_ignores_gross_outliers():
    pts1, pts2, H = _make_target()
    # corrupt 5 points to be gross outliers
    rng = np.random.default_rng(7)
    idx = rng.choice(len(pts1), 5, replace=False)
    bad = pts2.copy()
    bad[idx] += rng.uniform(0, 1, (5, 2)) * 500
    H, keep, r = _tighten(pts1, bad)
    inl = np.linalg.norm(_apply_homography(H, pts1) - bad, axis=1) < 2.0
    reclaim = inl[~np.isin(np.arange(len(pts1)), idx)]
    assert reclaim.all(), "valid points should survive tightening"


def test_rmse_reports_px():
    pts1, pts2, H = _make_target()
    rmse, res = _rmse(H, pts1, pts2)
    assert rmse == pytest.approx(0.0, abs=1e-6)
    assert len(res) == len(pts1)


def test_report_roundtrip(tmp_path):
    report = {"verdict": "registered", "rmse_px": 0.5, "inliers": 10,
              "method": "content"}
    p = tmp_path / "nested" / "report.json"
    write_json_report(report, str(p), root="")
    assert p.exists()
    loaded = load_report(str(p), root="")
    assert loaded == report


def test_summarize_registered():
    s = summarize({"verdict": "registered", "rmse_px": 0.5, "rmse_self_px": 0.9,
                   "inliers": 120, "inlier_ratio": 0.42, "n_matches": 286})
    assert s.startswith("REGISTERED")
    assert "0.5" in s


def test_summarize_geometry_registered():
    s = summarize({"verdict": "geometry_registered", "staged_gsd_m": 60.0})
    assert s.startswith("GEOMETRY REGISTERED")
    assert "60" in s


def test_summarize_failure():
    s = summarize({"verdict": "registration_failed", "notes": ["boom"]})
    assert "NOT COMPLETED" in s