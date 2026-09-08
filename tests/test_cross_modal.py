"""Unit tests for Phase 9 cross-modal registration (radiometric invariance).

Covers the normalisation front-ends (gradient_structure, log_ratio), the
cross-modal matcher with proof-based acceptance on a synthetic radiometrically-
mismatched pair, and front-end selection.
"""
import cv2
import numpy as np
import pytest

from src.detection.cross_modal import (
    gradient_structure,
    log_ratio,
    _apply_front,
    cross_modal_register,
    best_front_name,
    run_cross_modal,
)


def _synthetic_radiometric_pair(seed=1, size=256):
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 255, (size, size)).astype(np.uint8)
    base = cv2.GaussianBlur(base, (5, 5), 0)
    cv2.circle(base, (size // 2, size // 2), size // 4, 200, 2)
    cv2.line(base, (20, 40), (size - 20, size - 50), 180, 3)
    H = np.array([[1.002, 0.001, 6.0], [0.001, 1.001, -4.0], [2e-5, 1e-5, 1.0]])
    shifted = cv2.warpPerspective(base, H, (size, size))
    shifted = cv2.bitwise_not(shifted)                # radiometric inversion
    shifted = cv2.convertScaleAbs(shifted, alpha=0.4, beta=40)
    return base, shifted, H


def test_gradient_structure_dtype_range():
    rng = np.random.default_rng(2)
    img = (rng.random((64, 64)) * 255).astype(np.uint8)
    g = gradient_structure(img)
    assert g.shape == img.shape and g.dtype == np.uint8
    assert g.min() >= 0 and g.max() <= 255


def test_log_ratio_on_flat_is_zero():
    flat = np.full((40, 40), 128, np.uint8)
    out = log_ratio(flat)
    assert out.min() == out.max()  # no structure -> uniform


def test_apply_front_unknown_raises():
    img = np.zeros((8, 8), np.uint8)
    with pytest.raises(ValueError):
        _apply_front(img, "nope")


def test_cross_modal_recovers_homography_across_radiometric_gap():
    src, ref, Htrue = _synthetic_radiometric_pair()
    res = cross_modal_register(src, ref, min_inliers=8, min_ratio=0.02,
                               verbose=False)
    assert res is not None, "cross-modal should find the model despite radiometry"
    assert res["front"] == "gradient_structure"
    assert res["inliers"] >= 8
    # recovered scale should match truth (H00 ratio ~1)
    assert abs(res["H"][0, 0] / Htrue[0, 0] - 1.0) < 0.01
    # translation ~6 px
    assert abs(res["H"][0, 2] - 6.0) < 1.0


def test_run_cross_modal_ok_flag():
    src, ref, _ = _synthetic_radiometric_pair()
    ok, res = run_cross_modal(src, ref)
    assert ok is True
    assert res["ok"] is True


def test_best_front_name():
    assert best_front_name({"front": "histogram_match"}) == "histogram_match"
    assert best_front_name(None) == "n/a"
