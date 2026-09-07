"""Unit tests for the robust georeference overlap search (Phase 6).

These exercise the pure candidate-statistics / acceptance helpers with small
synthetic grayscale images (no multi-GB data dependency), covering:

* candidate_stats returns an entry per evaluated orientation with sane fields
* find_pair_transform_robust ACCEPTS a genuinely-overlapping crop pair
* find_pair_transform_robust REJECTS a non-overlapping pair (returns None)

The end-to-end pair1 regression and the phase5-diagnostics failure path are
verified in the Phase 6 functional run (scripts/run_pipeline.py + the
deliverable diagnostics artifact), not here, because that exercise needs the
full-size OHRC/NAC rasters.
"""
import cv2
import numpy as np
import pytest

from src.preprocessing.georeference import (
    candidate_stats, find_pair_transform_robust, find_pair_transform,
)


def _craters(shape, seed=0):
    """Synthetic 'lunar' image: gaussian-smoothed random structure with a few
    craters. Locally-unique patches -> mirror/flip copies do NOT produce repeat
    descriptors, so a genuine overlap is detected with high inlier dominance
    and NCC on exactly one orientation."""
    rng = np.random.default_rng(seed)
    h, w = shape
    img = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.6)
    img = (img - img.mean()) / (img.std() + 1e-6) * 70.0
    img = img + 140.0
    for _ in range(26):
        cy, cx = rng.integers(8, h - 8), rng.integers(8, w - 8)
        r = rng.integers(3, 9)
        yy, xx = np.ogrid[:h, :w]
        ring = (yy - cy) ** 2 + (xx - cx) ** 2
        img += np.clip(18.0 * np.sin(np.pi * np.sqrt(ring) / max(r, 1)), 0, 18)
    return np.clip(img, 0, 255).astype(np.uint8)


def _random_nac(shape, seed=1):
    """Random NaN-masked frame with no overlap (the no-overlap negative case)."""
    rng = np.random.default_rng(seed)
    nac = np.full(shape, np.nan, np.float32)
    nac[40:-40, 30:-30] = np.clip(rng.random((shape[0] - 80, shape[1] - 60)) * 255,
                                  0, 255).astype(np.float32)
    return nac


def _make_ohrc_ws():
    """OHRC working-set-like canvas: tall (scan) x narrow (pixel)."""
    return _craters((760, 220), seed=10)


def _overlapping_nac(ohrc):
    """Partial-footprint NAC: a pure-translation sub-block of an OHRC interior
    region placed in a NaN-masked frame (mirrors real NAC working-set images;
    translation-only keeps the synthetic geometry unambiguous for SIFT/NCC)."""
    h, w = ohrc.shape
    y0, y1, x0, x1 = 220, 480, 70, 180
    sub = ohrc[y0:y1, x0:x1].astype(np.float32)
    hh, ww = sub.shape
    nac = np.full((340, 150), np.nan, np.float32)
    nac[15:15 + hh, 20:20 + ww] = sub
    return nac


def test_candidate_stats_schema():
    ohrc = _make_ohrc_ws()
    nac = _overlapping_nac(ohrc)
    stats = candidate_stats(ohrc, nac, nfeatures=1500, contrast_threshold=0.004,
                            edge_threshold=10, ratio=0.8)
    assert "candidates" in stats and "sift_params" in stats
    if stats["candidates"]:
        c = stats["candidates"][0]
        for k in ("flip", "good", "inliers", "inlier_ratio", "ncc", "bbox"):
            assert k in c


def test_robust_accepts_overlap():
    ohrc = _make_ohrc_ws()
    nac = _overlapping_nac(ohrc)
    t, stats = find_pair_transform_robust(
        ohrc, nac, nac_fac=1, nfeatures=3000, contrast_threshold=0.004,
        edge_threshold=10, ratio=0.8)
    assert t is not None, "robust search should accept a genuine overlap"
    assert t["inliers"] >= 20
    assert t["ncc"] >= 0.15
    assert t["overlap_bbox"] is not None


def test_robust_rejects_no_overlap():
    ohrc = _make_ohrc_ws()
    nac = _random_nac((340, 150), seed=99)
    t, stats = find_pair_transform_robust(
        ohrc, nac, nac_fac=1, nfeatures=3000, contrast_threshold=0.004,
        edge_threshold=10, ratio=0.8)
    assert t is None, "robust search must NOT hallucinate overlap on noise"


def test_primary_accepts_overlap():
    ohrc = _make_ohrc_ws()
    nac = _overlapping_nac(ohrc)
    t = find_pair_transform(ohrc, nac, nac_fac=1)
    assert t is not None