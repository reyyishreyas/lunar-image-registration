"""Phase 1 outlier rejection with RANSAC (AI_EXECUTION_PLAN.md Step 1.4).

Baseline: cv2.findHomography with RANSAC.
"""

from __future__ import annotations

import cv2
import numpy as np


def find_homography_ransac(pts1, pts2, ransac_thresh=5.0, confidence=0.995,
                           max_iters=5000):
    """Estimate a 3x3 homography mapping pts1 -> pts2 with RANSAC.

    Args:
        pts1: (N, 2) float array of source points (x, y).
        pts2: (N, 2) float array of destination points (x, y).
        ransac_thresh: RANSAC reprojection threshold in pixels.
        confidence: (unused) kept for API compatibility.
        max_iters: (unused) kept for API compatibility.

    Returns:
        (H, inlier_mask): homography (3x3) or None, inlier boolean mask (N,).
    """
    if len(pts1) < 4 or len(pts2) < 4 or len(pts1) != len(pts2):
        raise ValueError("need >= 4 matched correspondences")
    src = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, ransac_thresh)
    if mask is None:
        return None, None
    return H, mask.ravel().astype(bool)