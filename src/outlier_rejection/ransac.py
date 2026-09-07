"""Phase 1 outlier rejection with RANSAC (AI_EXECUTION_PLAN.md Step 1.4).

Baseline: cv2.findHomography with RANSAC. Phase 4 Step 4.2 adds the
USAC_MAGSAC estimator, selectable via the `method` parameter.
"""

from __future__ import annotations

import cv2
import numpy as np

_METHODS = {
    "ransac": cv2.RANSAC,
    "usac_magsac": cv2.USAC_MAGSAC,
}


def find_homography_ransac(pts1, pts2, ransac_thresh=5.0, confidence=0.995,
                           max_iters=5000, method="ransac"):
    """Estimate a 3x3 homography mapping pts1 -> pts2 with robust estimation.

    Args:
        pts1: (N, 2) float array of source points (x, y).
        pts2: (N, 2) float array of destination points (x, y).
        ransac_thresh: RANSAC reprojection threshold in pixels.
        confidence: RANSAC confidence level.
        max_iters: maximum RANSAC iterations.
        method: one of "ransac" or "usac_magsac" (Phase 4 Step 4.2).

    Returns:
        (H, inlier_mask): homography (3x3) or None, inlier boolean mask (N,).
    """
    if len(pts1) < 4 or len(pts2) < 4 or len(pts1) != len(pts2):
        raise ValueError("need >= 4 matched correspondences")
    src = np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2)
    dst = np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2)
    flags = _METHODS[method]
    H, mask = cv2.findHomography(src, dst, flags, ransac_thresh,
                                 confidence=confidence, maxIters=max_iters)
    if mask is None:
        return None, None
    return H, mask.ravel().astype(bool)