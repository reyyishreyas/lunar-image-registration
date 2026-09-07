"""Phase 1 metrics (AI_EXECUTION_PLAN.md Step 1.7).

rmse(H, gt_points): registration error in pixels.
"""

from __future__ import annotations

import numpy as np


def apply_homography(H, points):
    """Transform (N,2) points by H; returns homogeneous-normalized (N,2)."""
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, 2)
    ones = np.ones((pts.shape[0], 1))
    xy1 = np.hstack([pts, ones])
    out = (H @ xy1.T).T
    x, y, w = out[:, 0], out[:, 1], out[:, 2]
    x = x / np.where(np.abs(w) < 1e-12, 1e-12, w)
    y = y / np.where(np.abs(w) < 1e-12, 1e-12, w)
    return np.stack([x, y], axis=1)


def rmse(H, gt_points):
    """Root-mean-square registration error.

    Args:
        H: 3x3 homography mapping image-1 coordinates to image-2 coordinates.
        gt_points: (N, 4) array of manual control points, columns
                   (x1, y1, x2, y2): (x1,y1) in image1, (x2,y2) in image2.

    Returns:
        (rmse_px, residuals): float RMSE in pixels and per-point residual array.
    """
    gt = np.asarray(gt_points, dtype=np.float64)
    if gt.size == 0:
        raise ValueError("rmse requires ground-truth points")
    p1 = gt[:, :2]
    p2 = gt[:, 2:]
    pred2 = apply_homography(H, p1)
    residuals = np.linalg.norm(pred2 - p2, axis=1)
    return float(np.sqrt(np.mean(residuals ** 2))), residuals