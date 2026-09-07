"""Phase 4 sub-pixel refinement of inlier correspondences (Step 4.3).

Refines inlier keypoint locations on both images with cv2.cornerSubPix inside a
small search window, then refits the homography by least squares on the refined
inliers.
"""

from __future__ import annotations

import cv2
import numpy as np


def refine_corner_subpix(src, ref, kp1, kp2, matches, inlier_mask, win_size=5,
                         max_iter=40, eps=1e-3):
    """Refine inlier point locations using cornerSubPix.

    Args:
        src, ref: uint8/float grayscale images (the normalized ones used for
            matching).
        kp1, kp2: keypoint lists.
        matches: the full match list; `inlier_mask` is aligned to it.
        inlier_mask: boolean mask over `matches` selecting inliers.
        win_size: half-width of the cornerSubPix search window.
        max_iter, eps: termination criteria for cornerSubPix.

    Returns:
        (refined_pts1, refined_pts2, H): refined (M, 2) float points on both
        sides and the least-squares homography mapping refined pts1 -> pts2.
    """
    idx = np.flatnonzero(np.asarray(inlier_mask, dtype=bool))
    if len(idx) < 4:
        raise ValueError("need >= 4 inliers to refine")
    p1 = np.asarray([kp1[matches[i].queryIdx].pt for i in idx], dtype=np.float32)
    p2 = np.asarray([kp2[matches[i].trainIdx].pt for i in idx], dtype=np.float32)

    if src.dtype != np.uint8:
        src8 = cv2.normalize(src, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        ref8 = cv2.normalize(ref, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        src8, ref8 = src, ref

    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, max_iter, eps)
    r1 = cv2.cornerSubPix(src8, p1.reshape(-1, 1, 2), (win_size, win_size),
                          (-1, -1), crit)
    r2 = cv2.cornerSubPix(ref8, p2.reshape(-1, 1, 2), (win_size, win_size),
                          (-1, -1), crit)
    r1 = r1.reshape(-1, 2)
    r2 = r2.reshape(-1, 2)

    H, _ = cv2.findHomography(r1.reshape(-1, 1, 2), r2.reshape(-1, 1, 2),
                              cv2.RANSAC, 2.0)
    return r1, r2, H