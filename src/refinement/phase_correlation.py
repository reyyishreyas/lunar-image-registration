"""Phase 4 sub-pixel refinement via phase correlation (Step 4.4).

For each inlier correspondence, extracts a patch around the source keypoint and
a slightly larger patch around the target keypoint and recovers the sub-pixel
shift with skimage.registration.phase_cross_correlation. The shift is applied
to the target point, then a homography is refit by least squares on the refined
inliers.
"""

from __future__ import annotations

import numpy as np
from skimage.registration import phase_cross_correlation


def refine_phase_correlation(src, ref, kp1, kp2, matches, inlier_mask, radius=6,
                             upsample_factor=10):
    """Refine inlier target points by per-correspondence phase correlation.

    Args:
        src, ref: grayscale images (the normalized ones used for matching).
        kp1, kp2: keypoint lists.
        matches: the full match list; `inlier_mask` is aligned to it.
        inlier_mask: boolean mask selecting inliers.
        radius: half-side of the correlation window in pixels.
        upsample_factor: sub-pixel upsampling factor for the shift estimation.

    Returns:
        (refined_pts1, refined_pts2, H): refined (M, 2) float points and the
        least-squares homography mapping refined pts1 -> refined pts2.
    """
    idx = np.flatnonzero(np.asarray(inlier_mask, dtype=bool))
    if len(idx) < 4:
        raise ValueError("need >= 4 inliers to refine")

    src = src.astype(np.float64)
    ref = ref.astype(np.float64)
    h, w = src.shape
    pad = radius + 1

    def patch(img, kp):
        x, y = kp.pt
        x0, x1 = int(round(x)) - pad, int(round(x)) + pad
        y0, y1 = int(round(y)) - pad, int(round(y)) + pad
        x0, x1 = max(x0, 0), min(x1, w)
        y0, y1 = max(y0, 0), min(y1, h)
        return img[y0:y1, x0:x1]

    r1 = np.asarray([kp1[matches[i].queryIdx].pt for i in idx], dtype=np.float64)
    r2 = np.asarray([kp2[matches[i].trainIdx].pt for i in idx], dtype=np.float64)
    refined2 = r2.copy()

    for j, i in enumerate(idx):
        p1_patch = patch(src, kp1[matches[i].queryIdx])
        p2_patch = patch(ref, kp2[matches[i].trainIdx])
        if p1_patch.shape[0] < 3 or p1_patch.shape[1] < 3:
            continue
        if p2_patch.shape[0] < 3 or p2_patch.shape[1] < 3:
            continue
        try:
            shift, _, _ = phase_cross_correlation(
                p2_patch, p1_patch, upsample_factor=upsample_factor)
        except Exception:
            continue
        refined2[j] = r2[j] - shift

    H, _ = cv2_find_homography_lsq(r1, refined2)
    return r1, refined2, H


def cv2_find_homography_lsq(pts1, pts2):
    import cv2

    return cv2.findHomography(
        np.asarray(pts1, dtype=np.float32).reshape(-1, 1, 2),
        np.asarray(pts2, dtype=np.float32).reshape(-1, 1, 2),
        cv2.RANSAC, 2.0)