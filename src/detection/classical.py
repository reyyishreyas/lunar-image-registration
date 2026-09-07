"""Phase 1 classical feature detection (AI_EXECUTION_PLAN.md Step 1.2).

Baseline detector for Config C1: SIFT.
"""

from __future__ import annotations

import cv2


def detect_sift(img, nfeatures=10000, contrast_threshold=0.04, edge_threshold=10,
                verbose=False):
    """Detect SIFT keypoints/descriptors on a single-band image.

    Args:
        img: grayscale image (H, W).
        nfeatures: max number of keypoints.
        contrast_threshold: SIFT contrast cutoff (raise to reduce keypoints).
        edge_threshold: SIFT edge cutoff.

    Returns:
        (keypoints, descriptors): cv2.KeyPoint list and int32 descriptor array
        (None descriptor array when no keypoints are found).
    """
    if img is None or img.ndim != 2:
        raise ValueError("detect_sift expects a grayscale image")
    gray = img
    if gray.dtype != "uint8":
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    sift = cv2.SIFT_create(nfeatures=nfeatures,
                           contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold)
    keypoints, descriptors = sift.detectAndCompute(gray, None)
    if verbose:
        print(f"[detect_sift] keypoints={0 if keypoints is None else len(keypoints)}")
    return keypoints, descriptors


def detect_akaze(img, nfeatures=10000, verbose=False):
    """Detect AKAZE (binary MLDB) keypoints/descriptors on a grayscale image.

    Args:
        img: grayscale image (H, W) uint8.
        nfeatures: maximum number of keypoints to retain (sorted by response).
        verbose: print keypoint count.

    Returns:
        (keypoints, descriptors): cv2.KeyPoint list and int32 descriptor array,
        where descriptors are binary MLDB vectors (match with NORM_HAMMING).
    """
    if img is None or img.ndim != 2:
        raise ValueError("detect_akaze expects a grayscale image")
    gray = img
    if gray.dtype != "uint8":
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    akaze = cv2.AKAZE_create()
    keypoints, descriptors = akaze.detectAndCompute(gray, None)
    if nfeatures and len(keypoints) > nfeatures:
        keypoints, descriptors = keypoints[:nfeatures], descriptors[:nfeatures]
    if verbose:
        print(f"[detect_akaze] keypoints={0 if keypoints is None else len(keypoints)}")
    return keypoints, descriptors