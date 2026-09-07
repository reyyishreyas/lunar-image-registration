"""Phase 1 classical descriptor matching (AI_EXECUTION_PLAN.md Step 1.3).

Baseline matcher for Config C1: BFMatcher with Lowe's ratio test.
"""

from __future__ import annotations

import cv2


def match_bf_ratio(desc1, desc2, ratio=0.75, norm_type=cv2.NORM_L2, verbose=False):
    """Brute-force kNN match with Lowe's ratio test.

    Args:
        desc1: descriptors from image 1 (query).
        desc2: descriptors from image 2 (train).
        ratio: Lowe ratio threshold on the two nearest neighbours.
        norm_type: cv2.NORM_L2 (SIFT).

    Returns:
        list of cv2.DMatch retained after the ratio test. Each DMatch.
        queryIdx refers to desc1, trainIdx to desc2.
    """
    if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
        return []
    bf = cv2.BFMatcher(norm_type)
    knn = bf.knnMatch(desc1, desc2, k=2)
    good = [m for m, n in knn if m.distance < ratio * n.distance]
    good = sorted(good, key=lambda m: m.distance)
    if verbose:
        print(f"[match_bf_ratio] good_matches={len(good)}")
    return good