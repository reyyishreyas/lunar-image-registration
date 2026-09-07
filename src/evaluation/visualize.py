"""Phase 1 visualization (AI_EXECUTION_PLAN.md Step 1.5).

Wraps cv2.drawMatches; saves a side-by-side matches figure.
"""

from __future__ import annotations

import os

import cv2


def draw_matches(img1, kp1, img2, kp2, matches, out_path,
                 inlier_mask=None, max_draw=100):
    """Save a side-by-side feature-match visualization to out_path."""
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    def to_rgb(im):
        if im.ndim == 2:
            return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        return im

    r1, r2 = to_rgb(img1), to_rgb(img2)
    selected = matches[:max_draw]
    if inlier_mask is None:
        mask = None
    else:
        import numpy as np

        mask = np.array([bool(inlier_mask[i]) for i, _ in enumerate(matches)],
                        dtype=np.uint8)[:max_draw].reshape(-1)

    vis = cv2.drawMatches(r1, kp1, r2, kp2, selected, None,
                          matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
                          matchesMask=mask, flags=2)
    cv2.imwrite(out_path, vis)
    return out_path