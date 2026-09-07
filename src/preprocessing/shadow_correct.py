"""Phase 2 shadow region correction (AI_EXECUTION_PLAN.md Step 2.3).

Gamma correction applied only to shadow (low-intensity) regions so dark lava
shadows / terminator areas become visible enough for feature detection while
bright terrain is left unchanged.
"""

from __future__ import annotations

import numpy as np


def gamma_shadow_correct(img, gamma=0.5, shadow_quantile=0.5):
    """Brighten shadow regions with a gamma < 1 curve, leaving highlights intact.

    Args:
        img: uint8 grayscale image (H, W).
        gamma: exponent for shadow pixels; < 1 brightens shadows.
        shadow_quantile: intensity quantile below which a pixel counts as shadow
            (0..1). None keeps a fixed threshold of 0.5 on the 0..1 scale.

    Returns:
        uint8 image (H, W) with brightened shadows.
    """
    if img is None or img.ndim != 2:
        raise ValueError("gamma_shadow_correct expects a grayscale image")
    if not 0 < gamma <= 1.5:
        raise ValueError("gamma must be in (0, 1.5]")
    f = img.astype(np.float32) / 255.0
    if shadow_quantile is not None:
        thresh = float(np.percentile(f, 100 * shadow_quantile))
    else:
        thresh = 0.5
    corrected = np.where(f <= thresh,
                         np.power(np.clip(f, 0.0, 1.0), gamma), f)
    return (np.clip(corrected, 0.0, 1.0) * 255.0).astype(np.uint8)