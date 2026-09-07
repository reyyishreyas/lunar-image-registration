"""Phase 2 image normalization preprocessing (AI_EXECUTION_PLAN.md Steps 2.1-2.2).

Histogram matching and CLAHE contrast enhancement for the pair crops. Both are
applied before feature detection so the two sensors' radiometry become comparable.
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.exposure import match_histograms


def histogram_match(img, reference):
    """Match the histogram of `img` to that of `reference` (uint8 in, uint8 out).

    Args:
        img: input grayscale image (H, W) uint8.
        reference: image whose histogram is the target (H, W) uint8.

    Returns:
        uint8 array (H, W) with img's histogram reshaped to reference's.
    """
    if img is None or reference is None:
        raise ValueError("histogram_match needs img and reference")
    if img.ndim != 2 or reference.ndim != 2:
        raise ValueError("histogram_match expects grayscale 2-D arrays")
    out = match_histograms(img, reference)
    return out.astype(np.uint8)


def apply_clahe(img, clip_limit=2.0, tile_grid=8):
    """Contrast-limited adaptive histogram equalization on a grayscale image.

    Args:
        img: uint8 grayscale image (H, W).
        clip_limit: cv2.createCLAHE clipLimit (higher = more contrast).
        tile_grid: tile grid side length (e.g. 8 -> 8x8 tiles).

    Returns:
        uint8 image (H, W) after CLAHE.
    """
    if img is None or img.ndim != 2:
        raise ValueError("apply_clahe expects a grayscale image")
    if img.dtype != np.uint8:
        raise ValueError("apply_clahe expects uint8 input")
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit),
                            tileGridSize=(int(tile_grid), int(tile_grid)))
    return clahe.apply(img)