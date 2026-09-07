"""Phase 5 Step 5.3 — learned-matcher feasibility on the polar hard case.

The classical (SIFT) georeferencing of the phase5 pair fails (14 inliers < 20).
This script re-runs the overlap search with the learned matchers (SuperPoint+
SuperGlue, then LoFTR) on the SAME working-set images the georeferencer used, so
the outcome is directly comparable: can a learned matcher establish the
OHRC<->NAC overlap that SIFT could not?

Usage: venv/bin/python scripts/run_polar_learned.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

OHRC_IMG = "data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img"
NAC_IMG = "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG"


def pr(*a):
    print(*a, flush=True)


def working_sets():
    """Return (ohrc_ws_uint8, nac_ws_uint8) normalized as the georeferencer uses."""
    from src.preprocessing.georeference import read_ohrc_raw, read_nac_img
    from src.preprocessing.georeference import OHRC_WS_FACTOR as owf

    ohrc = read_ohrc_raw(OHRC_IMG)
    ohrc_ws = ohrc[::owf, ::owf]
    del ohrc
    nac = read_nac_img(NAC_IMG)
    nac_ws = nac[::8, ::8]
    del nac

    lo, hi = np.nanpercentile(ohrc_ws, [2, 98])
    o8 = np.clip((ohrc_ws.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)
    lo, hi = np.nanpercentile(nac_ws, [2, 98])
    sub = nac_ws
    n8 = np.clip((sub.astype(np.float32) - lo) / (hi - lo) * 255, 0, 255)
    del sub
    n8[np.isnan(nac_ws)] = 0
    n8 = n8.astype(np.uint8)
    return o8, n8


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "superglue"
    o8, n8 = working_sets()
    pr(f"OHRC ws shape {o8.shape}, NAC ws shape {n8.shape}")

    def check(match_func, label):
        t = time.time()
        try:
            kp1, d1, kp2, d2, matches = match_func()
        except Exception as e:
            pr(f"[{label}] ERROR: {type(e).__name__}: {e}")
            return None
        n = len(matches)
        if n < 4:
            pr(f"[{label}] {n} matches (<4) -> fail")
            return None
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        H, mask = cv2.findHomography(pts1.reshape(-1, 1, 2), pts2.reshape(-1, 1, 2),
                                     cv2.USAC_MAGSAC, 5.0)
        inl = int(mask.sum()) if mask is not None else 0
        elapsed = round(time.time() - t, 1)
        pr(f"[{label}] matches={n} inliers={inl} ratio={inl / n if n else 0:.3f} "
           f"t={elapsed}s -> {'PASS' if inl >= 20 else 'FAIL (classical baseline 14)'}")
        return inl

    if which == "superglue":
        def sg():
            from src.detection.learned import detect_superpoint
            from src.matching.learned_match import match_superglue

            kp1, d1 = detect_superpoint(o8, verbose=False, max_keypoints=1024)
            kp2, d2 = detect_superpoint(n8, verbose=False, max_keypoints=1024)
            matches = match_superglue(kp1, d1, kp2, d2, weights="outdoor",
                                      match_threshold=0.2, verbose=False)
            return kp1, d1, kp2, d2, matches

        check(sg, "SuperPoint+SuperGlue")
    else:
        def loftr():
            from src.detection.learned import detect_superpoint
            from src.matching.learned_match import match_loftr

            target = 832
            limg = cv2.resize(o8, (target, target), interpolation=cv2.INTER_AREA)
            rimg = cv2.resize(n8, (target, target), interpolation=cv2.INTER_AREA)
            kp1, d1 = detect_superpoint(limg, verbose=False, max_keypoints=1024)
            kp2, d2 = detect_superpoint(rimg, verbose=False, max_keypoints=1024)
            matches = match_loftr(kp1, d1, kp2, d2, weights="outdoor", verbose=False)
            return kp1, d1, kp2, d2, matches

        check(loftr, "LoFTR (resized 832x832)")


if __name__ == "__main__":
    main()