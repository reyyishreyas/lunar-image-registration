"""Phase 9 — Cross-modal (multi-sensor / radiometric-invariant) correspondence.

PS 26166 headline: "Multi-modal, Sun angle and scale invariant image
correspondence" across Chandrayaan-2 OHRC, TMC, IIRS and lunar reference images.
Feature descriptors (SIFT etc.) are capture-window based and collapse when two
sensors' radiometry differs (e.g. OHRC visible vs IIRS thermal/near-IR,
grazing sun). This module closes that gap with:

  1. Radiometric normalisation front-ends that project each sensor's band into a
     shared, geometry-dominant representation before matching:
       * ``histogram_match``   -> match_histograms (already in preprocessing)
       * ``gradient_structure``-> normalized local gradient magnitude + orientation
                                  (edge map; the strongest geometry-dominant cue)
       * ``log_ratio``         -> log of the local-normalized response (contrast
                                  invariant to overall brightness/scaling)
       * ``clahe``             -> contrast-limited local equalization
  2. ``cross_modal_register`` — a self-verifying matcher that runs several
     front-ends, matches each with SIFT (RANSAC via the shared ransac module),
     and returns the candidate whose fitted homography is best supported
     (proof-based: inlier dominance + reprojection). This mirrors the
     ``find_pair_transform_robust`` acceptance philosophy already used for
     georeferencing, so multi-modal registration is never silently garbage.

The module reuses the repository pipeline building blocks (preprocessing /
normalize, detection / classical, outlier_rejection / ransac); it does not
reimplement OpenCV.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.preprocessing.normalize import histogram_match, apply_clahe
from src.detection.classical import detect_sift
from src.outlier_rejection.ransac import find_homography_ransac

NORM_FRONTS = ("gradient_structure", "histogram_match", "log_ratio", "clahe", "none")


# --------------------------------------------------------------------------- #
# Radiometric normalisation front-ends (all preserve geometry)
# --------------------------------------------------------------------------- #
def gradient_structure(img, *, ksize=3, mag_weight=1.0, keep_orientation=False):
    """Geometry-dominant representation: normalised gradient magnitude (+ opt. orientation).

    The Sobel gradient magnitude is largely insensitive to absolute brightness,
    illumination gradient and sensor response curve, so two bands of different
    radiometry but the same shape map to a common edge structure. Optionally
    incorporates the gradient orientation into the response for stronger
    geometric cuing (discriminates slope direction in grazing-light surfaces).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=ksize)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    if keep_orientation:
        ori = np.arctan2(gy, gx)
        mag = mag * (1.0 + 0.5 * np.cos(2 * ori))
    mag *= float(mag_weight)
    mx = float(mag.max()) if mag.size else 0.0
    if mx > 0:
        mag = mag / mx * 255.0
    return np.clip(mag, 0, 255).astype(np.uint8)


def log_ratio(img, *, ksize=31):
    """Log of the local-mean-normalised intensity (brightness/scale invariant).

    ``log( img / blur(img) + eps )`` highlights local radiometric contrast while
    removing the global (orbital altitude / sensor saturation) offset that
    separates sensors. Output is standardised to full uint8 range.
    """
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (ksize, ksize), 0)
    ratio = np.log1p(f) - np.log1p(np.maximum(blur, 1.0))
    lo, hi = float(np.percentile(ratio, 2)), float(np.percentile(ratio, 98))
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    out = np.clip((ratio - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)
    return out


def _apply_front(img, name, ref=None, params=None):
    """Apply a normalisation front to ``img``. ``ref`` only needed by histogram_match."""
    params = params or {}
    if name == "gradient_structure":
        return gradient_structure(img, **params)
    if name == "histogram_match":
        return histogram_match(img, ref) if ref is not None else img
    if name == "log_ratio":
        return log_ratio(img, **params)
    if name == "clahe":
        return apply_clahe(img, **dict({"clip_limit": 4.0, "tile_grid": 8}, **params))
    if name in ("none", ""):
        return img
    raise ValueError(f"unknown cross-modal front-end {name!r}")


# --------------------------------------------------------------------------- #
# Cross-modal detection + matching with proof-based acceptance
# --------------------------------------------------------------------------- #
def detect_cross_modal(img, *, front="gradient_structure", ref=None, front_params=None,
                       nfeatures=20000, contrast_threshold=0.02, verbose=False, **detect):
    """Normalise ``img`` to the chosen front-end, then run SIFT detection.

    Returns (keypoints, descriptors, front_image). If ``front`` is one of the
    geometry-dominant maps the descriptors are computed on geometry, which is
    what makes cross-sensor matching possible.
    """
    fi = _apply_front(img if img.dtype == np.uint8
                      else np.clip((img - img.min()) / max(1e-6, (img.max() -
                        img.min())) * 255, 0, 255).astype(np.uint8),
                      front, ref=ref, params=front_params)
    return detect_sift(fi, nfeatures=nfeatures,
                       contrast_threshold=contrast_threshold, verbose=verbose, **detect), fi


def cross_modal_register(src, ref, *, fronts=NORM_FRONTS,
                         ratio=0.75, ransac_thresh=5.0, min_inliers=12,
                         min_ratio=0.05, nfeatures=20000, contrast_threshold=0.02,
                         verbose=False):
    """Self-verifying cross-modal registration over several front-ends.

    For each normalisation front-end:
      1. detect SIFT on both, match by BF ratio,
      2. robust-fit a homography (USAC_MAGSAC),
      3. record inliers / ratio / RMSE.

    Returns the best front-end candidate (highest inlier count passing the
    acceptance: inliers >= min_inliers and inlier_ratio >= min_ratio), plus all
    per-front stats for transparency. Returns None when no front-end passes —
    the caller then reports a non-registration (never a fabricated model).

    Returns dict with keys: front, H, inliers, inlier_ratio, n_matches, rmse,
    kp1, kp2, per_front (list of per-front-end stat dicts).
    """
    per_front = []
    best = None
    for front in fronts:
        kp1, d1 = detect_sift(_apply_front(src, front, ref=ref), nfeatures=nfeatures,
                              contrast_threshold=contrast_threshold)
        kp2, d2 = detect_sift(_apply_front(ref, front, ref=ref), nfeatures=nfeatures,
                              contrast_threshold=contrast_threshold)
        n1 = 0 if kp1 is None else len(kp1)
        n2 = 0 if kp2 is None else len(kp2)
        if n1 < 8 or n2 < 8:
            per_front.append(dict(front=front, n_matches=0, inliers=0,
                                  inlier_ratio=0.0, note="too few keypoints"))
            continue
        # match with the same ratio strategy as the champion matcher
        from src.matching.classical_match import match_bf_ratio
        m = match_bf_ratio(d1, d2, ratio=ratio, norm_type=cv2.NORM_L2,
                           verbose=False)
        if m is None or len(m) < 8:
            per_front.append(dict(front=front, n_matches=0, inliers=0,
                                  inlier_ratio=0.0, note="few matches"))
            continue
        pts1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
        pts2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
        H, inl = find_homography_ransac(pts1, pts2, ransac_thresh=ransac_thresh,
                                        method="usac_magsac")
        stat = dict(front=front, n_matches=len(m), note="")
        if H is None or inl is None:
            stat.update(inliers=0, inlier_ratio=0.0, note="RANSAC failed")
            per_front.append(stat)
            continue
        nin = int(inl.sum())
        stat.update(inliers=nin, inlier_ratio=round(float(nin) / len(m), 4))
        per_front.append(stat)
        if nin >= min_inliers and (nin / len(m)) >= min_ratio:
            pw = pts1[inl]
            pw_h = np.hstack([pw, np.ones((pw.shape[0], 1))])
            proj = (H @ pw_h.T).T
            with np.errstate(divide="ignore", invalid="ignore"):
                proj = proj[:, :2] / np.where(proj[:, 2:3] == 0, np.nan,
                                              proj[:, 2:3])
            diffs = np.sqrt(np.nansum((proj - pts2[inl]) ** 2, axis=1))
            rmse = float(np.sqrt(np.nanmean(diffs ** 2))) if np.isfinite(
                diffs).any() else float("inf")
            stat["rmse"] = rmse
            # sanity: reject exploded / degenerate models. Also require the
            # inlier set to actually span the image (a cluster of near-identical
            # points gives ~0 RMSE but a meaningless homography), and the
            # homography itself to be well conditioned (RANSAC occasionally
            # returns near-singular matrices that compress the domain
            # invisibly in the inlier frame but garbage globally).
            span = (pw.max(axis=0) - pw.min(axis=0))
            span_px = float(np.sqrt(np.sum(span ** 2)))
            min_span = 0.15 * float(np.sqrt(src.shape[0] ** 2 + src.shape[1] ** 2))
            cond = float(np.linalg.cond(H))
            scale_ok = min(abs(H[0, 0]), abs(H[1, 1])) > 1e-2
            if (not np.isfinite(rmse) or np.isnan(rmse) or rmse > 25.0
                    or span_px < min_span or cond > 1e6 or not scale_ok):
                stat["note"] = (f"degenerate fit (rmse={rmse:.3g}, "
                                f"span={span_px:.1f}px, cond={cond:.1e})")
                continue
            stat["H"] = H
            stat["kp1"], stat["kp2"] = kp1, kp2
            stat["pts1"], stat["pts2"] = pts1, pts2
            stat["inl"] = inl
            stat["matches"] = m
            if best is None or nin > best["inliers"]:
                best = dict(stat)
    if best is not None:
        best["method"] = f"cross_modal:{best['front']}"
        best["per_front"] = per_front
        if verbose:
            print(f"[cross_modal_register] best front={best['front']} "
                  f"inliers={best['inliers']} ratio={best['inlier_ratio']}")
    return best


def best_front_name(stats):
    """Human label for the winning cross-modal front-end (or n/a)."""
    if not stats or stats.get("front") is None:
        return "n/a"
    return stats["front"]


def run_cross_modal(src, ref, **kw):
    """Convenience wrapper returning (ok:bool, result-dict) — see cross_modal_register."""
    res = cross_modal_register(src, ref, **kw)
    if res is None:
        return False, {"fronts": NORM_FRONTS, "ok": False}
    res["ok"] = True
    return True, res
