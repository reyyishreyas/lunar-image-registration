"""Fully-automatic registration pipeline (Phase 8).

Drop in the three inputs and this module does everything:

    OHRC .img  +  OHRC geometry CSV  +  NAC .IMG

Automatic steps:
  1. Georeference / stage both sensors to a common ground grid (SIFT anchor +
     equal-GSD self-calibration when geometry is available).
  2. Detect + match + robust-fit the champion config (SIFT + CLAHE + USAC_MAGSAC),
     then iteratively refit the homography to the tight sub-pixel set.
  3. Decide the outcome automatically:
       * content registration  -> overlay, matches, fitted homography, RMSE
       * no content correspondence (photometric gap, e.g. polar pair-2)
         -> geometry-based registration (SPICE + ISRO CSV) with an honest
            "feature correspondence not verifiable" note
       * too few reliable inliers -> explicit non-registration failure (never
            silently returns garbage)
  4. Emit a single self-contained report: {pair, method, RMSE, inliers, ratio,
     homography, staged GSD, coverage, verdict, artifacts}.

This is the single auto entry point shared by scripts/run_auto.py (CLI) and the
Streamlit "Automatic" page. It intentionally reuses the repository pipeline
modules (src/pipeline, src/preprocessing/*, src/evaluation/*) rather than
reimplementing any matching logic.
"""

from __future__ import annotations

import csv
import json
import os
import re
import tempfile

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINAL_CONFIG = "results/final_config.yaml"


# --------------------------------------------------------------------------- #
# small helpers (homography fit / residual — kept local & dependency-light)
# --------------------------------------------------------------------------- #
def _lsq_homography(pts1, pts2):
    """Least-squares homography pts1 -> pts2 (DLT, 8-dof)."""
    M, v = [], []
    for x, y, xx, yy in zip(pts1[:, 0], pts1[:, 1], pts2[:, 0], pts2[:, 1]):
        M.append([x, y, 1, 0, 0, 0, -x * xx, -y * xx])
        v.append(xx)
        M.append([0, 0, 0, x, y, 1, -x * yy, -y * yy])
        v.append(yy)
    h, _, _, _ = np.linalg.lstsq(np.float64(M), np.array(v, float), rcond=None)
    return np.concatenate([h, [1.0]]).reshape(3, 3)


def _apply_homography(H, pts):
    pts = np.asarray(pts, np.float64)
    ones = np.ones((pts.shape[0], 1))
    out = (H @ np.hstack([pts, ones]).T).T
    x, y, w = out[:, 0], out[:, 1], out[:, 2]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return np.stack([x / w, y / w], axis=1)


def _tighten(pts1, pts2, thresholds=(5.0, 3.0, 2.0, 1.5)):
    """Iteratively fit + threshold-tighten a homography to a tight sub-pixel set.

    Seeds from a robust USAC_MAGSAC fit so a few gross outliers can never
    destabilise the first least-squares model, then iteratively tightens.
    Returns (H, keep, res); keep is a boolean mask over the input array.
    """
    from src.outlier_rejection.ransac import find_homography_ransac

    H0, inl = find_homography_ransac(pts1, pts2, ransac_thresh=5.0,
                                     method="usac_magsac")
    if H0 is None or inl is None or inl.sum() < 8:
        return _lsq_homography(pts1, pts2), np.ones(len(pts1), bool), \
            np.linalg.norm(_apply_homography(_lsq_homography(pts1, pts2),
                                             pts1) - pts2, axis=1)

    keep = inl.copy()
    H = _lsq_homography(pts1[keep], pts2[keep])
    for t in thresholds:
        if keep.sum() < 8:
            break
        r = np.linalg.norm(_apply_homography(H, pts1) - pts2, axis=1)
        keep = keep & (r < t)
        if keep.sum() < 4:
            break
        H = _lsq_homography(pts1[keep], pts2[keep])
    r = np.linalg.norm(_apply_homography(H, pts1) - pts2, axis=1)
    # final clean refit on the tight set
    if keep.sum() >= 4:
        H = _lsq_homography(pts1[keep], pts2[keep])
        r = np.linalg.norm(_apply_homography(H, pts1) - pts2, axis=1)
    return H, keep, r


def _rmse(H, pts1, pts2):
    r = np.linalg.norm(_apply_homography(H, np.asarray(pts1)) -
                       np.asarray(pts2), axis=1)
    return float(np.sqrt(np.mean(r ** 2))), r


def _attach_decision(report, out_dir="", prefix="", join=lambda p: p,
                     src=None, ref=None, H=None):
    """Attach the definitive MATCH / NO MATCH decision + best product.

    For a content ``registered`` pair with a homography we also write the best
    aligned product: the source warped onto the reference frame (``_aligned``)
    plus a Turbo difference map against the reference.
    """
    try:
        if src is not None and ref is not None and H is not None:
            h, w = ref.shape[:2]
            warped = cv2.warpPerspective(src, H, (w, h))
            aligned = os.path.join(out_dir, f"{prefix}_aligned.png")
            if os.path.isdir(out_dir):
                cv2.imwrite(join(aligned), warped)
                diff = np.abs(warped.astype(np.float32)
                              - ref.astype(np.float32))
                d_path = os.path.join(out_dir, f"{prefix}_diff.png")
                cv2.imwrite(join(d_path), np.clip(diff, 0, 255).astype(np.uint8))
                report.setdefault("artifacts", {})["best_aligned"] = aligned
                report["artifacts"]["diff"] = d_path
    except Exception as exc:  # noqa: BLE001
        report.setdefault("notes", []).append(
            f"best-aligned product not written: {exc}")

    from src.evaluation.decision import classify
    report["decision"] = classify(report)
    return report


# --------------------------------------------------------------------------- #
# core automatic routine
# --------------------------------------------------------------------------- #
def run_auto(ohrc_img, ohrc_geom, nac_img, *, out_dir="data/processed/auto",
             prefix="auto", nac_geom_csv="", ground_truth="",
             normalize="clahe", crop_px=1024, equal_gsd=False, root=PROJECT_ROOT,
             use_pipeline=True, verbose=True, seed=0, fresh=True):
    """Run the whole registration automatically for one OHRC<->NAC pair.

    Parameters mirror the repository config keys so callers can pass real paths.
    Returns a single report dict (JSON-serialisable). Never raises for an
    un-registerable pair — it records an explicit verdict instead.

    ``seed`` makes the georeference flip-decision and RANSAC deterministic;
    ``fresh`` clears the staged crops for this prefix first so a stale crop from
    a previous equal-GSD / flip variant can never contaminate the result.
    """
    # deterministic RANSAC + SIFT so the flip decision is reproducible
    cv2.setRNGSeed(int(seed))
    np.random.seed(int(seed))
    join = lambda p: p if os.path.isabs(p) else os.path.join(root, p)  # noqa: E731
    os.makedirs(join(out_dir), exist_ok=True)
    if fresh:
        for f in (f"{prefix}_src.png", f"{prefix}_ref.png",
                  f"georef_{prefix}.json"):
            p = join(os.path.join(out_dir, f))
            if os.path.exists(p):
                try:
                    os.remove(p)
                except OSError:
                    pass
    report = {
        "pair": {"ohrc_img": ohrc_img, "ohrc_geometry": ohrc_geom,
                 "nac_img": nac_img},
        "config": {"crop_px": crop_px, "equal_gsd": equal_gsd,
                   "normalize": normalize, "nac_geom_csv": nac_geom_csv,
                   "ground_truth": ground_truth},
        "out_dir": out_dir,
        "method": None, "verdict": None, "rmse_px": None,
        "rmse_self_px": None, "inliers": 0, "inlier_ratio": 0.0,
        "n_matches": 0, "homography": None, "staged_gsd_m": None,
        "artifacts": {}, "notes": [],
    }

    # ---- 1. stage (georeference + equal-GSD + precondition) ----
    try:
        if use_pipeline:
            import yaml
            with open(join(FINAL_CONFIG)) as fh:
                base = yaml.safe_load(fh)
            pre = dict(base["preprocessing"])
            pre.update({
                "ohrc_img": ohrc_img, "ohrc_geometry": ohrc_geom,
                "nac_img": nac_img, "out_dir": out_dir,
                "prefix": prefix, "crop_px": crop_px,
                "equal_gsd": equal_gsd, "nac_geom_csv": nac_geom_csv,
                "outputs": {"src": os.path.join(out_dir, f"{prefix}_src.png"),
                            "ref": os.path.join(out_dir, f"{prefix}_ref.png")},
            })
            norm = dict(base["normalize"])
            if normalize != "none":
                norm["enabled"] = True
                norm["method"] = normalize
            else:
                norm["enabled"] = False

            from src.preprocessing.staging import stage_pair
            st = stage_pair(pre, norm, root=root)
            src, ref = st["src"], st["ref"]
            report["staged_gsd_m"] = st["gsd_m"]
            if st["meta"]:
                report["notes"].append(
                    f"georeference method={st['meta'].get('georef_method')}, "
                    f"flip={st['meta'].get('nac_flip')}")
        else:
            # minimal fallback: read the staged PNGs directly
            src = cv2.imread(join(os.path.join(out_dir, f"{prefix}_src.png")),
                             cv2.IMREAD_GRAYSCALE)
            ref = cv2.imread(join(os.path.join(out_dir, f"{prefix}_ref.png")),
                             cv2.IMREAD_GRAYSCALE)
            from src.preprocessing.staging import apply_precondition
            src, ref, _ = apply_precondition(src, ref, {"enabled": normalize != "none",
                                                        "method": normalize})
            if src is None or ref is None:
                report["verdict"] = "input_error"
                report["notes"].append(
                    "could not read staged crops; ensure georeferencing wrote "
                    f"{prefix}_src.png / {prefix}_ref.png")
                return _attach_decision(report, out_dir=out_dir, prefix=prefix)
    except Exception as exc:  # noqa: BLE001
        # content-style staging failed for this pair (e.g. incompatible CSV
        # format, photometric/polar case). If a NAC geometry CSV is available we
        # still try the SPICE+ISRO geometry route instead of hard-failing.
        report["method"] = "geometry"
        report["verdict"] = "no_content_correspondence"
        report["notes"].append(f"content staging/georeference failed: {exc}")
        diag = os.path.join(out_dir, f"georef_{prefix}_diagnostics.json")
        if os.path.exists(join(diag)):
            report["notes"].append(f"georef diagnostics: {diag}")
        if nac_geom_csv:
            return _geometry_fallback(report, join, out_dir, prefix,
                                      ohrc_img, ohrc_geom, nac_img,
                                      nac_geom_csv, root)
        return _attach_decision(report, out_dir=out_dir, prefix=prefix)

    # ---- 2. detect + match + robust fit (champion config: SIFT + CLAHE) ----
    try:
        from src.detection.classical import detect_sift
        from src.matching.classical_match import match_bf_ratio

        kp1, d1 = detect_sift(src, verbose=False, nfeatures=10000,
                              contrast_threshold=0.04, edge_threshold=10)
        kp2, d2 = detect_sift(ref, verbose=False, nfeatures=10000,
                              contrast_threshold=0.04, edge_threshold=10)
        m = match_bf_ratio(d1, d2, ratio=0.75, norm_type=cv2.NORM_L2,
                           verbose=False)
        pts1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
        pts2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
        report["n_matches"] = len(m)

        # robust initial fit (USAC_MAGSAC, 5px)
        from src.outlier_rejection.ransac import find_homography_ransac
        H0, inl = find_homography_ransac(pts1, pts2, ransac_thresh=5.0,
                                         method="usac_magsac")
        if H0 is None or inl is None or inl.sum() < 8:
            # ─ too few reliable inliers → non-content pair ─
            report["method"] = "geometry"
            report["verdict"] = "no_content_correspondence"
            report["inliers"] = int(inl.sum()) if inl is not None else 0
            report["notes"].append(
                "too few reliable inliers (< 8) from content matching; "
                "trying geometry-based registration")
            if nac_geom_csv:
                return _geometry_fallback(report, join, out_dir, prefix,
                                          ohrc_img, ohrc_geom, nac_img,
                                          nac_geom_csv, root)
            return report
        report["inliers"] = int(inl.sum())
        report["inlier_ratio"] = round(float(inl.sum()) / len(m), 4)

        # sub-pixel tightening on the MAGSAC inliers
        Ht, keep, r_self = _tighten(pts1, pts2,
                                    thresholds=(5.0, 3.0, 2.0, 1.5))
        rmse_self, _ = _rmse(Ht, pts1[keep], pts2[keep])
        report["homography"] = Ht.tolist()
        report["rmse_self_px"] = round(rmse_self, 4)
        report["inliers"] = int(keep.sum())
        report["inlier_ratio"] = round(float(keep.sum()) / len(m), 4)

        # RMSE against provided ground truth (if any)
        if ground_truth and os.path.exists(join(ground_truth)):
            gt = np.loadtxt(join(ground_truth), delimiter=",", skiprows=1)
            rmse_gt, _ = _rmse(Ht, gt[:, :2], gt[:, 2:])
            report["rmse_px"] = round(float(rmse_gt), 4)
        report["method"] = "content"
        report["verdict"] = "registered"
        _attach_decision(report, out_dir=out_dir, prefix=prefix, join=join,
                         src=src, ref=ref, H=Ht)
    except Exception as exc:  # noqa: BLE001
        report["verdict"] = "registration_failed"
        report["notes"].append(f"detection/matching/refinement failed: {exc}")
        return _attach_decision(report, out_dir=out_dir, prefix=prefix)

    # ---- 3. artifacts: overlay + match figure ----
    report["artifacts"] = _write_artifacts(report, src, ref, kp1, kp2, pts1,
                                           pts2, join, out_dir, prefix)
    try:
        p1i, p2i = pts1[keep], pts2[keep]
        csv_path = os.path.join(out_dir, f"{prefix}_inliers.csv")
        with open(csv_path, "w", newline="") as fh:
            fh.write("src_x,src_y,ref_x,ref_y\n")
            for (x1, y1), (x2, y2) in zip(p1i, p2i):
                fh.write(f"{float(x1):.4f},{float(y1):.4f},"
                         f"{float(x2):.4f},{float(y2):.4f}\n")
        report["artifacts"]["correspondences"] = csv_path
        report["inlier_points_saved"] = int(len(p1i))
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"correspondence save failed: {exc}")
    return report


def _geometry_fallback(report, join, out_dir, prefix, ohrc_img, ohrc_geom,
                       nac_img, nac_geom_csv, root):
    """Register by geometry (SPICE + ISRO CSV) when content matching fails."""
    try:
        from src.preprocessing.geometry import georeference_phase5
        meta = georeference_phase5(
            join(ohrc_img), join(ohrc_geom), join(nac_img), join(nac_geom_csv),
            os.path.join(out_dir, "phase5"),
        )
        report["georef_meta"] = meta
        report["method"] = "geometry"
        report["verdict"] = "geometry_registered"
        report["staged_gsd_m"] = meta.get("gsd_m")
        report["notes"].append("geometry registration successful; "
                               "feature correspondence NOT verifiable")
        report["artifacts"] = {
            "src": meta.get("src_png"), "ref": meta.get("ref_png"),
            "overlay": meta.get("overlay_png"),
        }
        # attach honest content-evidence even on the geometry route: overlap NCC
        # and per-crop contrast let the decision layer tell an illumination gap
        # apart from a genuine geodetic frame disagreement (never leave a blank).
        sp, rp = meta.get("src_png"), meta.get("ref_png")
        if sp and rp and os.path.exists(join(sp)) and os.path.exists(join(rp)):
            try:
                from src.preprocessing.ch2_staging import (enhance_dark,
                                                           low_contrast_score)
                sA = cv2.imread(join(sp), cv2.IMREAD_GRAYSCALE)
                rA = cv2.imread(join(rp), cv2.IMREAD_GRAYSCALE)
                if sA is not None and rA is not None and sA.shape == rA.shape:
                    se, re_ = enhance_dark(sA), enhance_dark(rA)
                    union = (se > 0) & (re_ > 0)
                    ncc = None
                    if union.sum() > 100:
                        ncc = float(np.corrcoef(
                            se[union].astype(np.float64),
                            re_[union].astype(np.float64))[0, 1])
                    report["diagnostics"] = {
                        "src_low_contrast": round(low_contrast_score(sA), 3),
                        "ref_low_contrast": round(low_contrast_score(rA), 3),
                        "src_mean": float(sA.mean()),
                        "ref_mean": float(rA.mean()),
                        "overlap_ncc": ncc,
                        "overlap_px": int(union.sum()),
                    }
            except Exception as exc:  # noqa: BLE001
                report.setdefault("notes", []).append(
                    f"geometry-route evidence unavailable: {exc}")
    except Exception as exc:  # noqa: BLE001
        report["verdict"] = "registration_failed"
        report["notes"].append(f"geometry fallback also failed: {exc}")
    return _attach_decision(report, out_dir=out_dir, prefix=prefix)


def _write_artifacts(report, src, ref, kp1, kp2, pts1, pts2, join, out_dir,
                     prefix):
    """Write match overlay + checkerboard; return artifact path dict."""
    out = {}
    try:
        from src.evaluation.visualize import draw_matches

        m_fig = os.path.join(out_dir, f"{prefix}_matches.png")
        draw_matches(src, kp1, ref, kp2,
                     [cv2.DMatch(i, i, 0.0) for i in range(len(pts1))],
                     join(m_fig), inlier_mask=np.ones(len(pts1), bool))
        out["matches"] = os.path.join(out_dir, f"{prefix}_matches.png")
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"match figure failed: {exc}")

    # checkerboard of the staged aligned crops
    try:
        chk = np.zeros_like(src)
        tiles = 8
        th, tw = src.shape[0] // tiles, src.shape[1] // tiles
        for i in range(tiles):
            for j in range(tiles):
                chk[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = (
                    ref[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
                    if (i + j) % 2 == 0
                    else src[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
        c_path = os.path.join(out_dir, f"{prefix}_checkerboard.png")
        cv2.imwrite(join(c_path), chk)
        out["checkerboard"] = os.path.join(out_dir, f"{prefix}_checkerboard.png")
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"checkerboard figure failed: {exc}")
    return out


# --------------------------------------------------------------------------- #
# report writer
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Phase 9 — multi-sensor dispatch (OHRC/TMC/IIRS) for PS 26166 coverage
# --------------------------------------------------------------------------- #
SUPPORTED_SENSOR_PAIRS = ("ohrc-nac", "tmc-ohrc", "iirs-ohrc",
                          "tmc-nac", "iirs-nac", "ohrc-ohrc", "tmc-tmc",
                          "iirs-iirs", "ohrc-tmc")


def _sensor_of(path):
    """Guess the sensor of a file from its name (ps26166 signals)."""
    name = os.path.basename(str(path))
    low = name.lower()
    if re.search(r"m\d{10}(rc|lc|le|me)\.img$", low):
        return "nac"
    if "ch2_ohr" in low:
        return "ohrc"
    if "ch2_tmc" in low:
        return "tmc"
    if "ch2_iir" in low:
        return "iirs"
    if low.endswith(".img") or low.endswith(".tif") or low.endswith(".png") \
            or low.endswith(".qub"):
        for tok in ("ohr",):
            if tok in low:
                return "ohrc"
    return "unknown"


def detect_pair(src_img, ref_img):
    """Auto-detect the (canonical) sensor pair for two arbitrary files."""
    s, r = _sensor_of(src_img), _sensor_of(ref_img)
    if s == "unknown" or r == "unknown":
        raise ValueError(
            f"cannot auto-detect sensor pair from "
            f"{os.path.basename(str(src_img))!r} / "
            f"{os.path.basename(str(ref_img))!r}; use --sensor explicitly")
    pairs = {
        ("ohrc", "nac"): "ohrc-nac", ("nac", "ohrc"): "ohrc-nac",
        ("tmc", "ohrc"): "tmc-ohrc", ("ohrc", "tmc"): "ohrc-tmc",
        ("iirs", "ohrc"): "iirs-ohrc", ("ohrc", "iirs"): "iirs-ohrc",
        ("tmc", "nac"): "tmc-nac", ("nac", "tmc"): "tmc-nac",
        ("iirs", "nac"): "iirs-nac", ("nac", "iirs"): "iirs-nac",
        ("ohrc", "ohrc"): "ohrc-ohrc", ("tmc", "tmc"): "tmc-tmc",
        ("iirs", "iirs"): "iirs-iirs",
    }
    pair = pairs.get((s, r))
    if pair is None:
        raise ValueError(f"no supported register for {s} <-> {r}")
    return pair


def run_sensor_auto(sensor_pair, src_img, src_geom, ref_img, ref_geom, *,
                    out_dir="data/processed/auto", prefix="sensor", root=PROJECT_ROOT,
                    crop_rows=1024, swath_pix=None, bands=None, n_bands=4,
                    min_inliers=12, min_ratio=0.02, verbose=True):
    """Run automatic registration for any supported Chandrayaan-2 sensor pair.

    This extends the Phase-8 auto entry point to the remaining PS 26166 sensors
    (TMC and IIRS) without touching the working OHRC<->NAC path:

      * ``ohrc-nac`` -> delegates to ``run_auto`` (Phase 8 champion path).
      * ``tmc-ohrc`` -> ``register_ch2_pair`` (CH2 vs CH2, content -> geometry).
      * ``iirs-ohrc`` -> ``register_iirs_to_ohrc`` (IIRS ENVI hyperspectral band
        selection + cross-modal matching vs OHRC).
      * ``tmc-nac``  -> ``register_ch2_to_nac`` (TMC source vs **LRO NAC lunar
        reference** via ISRO CSV + NAC SPICE ground grid).
      * ``iirs-nac`` -> ``register_iirs_to_nac`` (IIRS vs **LRO NAC** content
        attempt only — IIRS has no geometry CSV).

    Returns a single JSON report with an honest verdict; never raises for an
    un-registerable pair. Content matching is proof-based: if no reliable model
    is found the verdict is ``not_registered`` (TMC/IIRS get that from their own
    registers), never a fabricated homography.
    """
    join = lambda p: p if os.path.isabs(p) else os.path.join(root, p)  # noqa: E731
    os.makedirs(join(out_dir), exist_ok=True)

    if sensor_pair not in SUPPORTED_SENSOR_PAIRS:
        rep = {"sensor_pair": sensor_pair, "verdict": "input_error",
               "notes": [f"unsupported sensor pair {sensor_pair!r}; "
                         f"expected one of {SUPPORTED_SENSOR_PAIRS}"]}
        return _attach_decision(rep, out_dir=join(out_dir), prefix=prefix)

    if sensor_pair == "ohrc-nac":
        report = run_auto(src_img, src_geom, ref_img, out_dir=out_dir,
                          prefix=prefix, nac_geom_csv=ref_geom, root=root,
                          verbose=verbose)
        report.setdefault("sensor_pair", sensor_pair)
        _attach_decision(report, out_dir=join(out_dir), prefix=prefix)
        return report

    try:
        if sensor_pair in ("tmc-ohrc", "tmc-nac", "ohrc-tmc",
                           "tmc-tmc", "ohrc-ohrc"):
            from src.preprocessing.ch2_staging import (
                register_ch2_pair, register_ch2_to_nac)
            if sensor_pair == "tmc-nac":
                report = register_ch2_to_nac(
                    join(src_img), join(src_geom), join(ref_img), join(ref_geom),
                    out_dir=join(out_dir), prefix=prefix, n_along=crop_rows,
                )
            else:
                report = register_ch2_pair(
                    join(src_img), join(src_geom), join(ref_img), join(ref_geom),
                    out_dir=join(out_dir), prefix=prefix, n_along=crop_rows,
                )
        else:  # iirs-ohrc / iirs-nac / iirs-iirs
            from src.detection.iirs import (
                register_iirs_to_ohrc, register_iirs_to_nac)
            if sensor_pair == "iirs-nac":
                report = register_iirs_to_nac(
                    join(src_img), join(src_geom), join(ref_img),
                    out_dir=join(out_dir), prefix=prefix, bands=bands,
                    n_bands=n_bands, min_inliers=min_inliers, min_ratio=min_ratio,
                )
            elif sensor_pair == "iirs-iirs":
                report = register_iirs_to_ohrc(
                    join(src_img), join(src_geom), join(ref_img), join(ref_geom),
                    out_dir=join(out_dir), prefix=prefix, bands=bands,
                    n_bands=n_bands, min_inliers=min_inliers, min_ratio=min_ratio,
                )
            else:
                report = register_iirs_to_ohrc(
                    join(src_img), join(src_geom), join(ref_img), join(ref_geom),
                    out_dir=join(out_dir), prefix=prefix, bands=bands,
                    n_bands=n_bands, min_inliers=min_inliers, min_ratio=min_ratio,
                )
    except Exception as exc:  # noqa: BLE001
        report = {"sensor_pair": sensor_pair, "verdict": "registration_failed",
                  "notes": [f"{sensor_pair} registration failed: {exc}"]}
    report.setdefault("sensor_pair", sensor_pair)
    _attach_decision(report, out_dir=join(out_dir), prefix=prefix)
    if verbose:
        print(f"[run_sensor_auto] {sensor_pair}: verdict={report.get('verdict')}"
              + (f", inliers={report.get('inliers')}" if report.get('inliers') is not None else ""))
    return report


def write_json_report(report, path, root=PROJECT_ROOT):
    p = path if os.path.isabs(path) else os.path.join(root, path)
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return p


def load_report(path, root=PROJECT_ROOT):
    p = path if os.path.isabs(path) else os.path.join(root, path)
    with open(p) as fh:
        return json.load(fh)


def summarize(report):
    """Human-readable single-line summary of a report."""
    d = report.get("decision") or {}
    if d.get("matched"):
        cause = d.get("cause", "")
        return (f"MATCH ({cause}): {d.get('explanation')} — "
                f"best product: {d.get('best_product')}")
    if d.get("cause"):
        return (f"{d.get('explanation')} — best product: "
                f"{d.get('best_product')}")
    v = report.get("verdict")
    if v == "registered":
        return ("REGISTERED (content): RMSE={rmse_px} px, self-RMSE="
                "{rmse_self_px} px, {inliers} inliers / {inlier_ratio} ratio, "
                "{n_matches} matches").format(**report)
    if v == "geometry_registered":
        gsd = report.get("staged_gsd_m") or report.get("gsd_m")
        method = report.get("method")
        return (f"GEOMETRY REGISTERED (no content): GSD={gsd} m, "
                f"method={method}")
    if v == "no_content_correspondence":
        return "NO CONTENT CORRESPONDENCE — trying geometry registration"
    return f"REGISTRATION NOT COMPLETED — verdict={v} ({len(report.get('notes', []))} notes)"


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="auto registration (single pair)")
    ap.add_argument("--ohrc_img", required=True)
    ap.add_argument("--ohrc_geom", required=True)
    ap.add_argument("--nac_img", required=True)
    ap.add_argument("--nac_geom", default="", help="NAC SPICE geometry CSV (for equal-GSD seed / geometry fallback)")
    ap.add_argument("--ground_truth", default="data/ground_truth/pair1_gt_v2.csv")
    ap.add_argument("--out_dir", default="data/processed/auto")
    ap.add_argument("--prefix", default="auto")
    ap.add_argument("--no-nac-geom", action="store_true")
    args = ap.parse_args()

    rep = run_auto(
        args.ohrc_img, args.ohrc_geom, args.nac_img,
        out_dir=args.out_dir, prefix=args.prefix,
        nac_geom_csv="" if args.no_nac_geom else args.nac_geom,
        ground_truth=args.ground_truth,
    )
    print(summarize(rep))
    write_json_report(rep, os.path.join(args.out_dir, "report.json"))
    print("report ->", os.path.join(args.out_dir, "report.json"))
