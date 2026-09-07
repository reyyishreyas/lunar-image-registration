"""Phase 1 end-to-end baseline pipeline (AI_EXECUTION_PLAN.md Step 1.8).

Orchestrates preprocessing -> detection -> matching -> outlier rejection ->
evaluation for a registered pair, driven by a YAML experiment config.

Config C1 (configs/experiment_C1.yaml) is the single baseline entry point;
later phases extend this same pipeline rather than creating competing ones.
"""

from __future__ import annotations

import csv
import os
import time

import cv2
import numpy as np
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _abs(jpath):
    return os.path.join(PROJECT_ROOT, jpath)


def load_pair(src_path, ref_path):
    src = cv2.imread(_abs(src_path), cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(_abs(ref_path), cv2.IMREAD_GRAYSCALE)
    if src is None or ref is None:
        raise FileNotFoundError(f"could not read pair inputs {src_path}, {ref_path}")
    return src, ref


def maybe_georeference(cfg):
    pre = cfg.get("preprocessing", {})
    if not pre.get("enabled", True):
        return
    out = pre["out_dir"]
    src, ref = pre["outputs"]["src"], pre["outputs"]["ref"]
    if os.path.exists(_abs(src)) and os.path.exists(_abs(ref)):
        return
    from src.preprocessing.georeference import georeference_pair

    meta = georeference_pair(
        ohrc_img_path=_abs(pre["ohrc_img"]),
        ohrc_csv_path=_abs(pre["ohrc_geometry"]),
        nac_img_path=_abs(pre["nac_img"]),
        out_dir=_abs(out),
        crop_px=pre.get("crop_px", 1024),
    )
    return meta


def apply_normalize(src, ref, ncfg):
    """Apply the config's normalization stage to the pair before detection.

    Args:
        src, ref: uint8 grayscale crops.
        ncfg: config dict under top-level key "normalize" (empty if unset).

    Returns:
        (normalized_src, normalized_ref, label): label is a short string for the
        ablation row's preproc column.
    """
    if not ncfg.get("enabled", False):
        return src, ref, "raw"
    method = ncfg.get("method", "")
    if method == "histogram_match":
        from src.preprocessing.normalize import histogram_match
        return histogram_match(src, ref), ref, "histmatch"
    if method == "clahe":
        from src.preprocessing.normalize import apply_clahe
        clip = float(ncfg.get("clip_limit", 2.0))
        tile = int(ncfg.get("tile_grid", 8))
        return (apply_clahe(src, clip_limit=clip, tile_grid=tile),
                apply_clahe(ref, clip_limit=clip, tile_grid=tile),
                f"clahe:{clip}")
    if method == "gamma_shadow":
        from src.preprocessing.shadow_correct import gamma_shadow_correct
        gamma = float(ncfg.get("gamma", 0.5))
        return (gamma_shadow_correct(src, gamma=gamma),
                gamma_shadow_correct(ref, gamma=gamma),
                f"gamma_shadow:{gamma}")
    raise ValueError(f"unknown normalize method {method!r}")


def run_experiment(config_path, verbose=True):
    with open(_abs(config_path)) as fh:
        cfg = yaml.safe_load(fh)
    cfg["config_path"] = config_path

    t_start = time.time()
    geo_meta = maybe_georeference(cfg)
    t_geo = time.time() - t_start

    fmt = cfg["inputs"]
    src, ref = load_pair(fmt["src"], fmt["ref"])

    src, ref, norm_label = apply_normalize(src, ref, cfg.get("normalize", {}))

    det_cfg = cfg["detector"]
    if det_cfg.get("name") == "sift":
        from src.detection.classical import detect_sift

        kp1, d1 = detect_sift(src, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                       if k != "name"})
        kp2, d2 = detect_sift(ref, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                       if k != "name"})
    elif det_cfg.get("name") == "akaze":
        from src.detection.classical import detect_akaze

        kp1, d1 = detect_akaze(src, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                        if k != "name"})
        kp2, d2 = detect_akaze(ref, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                        if k != "name"})
    elif det_cfg.get("name") == "rift2":
        from src.detection.rift2 import detect_rift2

        kp1, d1 = detect_rift2(src, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                        if k != "name"})
        kp2, d2 = detect_rift2(ref, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                        if k != "name"})
    elif det_cfg.get("name") == "superpoint":
        from src.detection.learned import detect_superpoint

        kp1, d1 = detect_superpoint(src, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                             if k != "name"})
        kp2, d2 = detect_superpoint(ref, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                             if k != "name"})
    else:
        raise ValueError(f"unknown detector {det_cfg.get('name')}")

    m_cfg = cfg["matcher"]
    if m_cfg.get("name") == "bf_ratio":
        from src.matching.classical_match import match_bf_ratio

        norm = {"l2": cv2.NORM_L2, "hamming": cv2.NORM_HAMMING}.get(
            m_cfg.get("norm", "l2"), cv2.NORM_L2)
        matches = match_bf_ratio(d1, d2, ratio=m_cfg.get("ratio", 0.75),
                                 norm_type=norm, verbose=verbose)
    elif m_cfg.get("name") == "rift2":
        from src.detection.rift2 import match_rift2_nn

        matches = match_rift2_nn(
            kp1, d1, kp2, d2,
            lowes_ratio=m_cfg.get("lowes_ratio", 0.95),
            mutual=m_cfg.get("mutual", False), verbose=verbose)
    elif m_cfg.get("name") == "superglue":
        from src.matching.learned_match import match_superglue

        matches = match_superglue(
            kp1, d1, kp2, d2,
            weights=m_cfg.get("weights", "outdoor"),
            match_threshold=m_cfg.get("match_threshold", 0.2),
            sinkhorn_iterations=m_cfg.get("sinkhorn_iterations", 20),
            verbose=verbose)
    elif m_cfg.get("name") == "loftr":
        from src.matching.learned_match import match_loftr

        matches = match_loftr(
            kp1, d1, kp2, d2,
            weights=m_cfg.get("weights", "outdoor"), verbose=verbose)
    else:
        raise ValueError(f"unknown matcher {m_cfg.get('name')}")

    or_cfg = cfg["outlier_rejection"]
    if or_cfg.get("name") == "ransac":
        from src.outlier_rejection.ransac import find_homography_ransac

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        H, inlier_mask = find_homography_ransac(pts1, pts2,
                                                ransac_thresh=or_cfg.get("ransac_thresh", 5.0))
        inliers = int(inlier_mask.sum())
        ratio = inliers / len(matches) if len(matches) else 0.0
    else:
        raise ValueError(f"unknown outlier rejection {or_cfg.get('name')}")

    out_cfg = cfg["outputs"]
    from src.evaluation.visualize import draw_matches

    draw_matches(src, kp1, ref, kp2, matches, _abs(out_cfg["matches_figure"]),
                 inlier_mask=inlier_mask)

    gt_path = _abs(cfg.get("ground_truth", ""))
    rmse_px = ""
    if os.path.exists(gt_path):
        from src.evaluation.metrics import rmse

        gt = np.loadtxt(gt_path, delimiter=",", skiprows=1,
                        usecols=(0, 1, 2, 3)).reshape(-1, 4)
        rmse_px, residuals = rmse(H, gt)
        rmse_px = round(rmse_px, 4)

    t_total = time.time() - t_start
    preproc = f"georef+{norm_label}" if norm_label != "raw" else "georef"
    _matcher_thresh = (m_cfg.get("ratio", m_cfg.get("lowes_ratio",
                                                    m_cfg.get("match_threshold", 0.75))))
    row = {
        "config_id": cfg.get("experiment", {}).get("id", "C1"),
        "preproc": f"{preproc} (crop {src.shape[1]}x{src.shape[0]})",
        "detector": det_cfg.get("name", ""),
        "matcher": f"{m_cfg.get('name')}:{_matcher_thresh}",
        "outlier": f"{or_cfg.get('name')}:{or_cfg.get('ransac_thresh', 5.0)}",
        "refinement": "",
        "rmse_px": rmse_px,
        "inliers": inliers,
        "inlier_ratio": round(ratio, 4),
        "n_matches": len(matches),
        "time_s": round(t_total, 2),
        "georef_s": round(t_geo, 2),
    }

    def _append_ablation(row_cfg, cfg_for_path):
        log_path = _abs(cfg_for_path["outputs"]["ablation_log"])
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        cols = ["config_id", "preproc", "detector", "matcher", "outlier",
                "refinement", "rmse_px", "inliers", "inlier_ratio", "n_matches",
                "time_s", "georef_s"]
        exists = os.path.exists(log_path)
        with open(log_path, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            if not exists:
                w.writeheader()
            w.writerow(row_cfg)

    _append_ablation(row, cfg)

    if verbose:
        print(f"\n===== {row['config_id']} results =====")
        for k, v in row.items():
            print(f"  {k}: {v}")
        print("matches fig:", out_cfg["matches_figure"])
    return row