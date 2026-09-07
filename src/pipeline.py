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


def run_experiment(config_path, verbose=True):
    with open(_abs(config_path)) as fh:
        cfg = yaml.safe_load(fh)
    cfg["config_path"] = config_path

    t_start = time.time()
    geo_meta = maybe_georeference(cfg)
    t_geo = time.time() - t_start

    fmt = cfg["inputs"]
    src, ref = load_pair(fmt["src"], fmt["ref"])

    det_cfg = cfg["detector"]
    if det_cfg.get("name") == "sift":
        from src.detection.classical import detect_sift

        kp1, d1 = detect_sift(src, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                       if k != "name"})
        kp2, d2 = detect_sift(ref, verbose=verbose, **{k: v for k, v in det_cfg.items()
                                                       if k != "name"})
    else:
        raise ValueError(f"unknown detector {det_cfg.get('name')}")

    m_cfg = cfg["matcher"]
    if m_cfg.get("name") == "bf_ratio":
        from src.matching.classical_match import match_bf_ratio

        matches = match_bf_ratio(d1, d2, ratio=m_cfg.get("ratio", 0.75), verbose=verbose)
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
    row = {
        "config_id": cfg.get("experiment", {}).get("id", "C1"),
        "preproc": f"georef+normalize (crop {src.shape[1]}x{src.shape[0]})",
        "detector": det_cfg.get("name", ""),
        "matcher": f"{m_cfg.get('name')}:{m_cfg.get('ratio', 0.75)}",
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
        print("\n===== C1 results =====")
        for k, v in row.items():
            print(f"  {k}: {v}")
        print("matches fig:", out_cfg["matches_figure"])
    return row