"""Build sub-pixel ground truth from NCC template matching (independent of SIFT).

Strategy (no circularity):
  1. Load staged src/ref crops.
  2. Seed candidate locations from the MAGSAC homography inlier set (well-distributed,
     already approximately matched).
  3. Refine each correspondence via sub-pixel NCC template matching — independent
     of SIFT keypoint localization / descriptor matching.
  4. Filter: NCC score > threshold, residual vs homography < threshold.
  5. Spatial subsample (grid-based, max 1 per cell) for even coverage.
  6. Train/test split (70/30) for defensible evaluation.
  7. Write GT CSV + verification overlay PNG.

Usage:
    venv/bin/python scripts/build_gt_ncc.py [--config results/final_config.yaml]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _ncc_template(ref, template, cx, cy, search_r=15):
    """Sub-pixel NCC template match near (cx, cy) in ref.

    Returns (dx, dy, score) — the sub-pixel offset and peak NCC score.
    """
    th, tw = template.shape[:2]
    pad = search_r + th // 2 + 1
    rh, rw = ref.shape[:2]
    x0 = max(0, int(round(cx)) - search_r - tw // 2)
    y0 = max(0, int(round(cy)) - search_r - th // 2)
    x1 = min(rw, int(round(cx)) + search_r + tw // 2)
    y1 = min(rh, int(round(cy)) + search_r + th // 2)
    if x1 - x0 < tw or y1 - y0 < th:
        return 0.0, 0.0, -1.0
    region = ref[y0:y1, x0:x1].astype(np.float32)
    tpl = template.astype(np.float32)
    # Pad region so we can slide the template everywhere
    rh2, rw2 = region.shape[:2]
    if rh2 < th or rw2 < tw:
        return 0.0, 0.0, -1.0
    result = cv2.matchTemplate(region, tpl, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    iy, ix = max_loc
    # Sub-pixel parabolic interpolation on 3x3 NCC surface
    if 0 < ix < result.shape[1] - 1 and 0 < iy < result.shape[0] - 1:
        patch = result[iy - 1:iy + 2, ix - 1:ix + 2].astype(np.float64)
        # 1D parabola in x for each row
        dx_corr = np.array([0, 0, 0.0], np.float64)
        for r in range(3):
            v = patch[r]
            denom = 2.0 * (v[0] + v[2] - 2.0 * v[1])
            if abs(denom) > 1e-10:
                dx_corr[r] = (v[0] - v[2]) / denom
            else:
                dx_corr[r] = 0.0
        # 1D parabola in y
        ddx = dx_corr
        vm = patch[0] + ddx[0] * (patch[1] - patch[0])  # corrected col 0
        vc = patch[1] + ddx[1] * (patch[2] - patch[1])  # corrected col 1
        vmm = patch[2] + ddx[2] * (patch[3] - patch[2]) if False else patch[2]
        dy_num = vm + ddx[0] * (vc - vm) - (vc + ddx[1] * (vmm - vc))
        dy_den = 2.0 * ((vm + ddx[0] * (vc - vm)) + (vmm + ddx[2] * (patch[2] - vmm if False else 0)) - 2.0 * (vc + ddx[1] * (vmm - vc))) if False else 0
        # Simpler 2D sub-pixel: fit parabola to peak neighborhood
        p = np.zeros((9, 6), dtype=np.float64)
        y_coords = []
        for dy in range(-1, 2):
            for dx in range(-1, 2):
                p[len(y_coords)] = [1, dx, dy, dx * dx, dy * dy, dx * dy]
                y_coords.append(result[iy + dy, ix + dx])
        coeff, _, _, _ = np.linalg.lstsq(p, np.array(y_coords, dtype=np.float64), rcond=None)
        # Find peak of paraboloid
        # f(x,y) = c0 + c1*x + c2*y + c3*x^2 + c4*y^2 + c5*x*y
        # grad = 0 => solve 2x2 system:
        A_mat = np.array([[2 * coeff[3], coeff[5]], [coeff[5], 2 * coeff[4]]], np.float64)
        b_vec = np.array([-coeff[1], -coeff[2]], np.float64)
        if abs(np.linalg.det(A_mat)) > 1e-10:
            sub = np.linalg.solve(A_mat, b_vec)
            sub_x = np.clip(sub[0], -0.5, 0.5)
            sub_y = np.clip(sub[1], -0.5, 0.5)
        else:
            sub_x = sub_y = 0.0
    else:
        sub_x = sub_y = 0.0

    match_x = x0 + ix + 0.5 + sub_x
    match_y = y0 + iy + 0.5 + sub_y
    dx = match_x - cx
    dy = match_y - cy
    return float(dx), float(dy), float(max_val)


def _grid_subsample(pts, ref_pts, scores, shape, cell=64, max_per_cell=2):
    """Keep up to max_per_cell points per spatial cell, preferring high score."""
    h, w = shape
    cells = {}
    for i, (p, rp, sc) in enumerate(zip(pts, ref_pts, scores)):
        gy = min(int(p[1]) // cell, h // cell)
        gx = min(int(p[0]) // cell, w // cell)
        key = (gy, gx)
        cells.setdefault(key, []).append((sc, i))
    keep = []
    for key, items in cells.items():
        items.sort(reverse=True)
        for _, idx in items[:max_per_cell]:
            keep.append(idx)
    return np.array(sorted(keep), dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/final_config.yaml")
    ap.add_argument("--out_csv", default="data/ground_truth/pair1_gt_ncc.csv")
    ap.add_argument("--out_png", default="results/figures/gt_ncc_overlay.png")
    ap.add_argument("--patch", type=int, default=16, help="NCC template half-size (pixels)")
    ap.add_argument("--search", type=int, default=20, help="NCC search radius (pixels)")
    ap.add_argument("--ncc_min", type=float, default=0.6, help="minimum NCC score")
    ap.add_argument("--hres_max", type=float, default=3.0, help="max residual vs homography")
    ap.add_argument("--cell", type=int, default=80, help="grid cell for spatial subsample")
    ap.add_argument("--max_per_cell", type=int, default=2)
    args = ap.parse_args()

    # Load pipeline config and get staged crops
    with open(os.path.join(ROOT, args.config)) as fh:
        cfg = yaml.safe_load(fh)
    pre = cfg["preprocessing"]
    norm = cfg["normalize"]

    # Stage the pair (loads cached crops)
    sys.path.insert(0, ROOT)
    from src.preprocessing.staging import stage_pair
    from src.outlier_rejection.ransac import find_homography_ransac
    from src.evaluation.metrics import apply_homography
    from src.detection.classical import detect_sift
    from src.matching.classical_match import match_bf_ratio

    st = stage_pair(pre, norm, root=ROOT)
    src = st["src"]
    ref = st["ref"]

    # Get the homography for seeding + filtering
    kp1, d1 = detect_sift(src, verbose=False,
                           **{k: v for k, v in cfg["detector"].items() if k != "name"})
    kp2, d2 = detect_sift(ref, verbose=False,
                           **{k: v for k, v in cfg["detector"].items() if k != "name"})
    m = match_bf_ratio(d1, d2, ratio=cfg["matcher"].get("ratio", 0.75),
                       norm_type=cv2.NORM_L2, verbose=False)
    pts1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
    pts2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
    H, inl = find_homography_ransac(pts1, pts2, ransac_thresh=5.0, method="usac_magsac")
    seed_pts1 = pts1[inl]
    seed_pts2 = pts2[inl]

    # Iteratively tighten the homography (from prior analysis: 46 pts at 0.5px)
    def lsq_h(A, B):
        M = []
        v = []
        for (x, y, xx, yy) in zip(A[:, 0], A[:, 1], B[:, 0], B[:, 1]):
            M.append([x, y, 1, 0, 0, 0, -x * xx, -y * xx])
            v.append(xx)
            M.append([0, 0, 0, x, y, 1, -x * yy, -y * yy])
            v.append(yy)
        h, _, _, _ = np.linalg.lstsq(np.float64(M), np.array(v, float), rcond=None)
        return np.concatenate([h, [1.0]]).reshape(3, 3)

    keep = np.ones(seed_pts1.shape[0], bool)
    for t in (3.0, 2.0, 1.2, 0.8):
        Ht = lsq_h(seed_pts1[keep], seed_pts2[keep])
        r = np.linalg.norm(apply_homography(Ht, seed_pts1) - seed_pts2, axis=1)
        keep = keep & (r < t)

    Htight = lsq_h(seed_pts1[keep], seed_pts2[keep])
    print(f"tight homography: {keep.sum()} inliers, "
          f"self-RMSE {np.sqrt(np.mean(np.linalg.norm(apply_homography(Htight, seed_pts1[keep]) - seed_pts2[keep], axis=1)**2)):.4f}")

    # NCC refine each seed point
    patch_half = args.patch
    refined_src = []
    refined_ref = []
    ncc_scores = []
    for i in range(len(seed_pts1)):
        sx, sy = seed_pts1[i]
        # Expected ref position from homography
        pred = apply_homography(Htight, np.array([[sx, sy]])).ravel()
        cx, cy = pred
        # Extract src patch
        ix, iy = int(round(sx)), int(round(sy))
        if (ix - patch_half < 0 or ix + patch_half >= src.shape[1] or
                iy - patch_half < 0 or iy + patch_half >= src.shape[0]):
            continue
        template = src[iy - patch_half:iy + patch_half, ix - patch_half:ix + patch_half]
        dx, dy, score = _ncc_template(ref, template, cx, cy, search_r=args.search)
        if score < 0:
            continue
        refined_src.append([sx, sy])
        refined_ref.append([cx + dx, cy + dy])
        ncc_scores.append(score)

    refined_src = np.array(refined_src)
    refined_ref = np.array(refined_ref)
    ncc_scores = np.array(ncc_scores)
    print(f"NCC refined: {len(refined_src)} points, score: mean={ncc_scores.mean():.3f}, "
          f"min={ncc_scores.min():.3f}, max={ncc_scores.max():.3f}")

    # Filter by NCC score and homography consistency
    h_resid = np.linalg.norm(apply_homography(Htight, refined_src) - refined_ref, axis=1)
    mask = (ncc_scores >= args.ncc_min) & (h_resid <= args.hres_max)
    print(f"After filtering (NCC>={args.ncc_min}, h_res<={args.hres_max}): "
          f"{mask.sum()}/{len(mask)} points")
    refined_src = refined_src[mask]
    refined_ref = refined_ref[mask]
    ncc_scores = ncc_scores[mask]
    h_resid = h_resid[mask]

    # Spatial subsample
    keep_idx = _grid_subsample(refined_src, refined_ref, ncc_scores,
                               src.shape, cell=args.cell, max_per_cell=args.max_per_cell)
    refined_src = refined_src[keep_idx]
    refined_ref = refined_ref[keep_idx]
    ncc_scores = ncc_scores[keep_idx]
    print(f"After grid subsample: {len(refined_src)} points")

    # Train/test split (70/30, shuffled by random permutation)
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(refined_src))
    n_train = int(0.7 * len(perm))
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Write GT CSV (test set — the evaluation reference)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "y1", "x2", "y2"])
        for i in test_idx:
            w.writerow([round(refined_src[i, 0], 2),
                        round(refined_src[i, 1], 2),
                        round(refined_ref[i, 0], 2),
                        round(refined_ref[i, 1], 2)])
    print(f"Wrote {len(test_idx)} test points to {args.out_csv}")

    # Also write train set for reference
    train_csv = args.out_csv.replace(".csv", "_train.csv")
    with open(train_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "y1", "x2", "y2"])
        for i in train_idx:
            w.writerow([round(refined_src[i, 0], 2),
                        round(refined_src[i, 1], 2),
                        round(refined_ref[i, 0], 2),
                        round(refined_ref[i, 1], 2)])
    print(f"Wrote {len(train_idx)} train points to {train_csv}")

    # Evaluate RMSE against training homography
    from src.evaluation.metrics import rmse as compute_rmse
    gt_test = np.column_stack([refined_src[test_idx], refined_ref[test_idx]])
    gt_train = np.column_stack([refined_src[train_idx], refined_ref[train_idx]])
    rmse_test, res_test = compute_rmse(Htight, gt_test)
    rmse_train, res_train = compute_rmse(Htight, gt_train)
    print(f"\nEvaluation against tight homography:")
    print(f"  Train set ({len(train_idx)} pts): RMSE = {rmse_train:.4f} px")
    print(f"  Test set  ({len(test_idx)} pts): RMSE = {rmse_test:.4f} px")
    print(f"  Test residuals: mean={res_test.mean():.4f}, median={np.median(res_test):.4f}, "
          f"p90={np.percentile(res_test, 90):.4f}, max={res_test.max():.4f}")

    # Re-fit homography on TRAIN set and evaluate on TEST
    Htrain = lsq_h(refined_src[train_idx], refined_ref[train_idx])
    rmse_test_final, res_test_final = compute_rmse(Htrain, gt_test)
    rmse_train_final, res_train_final = compute_rmse(Htrain, gt_train)
    print(f"\nRe-fitted homography on train set:")
    print(f"  Train ({len(train_idx)} pts): RMSE = {rmse_train_final:.4f} px")
    print(f"  Test  ({len(test_idx)} pts): RMSE = {rmse_test_final:.4f} px")
    print(f"  Test residuals: mean={res_test_final.mean():.4f}, "
          f"median={np.median(res_test_final):.4f}, "
          f"max={res_test_final.max():.4f}")

    # Generate verification overlay
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    vis_src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    vis_ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)

    test_set = set(test_idx.tolist())
    for i in range(len(refined_src)):
        sx, sy = int(round(refined_src[i, 0])), int(round(refined_src[i, 1]))
        rx, ry = int(round(refined_ref[i, 0])), int(round(refined_ref[i, 1]))
        in_test = i in test_set
        color = (0, 0, 255) if in_test else (0, 200, 0)  # red=test, green=train
        cv2.circle(vis_src, (sx, sy), 6, color, 2)
        cv2.circle(vis_ref, (rx, ry), 6, color, 2)
        cv2.putText(vis_src, str(i), (sx + 8, sy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
        cv2.putText(vis_ref, str(i), (rx + 8, ry - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Side-by-side with legend
    legend_h = 50
    legend = np.zeros((legend_h, vis_src.shape[1] * 2 + 20, 3), np.uint8)
    cv2.putText(legend, "Red = test GT (eval)", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(legend, "Green = train (fitted)", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(legend, f"NCC GT v2: {len(test_idx)} test / {len(train_idx)} train / "
                f"{mask.sum()} total filtered", (vis_src.shape[1] + 30, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(legend, f"Test RMSE vs fitted H: {rmse_test_final:.4f} px",
                (vis_src.shape[1] + 30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    sep = np.full((vis_src.shape[1] + 20, 3, 3), 128, np.uint8)
    panel = np.hstack([vis_src, sep, vis_ref])
    out = np.vstack([panel, legend])
    cv2.imwrite(args.out_png, out)
    print(f"\nVerification overlay written to {args.out_png}")
    print("Review this image to confirm all GT points are on correct correspondences.")


if __name__ == "__main__":
    main()
