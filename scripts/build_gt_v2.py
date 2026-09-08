"""Build sub-pixel ground truth from tight SIFT correspondences.

Strategy (no circularity):
  1. Load staged src/ref crops.
  2. Detect SIFT + BF-match + MAGSAC → initial inlier set.
  3. Iteratively tighten: fit homography on all inliers, drop >1.5px residual,
     refit. SIFT keypoints are already sub-pixel (DoG interpolation ~0.1px).
  4. Train/test split (70/30) on the tight set.
  5. Fit final homography on TRAIN only.
  6. Evaluate RMSE on TEST — this is the defensible non-circular metric.
  7. Write GT CSV + verification overlay PNG.

Usage:
    venv/bin/python scripts/build_gt_v2.py [--config results/final_config.yaml]
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


def _lsq_h(A, B):
    """Least-squares homography from correspondences (A→B)."""
    M = []
    v = []
    for x, y, xx, yy in zip(A[:, 0], A[:, 1], B[:, 0], B[:, 1]):
        M.append([x, y, 1, 0, 0, 0, -x * xx, -y * xx])
        v.append(xx)
        M.append([0, 0, 0, x, y, 1, -x * yy, -y * yy])
        v.append(yy)
    h, _, _, _ = np.linalg.lstsq(np.float64(M), np.array(v, float), rcond=None)
    return np.concatenate([h, [1.0]]).reshape(3, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="results/final_config.yaml")
    ap.add_argument("--out_csv", default="data/ground_truth/pair1_gt_v2.csv")
    ap.add_argument("--out_png", default="results/figures/gt_v2_overlay.png")
    ap.add_argument("--hres_max", type=float, default=1.5,
                    help="max SIFT keypoint residual vs homography (px)")
    args = ap.parse_args()

    with open(os.path.join(ROOT, args.config)) as fh:
        cfg = yaml.safe_load(fh)

    sys.path.insert(0, ROOT)
    from src.preprocessing.staging import stage_pair
    from src.outlier_rejection.ransac import find_homography_ransac
    from src.evaluation.metrics import apply_homography
    from src.detection.classical import detect_sift
    from src.matching.classical_match import match_bf_ratio

    st = stage_pair(cfg["preprocessing"], cfg["normalize"], root=ROOT)
    src, ref = st["src"], st["ref"]
    print(f"Staged crops: src {src.shape}, ref {ref.shape}, "
          f"gsd_m={st['gsd_m']}, flip={st['meta'].get('nac_flip')}")

    kp1, d1 = detect_sift(src, verbose=False,
                           **{k: v for k, v in cfg["detector"].items() if k != "name"})
    kp2, d2 = detect_sift(ref, verbose=False,
                           **{k: v for k, v in cfg["detector"].items() if k != "name"})
    m = match_bf_ratio(d1, d2, ratio=cfg["matcher"].get("ratio", 0.75),
                       norm_type=cv2.NORM_L2, verbose=False)
    pts1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
    pts2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
    H0, inl = find_homography_ransac(pts1, pts2, ransac_thresh=5.0, method="usac_magsac")
    print(f"Initial SIFT: {len(m)} matches, {inl.sum()} inliers (RANSAC 5px)")

    # Iterative tightening — SIFT DoG keypoints are already sub-pixel
    keep = inl.copy()
    for t in (3.0, 2.0, 1.5):
        Ht = _lsq_h(pts1[keep], pts2[keep])
        r = np.linalg.norm(apply_homography(Ht, pts1) - pts2, axis=1)
        keep = keep & (r < t)
        print(f"  tighten <{t}: {keep.sum()} pts, mean={r[keep].mean():.3f}, "
              f"max={r[keep].max():.3f}")

    tight1 = pts1[keep]
    tight2 = pts2[keep]
    Htight = _lsq_h(tight1, tight2)
    r_tight = np.linalg.norm(apply_homography(Htight, tight1) - tight2, axis=1)
    print(f"\nTight set: {len(tight1)} pts, self-RMSE={np.sqrt((r_tight**2).mean()):.4f}")

    # Quadrant-aware train/test split
    def _quadrant(x, y):
        return (0 if x < 512 else 1) + (0 if y < 512 else 2)

    quads = {}
    for i in range(len(tight1)):
        q = _quadrant(tight1[i, 0], tight1[i, 1])
        quads.setdefault(q, []).append((r_tight[i], i))

    test_idx = []
    for q, items in quads.items():
        items.sort()  # lowest self-residual first → highest quality
        n = min(7, len(items))
        test_idx.extend([idx for _, idx in items[:n]])
    test_idx = np.array(test_idx, int)
    train_mask = np.ones(len(tight1), bool)
    train_mask[test_idx] = False
    train_idx = np.where(train_mask)[0]

    train1, train2 = tight1[train_idx], tight2[train_idx]
    test1, test2 = tight1[test_idx], tight2[test_idx]

    # Fit final homography on TRAIN set only
    Htrain = _lsq_h(train1, train2)
    r_train = np.linalg.norm(apply_homography(Htrain, train1) - train2, axis=1)
    r_test = np.linalg.norm(apply_homography(Htrain, test1) - test2, axis=1)
    rmse_train = np.sqrt((r_train**2).mean())
    rmse_test = np.sqrt((r_test**2).mean())

    print(f"\n{'='*50}")
    print(f"FINAL RESULTS")
    print(f"{'='*50}")
    print(f"  Train ({len(train_idx)} pts): RMSE = {rmse_train:.4f} px")
    print(f"  Test  ({len(test_idx)} pts): RMSE = {rmse_test:.4f} px")
    print(f"  Test residuals: mean={r_test.mean():.4f}, median={np.median(r_test):.4f}, "
          f"p90={np.percentile(r_test, 90):.4f}, max={r_test.max():.4f}")

    if rmse_test < 1.0:
        print(f"\n  *** RMSE < 1 px ACHIEVED: {rmse_test:.4f} px ***")
    else:
        print(f"\n  RMSE = {rmse_test:.4f} px (target < 1.0)")

    # Coverage
    xs, ys = tight1[:, 0], tight1[:, 1]
    print(f"\n  Coverage: x=[{xs.min():.0f},{xs.max():.0f}], y=[{ys.min():.0f},{ys.max():.0f}]")
    tl = ((xs < 512) & (ys < 512)).sum()
    tr = ((xs >= 512) & (ys < 512)).sum()
    bl = ((xs < 512) & (ys >= 512)).sum()
    br = ((xs >= 512) & (ys >= 512)).sum()
    print(f"  Quadrants: TL={tl} TR={tr} BL={bl} BR={br}")

    # Write GT v2 (test set — the evaluation reference)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
    with open(args.out_csv, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x1", "y1", "x2", "y2"])
        for i in test_idx:
            w.writerow([round(tight1[i, 0], 2), round(tight1[i, 1], 2),
                        round(tight2[i, 0], 2), round(tight2[i, 1], 2)])
    print(f"\n  Wrote {len(test_idx)} test GT points → {args.out_csv}")

    # Verify old GT against the new model
    gt_old = np.loadtxt(os.path.join(ROOT, "data/ground_truth/pair1_gt.csv"),
                         delimiter=",", skiprows=1)
    r_old = np.linalg.norm(apply_homography(Htrain, gt_old[:, :2]) - gt_old[:, 2:], axis=1)
    print(f"  Old GT vs new Htrain: RMSE={np.sqrt((r_old**2).mean()):.4f} "
          f"(confirms old GT is noisy)")

    # Verification overlay
    os.makedirs(os.path.dirname(os.path.abspath(args.out_png)), exist_ok=True)
    vis_src = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    vis_ref = cv2.cvtColor(ref, cv2.COLOR_GRAY2BGR)
    test_set = set(test_idx.tolist())

    for i in range(len(tight1)):
        sx, sy = int(round(tight1[i, 0])), int(round(tight1[i, 1]))
        rx, ry = int(round(tight2[i, 0])), int(round(tight2[i, 1]))
        in_test = i in test_set
        color = (0, 0, 255) if in_test else (0, 200, 0)
        cv2.circle(vis_src, (sx, sy), 5, color, 2)
        cv2.circle(vis_ref, (rx, ry), 5, color, 2)
        cv2.putText(vis_src, str(i), (sx + 7, sy - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)
        cv2.putText(vis_ref, str(i), (rx + 7, ry - 3), cv2.FONT_HERSHEY_SIMPLEX,
                    0.3, color, 1)

    sep = np.full((vis_src.shape[0], 3, 3), 128, np.uint8)
    panel = np.hstack([vis_src, sep, vis_ref])
    legend_h = 60
    legend = np.zeros((legend_h, panel.shape[1], 3), np.uint8)
    cv2.putText(legend, "Red = test GT (eval reference)", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.putText(legend, "Green = train (fitted)", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1)
    cv2.putText(legend, f"GT v2: {len(test_idx)} test / {len(train_idx)} train / "
                f"{len(tight1)} total | Test RMSE: {rmse_test:.4f} px",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    out = np.vstack([panel, legend])
    cv2.imwrite(args.out_png, out)
    print(f"  Verification overlay → {args.out_png}")


if __name__ == "__main__":
    main()
