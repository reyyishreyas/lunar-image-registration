# Phase 1 working log

## Pair change (documented in manifest + plan)
- Phase 0 tagged M1430572656LC as phase1 NAC candidate.
- Feature co-registration against the OHRC-2021 ground grid: M1430572656LC -> ~5 SIFT inliers (no overlap). REJECTED per OVERLAP_ANALYSIS criteria.
- Across all 43 NACs, M1469248775LC / LE had the strongest verified overlap with OHRC-2021 (LC: 272 inliers in sweep; final georef: 249 inliers, flipV, NCC-confirmed).
- User approved swapping phase1 NAC to M1469248775LC. Manifest updated.

## Step 1.1 georeference
- data/processed/pair1_src.png (OHRC-2021 1024x1024 crop),
  pair1_ref.png (LRO NAC M1469248775LC warped crop), georef_pair1.json.
- OHRC crop ground area: lon 336.576..336.585, lat -3.4074..-3.4168.
- NAC flip = vertical; working-set factors OHRC=10, NAC=8; transform H in georef json.

## Step 1.2-1.5 (Config C1)
- detect_sift: 328 kps (OHRC crop), 2959 kps (NAC crop).
- match_bf_ratio(0.75): 61 good matches.
- find_homography_ransac(5.0): 56 inliers (ratio 0.918).
- inlier residual: mean 1.05 px, RMS 1.26 px, max 3.24 px (on crops).
- results/figures/pair1_matches.png saved.
- runtime total 0.41 s (excl. georef).

## Blocked (manual)
- Step 1.6 needs a human to pick 20 GT control points (scripts/pick_ground_truth.py).
  RMSE (Step 1.7/1.9) numeric value pending that.

## Final C1 numbers (after human GT)
- GT: 21 control points in data/ground_truth/pair1_gt.csv (user's first 12-point attempt
  was internally inconsistent -> re-picked with guidance).
- C1 RMSE = 22.66 px over all 21 GT points (pipeline-computed, logged in ablation.csv).
- Cleaned RMSE (19/21, exclude 2 outright miss-clicks >3x median) = 11.21 px; median 9.3 px.
- GT self-consistency: 9/21 points within 5 px of fitted homography, RMS 2.78 px among them.
- SIFT inlier residuals remained 1.05 px mean / 1.26 px RMS (unaltered by GT).

## Module tests
- rmse(exact H, exact GT)=0.0; ransac-recovered H rmse~1e-5; visualize writes PNG OK.
