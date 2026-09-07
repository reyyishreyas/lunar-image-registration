# Phase 1 Summary — End-to-end baseline pipeline (Config C1)

## 1. Phase number and name
**Phase 1 — End-to-end baseline pipeline (Config C1)**

## 2. Steps completed

| Checkbox | Step | Result |
|---|---|---|
| [x] | 1.1 georeference | pass — pair1_src.png + pair1_ref.png (1024x1024) on OHRC-2021 + LRO NAC M1469248775LC; crops share lon 336.576..336.585 / lat -3.407..-3.417 |
| [x] | 1.2 detect_sift | pass — 328 kps (OHRC crop) / 2959 kps (NAC crop) |
| [x] | 1.3 match_bf_ratio | pass — 61 good matches at ratio 0.75 |
| [x] | 1.4 find_homography_ransac | pass — 56 inliers (ratio 0.918) |
| [x] | 1.5 visualize | pass — results/figures/pair1_matches.png (2.4 MB) |
| [x] | 1.6 manual GT | pass (human) — 21 control points in data/ground_truth/pair1_gt.csv |
| [x] | 1.7 metrics rmse | pass — C1 RMSE 22.66 px (21 points) |
| [x] | 1.8 pipeline | pass — src/pipeline.py + configs/experiment_C1.yaml + scripts/run_pipeline.py |
| [x] | 1.9 ablation | pass — C1 row in results/logs/ablation.csv, all numeric columns filled |
| VERIFY | — | pass — C1 row numeric everywhere; pair1_matches.png exists |

## 3. Numbers produced this phase

| Key | Value |
|---|---|
| pair1 crops | 1024x1024, OHRC crop at lon 336.576-336.585, lat -3.4074..-3.4168 |
| georef NAC overlap | 249 SIFT inliers, flipV (NCC-confirmed mirror) |
| SIFT keypoints (src/ref) | 328 / 2959 |
| matches (ratio 0.75) | 61 |
| RANSAC inliers | 56 (ratio 0.918) |
| SIFT inlier residual | mean 1.05 px, RMS 1.26 px, max 3.24 px |
| GT points | 21 |
| C1 RMSE | 22.66 px (all 21) |
| C1 RMSE cleaned 19/21 | 11.21 px; median 9.31 px |
| C1 runtime | 0.88 s pipeline (+ georef ~2 s) |

## 4. Files created or modified this phase
- `AI_EXECUTION_PLAN.md` (Phase 1 checkboxes + RESULT/VERIFY lines)
- `data/manifest.csv` (phase1 NAC swapped to M1469248775LC; M1430572656LC -> reference, rejection noted)
- `src/preprocessing/georeference.py`
- `src/detection/classical.py`
- `src/matching/classical_match.py`
- `src/outlier_rejection/ransac.py`
- `src/evaluation/visualize.py`
- `src/evaluation/metrics.py`
- `src/pipeline.py`
- `configs/experiment_C1.yaml`
- `scripts/run_pipeline.py`
- `scripts/pick_ground_truth.py`
- `data/processed/pair1_src.png`, `pair1_ref.png`, `georef_pair1.json` (gitignored)
- `data/ground_truth/pair1_gt.csv` (gitignored)
- `results/figures/pair1_matches.png` (gitignored)
- `results/logs/ablation.csv`, `results/logs/phase1_worklog.md` (gitignored; force-added)

## 5. Blocked or failed items
- None. Deviations documented:
  - Phase 0 phase1 NAC M1430572656LC rejected (no overlap with OHRC-2021; ~5 SIFT inliers). User approved replacement M1469248775LC (feature-verified 249 inliers). Recorded in manifest + plan + summary.
  - Plan target 20 GT points; user provided 21 (acceptable).
  - `results/logs/*` gitignored; plan deliverables force-added (same as Phase 0).

## 6. Deliverables satisfied
- Complete end-to-end baseline pipeline (preprocessing -> detection -> matching -> mismatch
  rejection -> evaluation) driven by one config, with real measured C1 results in
  `results/logs/ablation.csv` and a verified matches figure. This is the single entry
  point that Phases 2-6 extend.