# Phase 2 Worklog — Preprocessing (Configs C2–C4)

Branch: phase-2-preprocessing (from main @1d15306)

## 2.1 Histogram matching (Config C2)
- Added `src/preprocessing/normalize.py::histogram_match` (skimage.exposure.match_histograms).
- Direction: match the OHRC crop (src) histogram to the NAC crop (ref); both uint8, out uint8.
- Pipeline integration: `apply_normalize()` in src/pipeline.py dispatches on config `normalize.method`
  (histogram_match / clahe / gamma_shadow), applied before detection; ablation `preproc` column now
  reflects the method (C1 row string unchanged).
- C2 run: georef+histmatch — rmse 22.6605, inliers 223/325, time 2.68s.
  Matches 61 -> 325 (better radiometric agreement -> far more SIFT correspondences).
- Fixed cosmetic console header (was hardcoded "C1 results").

## 2.2 CLAHE sweep + Config C3
- Added `normalize.py::apply_clahe` (cv2.createCLAHE, tile 8x8 default).
- Sweep trials (clip 2.0/4.0/8.0, 8x8) run via pipeline, rows logged to results/logs/clahe_sweep.csv:
  clip 2.0 -> 22.0273? NO -> 23.0273 (154 inliers/201), 4.0 -> 22.6316 (236/286),
  8.0 -> 22.6496 (241/315).
- Selected clipLimit=4.0 (lowest RMSE). Wrote configs/experiment_C3.yaml; official C3 run appended:
  rmse 22.6316, inliers 236/286, ratio 0.8252, time 1.03s.

## 2.3 Gamma shadow correction (Config C4)
- Added `src/preprocessing/shadow_correct.py::gamma_shadow_correct` (gamma 0.5, 50%-quantile mask).
- C4 run: rmse 8353.4457, inliers 6/40 (ratio 0.15) — gamma shadow correction DEGRADED matching.
- Diagnosis: sharp binary mask (<= median) applies gamma to ~50% of pixels, creating an intensity
  discontinuity and spurious edges in flat dark terrain; SIFT matches collapse (40 matches, 6 inliers).
  Sanity-checked the function itself (changed_px ~0.5, value range retained, shadows brightened,
  highlights unchanged) — the pipeline result is a genuine algorithm measure, not a bug.
- Recorded as-is (no tuning). C2/C3 are the useful preprocessings for Phase 3.

## Verify (Phase 2)
- ablation.csv rows C1..C4, all numeric fields present. PASS.
- clahe_sweep.csv 3 rows (2.0/4.0/8.0). PASS.
- Module tests (synthetic 64x64) for histogram_match / apply_clahe / gamma_shadow_correct PASS.

## Verification re-run (independent, reproducible)
Full re-run of every Phase 2 experiment through the ACTUAL pipeline + YAML configs:
- C2 (configs/experiment_C2.yaml): reproduced 22.6605 / 223 / 0.6862 / 325 (time 2.02 s measured).
- CLAHE sweep (trials via C1 base + normalize clahe clip 2/4/8, tile 8): 2.0 -> 23.0273 (154/201,
  ratio 0.7662), 4.0 -> 22.6316 (236/286, 0.8252), 8.0 -> 22.6496 (241/315, 0.7651). Selected
  clipLimit = 4.0 by lowest RMSE.
- C3 (configs/experiment_C3.yaml): run twice -> identical 22.6316 / 236 / 0.8252 / 286
  (times 2.19 s, 0.83 s); canonical run recorded 0.90 s. Metrics deterministic & reproducible.
- C4 (configs/experiment_C4.yaml): reproduced 8353.4457 / 6 / 0.15 / 40 (time 1.06 s).
- Reproducibility of metrics: exact every time. Only runtime varies (CPU/FUSE scheduling).
- C4 genuineness: same 1024x1024 crops, same GT file/convention, H finite but wrong (6/40
  speculative inliers -> GT residuals 73.5..29610 px); same eval on C3's H -> 22.6316 px.
- Canonical results/logs/ablation.csv rebuilt: C1 (Phase 1 immutable) + verified C2/C3/C4.
- Canonical results/logs/clahe_sweep.csv rebuilt: 3 trials with full reproducibility columns.
- C1 row, data/ground_truth/pair1_gt.csv, pair crops all verified unmodified (GT sha1 matches
  committed file). Inlier ratios math-checked for every row. No NaN/blank/fabricated values.