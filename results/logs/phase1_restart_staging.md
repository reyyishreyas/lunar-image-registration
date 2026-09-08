# Phase 1 Restart — Equal-GSD Staging + Photometric Preconditioning

**Branch:** `phase-1-best-model` **Date:** 2026-09-08
**Plan item:** Restart Phase 1 (post pair-2) — common-GSD staging, per-config
preconditioning, FINAL_CONFIG regression.

## What changed

### Equal-GSD working-set staging (no fixed `nac_fac` guess)
`src/preprocessing/georeference.py`:
- `ws_scale(H)`: OHRC-working-set -> NAC-working-set ground-scale ratio from the
  linear part of the overlap homography.
- `refine_ws_factor(fac, scale)`: next NAC factor so both working sets share a
  GSD (`fac' = fac * scale`).
- `georeference_pair(..., equal_gsd=False)`: opt-in loop (<= 4 iters) that
  re-runs the overlap search until the NAC factor locks. `nac_geom_csv` seeds
  the first guess from NAC per-line SPICE ground spacing when provided.
- Champion `FINAL_CONFIG` path is byte-identical (equal_gsd defaults off).

### Per-config photometric preconditioning
- `normalize.method: edges` -> `src/preprocessing/normalize.apply_edges`
  (Sobel magnitude, 1-99 percentile stretch to uint8).
- Existing `none` (enabled=false) and `clahe` unchanged.

## Measurements (pair 1, 1024x1024 crops)

| config | staging | precond | RMSE px | inliers | ratio | n | notes |
|---|---|---|---|---|---|---|---|
| FINAL | native | clahe 4.0 | 22.6172 | 235 | 0.8217 | 286 | champion, unchanged |
| C13 | equal-GSD | clahe 4.0 | 26.977 | 245 | 0.8249 | 297 | crop re-anchored; GT window differs |
| C14 | native | edges | 742.1478 | 10 | 0.5882 | 17 | gradients-only fails OHRC |

### C13 equal-GSD self-calibration
- Working-set factor trajectory: 8 -> 3 -> 3 (locked).
- Iter scales: 0.3875 (fac 8), then 1.0317 (fac 3) -> the historical fixed
  NAC_WS_FACTOR=8 was ~2.7x too coarse relative to OHRC working set.
- Lock at fac 3: NAC ws and OHRC ws match ground resolution; found transform
  flipH, 480 SIFT inliers in the search; crop window moved to
  (pixel 5428, scan 46333).
- RMSE vs the fixed `pair1_gt.csv` rose (26.98) because the GT correspondence
  points are anchored to the old crop window — the two staging modes are not
  directly comparable on RMSE; inlier count/ratio improved marginally
  (245/0.8249 vs 235/0.8217).

### C14 edges preconditioning
- Collapsed the pair to 17 matches / 10 inliers (RMSE 742.1). Pair-1 radiometry
  is comparable between sensors, so stripping to gradients destroys usable
  texture. CLAHE remains the champion preconditioner.
- `edges` is retained as an option for genuinely photometrically-incompatible
  pairs (e.g., pair-2 geometry path), not the default.

## Regression gate
- FINAL_CONFIG (results/final_config.yaml) after the refactor:
  **RMSE 22.6172 px, 235 inliers, 0.8217** — matches the Phase 4 champion
  exactly (verified twice, see ablation.csv FINAL rows).

## Tests
- `tests/test_staging.py`: 8 new unit tests (ws_scale, refine bounds/lock,
  geometry-CSV seed + fallback, apply_edges gradient response).
- Full suite: `venv/bin/python -m pytest tests/ -q` -> **19 passed**.
- Note: `third_party/.  _test.py` is AppleDouble noise (present before).

## Artifacts
- configs: `experiment_C13_equal_gsd.yaml`, `experiment_C14_edges.yaml`
- data/processed_equal_gsd/: `pair1_src.png`, `pair1_ref.png`,
  `georef_pair1.json` (equal_gsd, ws_factor_iters, nac_ws_factor_final=3)
- results/logs/ablation.csv: C13, C14, FINAL regression rows appended
- results/figures/pair1_matches_C13.png, pair1_matches_C14.png