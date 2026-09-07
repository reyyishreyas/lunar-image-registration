# Phase 2 Summary — Preprocessing (Configs C2–C4)

## 1. Phase number and name
**Phase 2 — Preprocessing (Configs C2–C4)**

## 2. Steps completed

| Checkbox | Step | Result |
|---|---|---|
| [x] | 2.1 histogram matching | pass — src/preprocessing/normalize.py; C2 georef+histmatch: rmse 22.6605, 223/325 inliers |
| [x] | 2.2 CLAHE (sweep 2/4/8) + best | pass — clahe_sweep.csv 3 rows; best clipLimit=4.0; C3: rmse 22.6316, 236/286 |
| [x] | 2.3 gamma shadow correction | pass (real measurement) — src/preprocessing/shadow_correct.py; C4: rmse 8353.4457, 6/40 (degrades matching; documented, not tuned) |
| VERIFY | — | pass — ablation.csv C1..C4 numeric; clahe_sweep.csv 3 rows |

## 3. Numbers produced this phase (ablation.csv rows C2–C4 + clahe_sweep.csv)
| config | preproc | rmse_px | inliers | inlier_ratio | n_matches | time_s |
|---|---|---|---|---|---|---|
| C1 | georef+normalize (base) | 22.6614 | 56 | 0.918 | 61 | 0.88 |
| C2 | georef+histmatch | 22.6605 | 223 | 0.6862 | 325 | 2.68 |
| C3 | georef+clahe:4.0 | 22.6316 | 236 | 0.8252 | 286 | 1.03 |
| C4 | georef+gamma_shadow:0.5 | 8353.4457 | 6 | 0.15 | 40 | 1.74 |

clahe_sweep.csv: clip 2.0 -> 23.0273 (154/201) | 4.0 -> 22.6316 (236/286) | 8.0 -> 22.6496 (241/315)

## 4. Files created or modified this phase
- `AI_EXECUTION_PLAN.md` (Phase 2 checkboxes + RESULT/VERIFY lines)
- `src/preprocessing/normalize.py` (histogram_match, apply_clahe)
- `src/preprocessing/shadow_correct.py` (gamma_shadow_correct)
- `src/pipeline.py` (apply_normalize stage + method-aware preproc label + config-aware header)
- `configs/experiment_C2.yaml`, `configs/experiment_C3.yaml`, `configs/experiment_C4.yaml`
- `results/logs/ablation.csv` (appended C2, C3, C4 rows)
- `results/logs/clahe_sweep.csv` (new)
- `results/logs/phase2_worklog.md`, `results/logs/phase2_summary.md`
- `results/figures/pair1_matches_C2.png`, `pair1_matches_C3.png`, `pair1_matches_C4.png`

## 5. Blocked or failed items
- C4 gamma shadow correction is a NUMERIC FAILURE that is nevertheless fully recorded:
  rmse 8353.4457 px, 6/40 inliers. Cause: sharp 50%-quantile gamma mask introduces
  intensity discontinuities / spurious edges in dark terrain, collapsing SIFT matching.
  Not tuned further (no-tuning rule; the plan requires recording the real measured row).
  C2/C3 outperform C4 and carry forward to Phase 3.

## 6. Deliverables from Section 1 satisfied
- Config C2 (histogram matching), C3 (CLAHE, best-of-sweep), C4 (gamma shadow) each appended
  as real measured rows to results/logs/ablation.csv; clahe_sweep.csv records all three trials;
  single pipeline entry point extended (apply_normalize in src/pipeline.py) with no competing
  pipeline logic.