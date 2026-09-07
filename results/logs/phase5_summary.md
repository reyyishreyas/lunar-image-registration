# Phase 5 Summary — Polar / low-sun hard case

Branch: `phase-5-polar-hard-case` (off `main` = 3c543d2)
Date: 2026-09-08

## Task
Run the Phase 4 FINAL_CONFIG pipeline, unchanged, on the `role=phase5` polar
pair; log a real result or an explicit failure note; test whether learned
matchers (SuperGlue/LoFTR) succeed where classical methods fail.

## Pair
- Source: OHRC-2026 `data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img`
  (12000 x 93692, uint8, sun_el ~1.9 deg — extreme low sun)
- Reference: `data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG`
- No manual ground truth exists for this pair.

## Code changes (needed to run FINAL_CONFIG unchanged on a second pair)
- `src/preprocessing/georeference.py`: `read_ohrc_raw` now infers the scan-line
  count from file size (Phase 1 = 93693, Phase 5 = 93692) instead of the
  hardcoded `OHRC_SHAPE`; `georeference_pair` gained a `prefix` parameter so
  outputs are `{prefix}_src.png`, `{prefix}_ref.png`, `georef_{prefix}.json`
  (Phase 1 outputs untouched).
- `src/pipeline.py`: `maybe_georeference` passes `prefix` through from config.
- `configs/experiment_C9_polar.yaml`: algorithm blocks byte-identical to
  FINAL_CONFIG (C10); only I/O paths + `prefix: phase5` differ.
- `scripts/run_polar_learned.py`: Step 5.3 feasibility checker (SuperGlue +
  LoFTR on the same georef working-set images).

## Results
| Item | Classical SIFT (FINAL_CONFIG) | SuperPoint+SuperGlue | LoFTR (832x832) |
|---|---|---|---|
| Overlap search | 14 inliers < 20 -> FAIL | 2 matches -> FAIL | 419 matches, 9 inliers (2.1%) -> FAIL |
| Crops produced | No | No | No |
| Homography / RMSE | None (RMSE=FAIL) | None | None |

- ablation.csv row `C9 (polar test)`: rmse_px=FAIL, inliers=0, inlier_ratio=0,
  n_matches=0, time_s=37.9, notes="georeference: no reliable OHRC<->NAC overlap
  (inliers=14). Cannot produce same-ground-area crops."
- Terminal cause: with sun_el ~1.9 deg the low-sun OHRC-2026 has almost no
  stable contrast structure at working-set scale, so no geo-overlap can be
  auto-found by ANY available matcher. Failure happens at the GEO-REFERENCE
  stage (overlap search), not the registrations stage. Learned methods do NOT
  produce usable matches here while classical fails — no differentiation slide
  for Phase 6 from this pair. Full write-up:
  `results/logs/polar_case_findings.md`.

## Verification
- `results/logs/ablation.csv` contains the C9 polar row. ✔
- `results/logs/polar_case_findings.md` exists with pass/fail + explanation. ✔
- Row format: failure explicitly logged (REQUIRED by plan 5.2 — a failure
  result is a completed deliverable). ✔

## Files created/modified
- `src/preprocessing/georeference.py` (dynamic shape + prefix)
- `src/pipeline.py` (prefix passthrough)
- `configs/experiment_C9_polar.yaml` (new)
- `scripts/run_polar_learned.py` (new)
- `results/logs/ablation.csv` (C9 polar FAIL row + notes column)
- `results/logs/polar_case_findings.md` (new)
- `results/logs/polar_learned_run.log`, `results/logs/polar_loftr_run.log` (new)
- `AI_EXECUTION_PLAN.md` (Phase 5 marked [x] + RESULT lines)

## Git
- Branch: phase-5-polar-hard-case
- Commit + push pending at time of write.