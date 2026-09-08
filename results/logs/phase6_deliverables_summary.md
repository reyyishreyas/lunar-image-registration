# Phase 6 — Demo, final table, presentation, report: Phase Summary

**Branch:** `phase-6-deliverables` (off `main` = aa6695d)
**Date:** 2026-09-08

## Steps completed

- [x] 6.1 Streamlit demo — `demo/app.py` (calls `src.pipeline.run_experiment`
      with a FINAL_CONFIG-derived YAML only; no reimplemented logic). Modes:
      pair-1 (RMSE 22.6172 verified through the demo config), pair-2 (expected
      refused with diagnostics path), upload aligned crops. Displays match
      overlay, checkerboard of staged aligned crops, RMSE/inliers/ratio, GSD.
- [x] 6.2 Final ablation table — `results/final_ablation_table.csv` (16 configs,
      C9-polar/C9-grid disambiguated); official ablation log cleaned of demo-run
      row pollution; demo runs route to results/logs/demo_ablation.csv.
- [x] 6.3 Presentation deck — `docs/deck.md` (7 mandated sections in order).
- [x] 6.4 Written report — `docs/report.md` (~6 pages, all mandated sections).
- [x] Section 9 demo fallback — `demo/fallback/demo_recording.mp4`,
      `cached_final_output.json`, `pair1_matches_FINAL.png`,
      `pair1_checkerboard.png`; live demo is classical-only (no learned
      inference, no network).
- [x] Section 1 mandatory deliverables — all 8 checked in the plan.

## Numbers produced

- FINAL_CONFIG pair-1 regression (through the demo path): RMSE 22.6172 px, 235
  inliers, 0.8217, 286 matches.
- Final table: 16 configurations; best = FINAL/C10 (SIFT + RAFE 4.0 + BF 0.75 +
  USAC_MAGSAC 5.0): RMSE 22.6172 / 235 / 0.8217. Beats C8 LoFTR 22.6557 and C7
  SuperGlue 22.6703.
- Pair-2 refusal (demo, headless): georeference refused (14 inliers) with
  diagnostics written — geometry fallback documented in pair-2 logs.
- Fallback artifacts: demo_recording.mp4 (1.8 MB), cached_final_output.json.

## Testing

- `venv/bin/python -m pytest tests/ -q` -> **24 passed**.
- `venv/bin/python -m py_compile demo/app.py scripts/*.py` -> clean.
- Headless end-to-end through the demo config builders: pair-1 FINAL 22.6172/235,
  C14 edges 742.1/10, pair-2 expected refusal + diagnostics file.

## Files created/modified

- demo/app.py (new); demo/fallback/* (4 artifacts); docs/deck.md; docs/report.md;
  results/final_ablation_table.csv; results/logs/ablation.csv (cleaned);
  scripts/build_final_table.py; scripts/make_demo_fallback.py;
  AI_EXECUTION_PLAN.md (6.1-6.4, Section 9, Section 1 checklist marked).

## Git

- Branch: phase-6-deliverables
- Commit: b28a598 (summary committed separately)
- Push: SUCCESS

## Status

READY FOR MERGE into main by user review.