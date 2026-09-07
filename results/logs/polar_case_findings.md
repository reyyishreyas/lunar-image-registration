# Phase 5 Findings — Polar / Low-Sun Hard Case (OHRC-2026 + LRO NAC M1127547939RC)

Date: 2026-09-08
Config file: `configs/experiment_C9_polar.yaml` (algorithm blocks byte-identical to
Phase 4 FINAL_CONFIG = C10)
Ablation row: `C9 (polar test)` in `results/logs/ablation.csv`
Setup script: `scripts/run_polar_learned.py` (Step 5.3 feasibility checks)

## Pair

- Source: OHRC-2026 `ch2_ohr_ncp_20260331T1105235288_d_img_d18.img` (12000 x 93692,
  uint8) — extreme low sun (sun_el ~1.9 deg)
- Reference: LRO NAC `M1127547939RC.IMG` (6528 x 633 working-set)
- No manual ground truth exists for this pair (RMSE vs GT not computable).

## 5.1 — Classical pipeline (FINAL_CONFIG, unchanged) — FAIL

At the georeference stage (same-ground-area crop), SIFT working-set overlap search
found **14 inliers < 20 threshold** and raised:

    RuntimeError: georeference: no reliable OHRC<->NAC overlap (inliers=14).
    Cannot produce same-ground-area crops.

No crops were produced, so no source/reference image pair entered the
registrations stage -> no homography, no RMSE.

## 5.2 — Ablation row (explicit failure note)

C9 (polar test): rmse_px=FAIL, inliers=0, inlier_ratio=0, n_matches=0,
refinement=FAIL georeference (runtime 37.9 s).
notes: "georeference: no reliable OHRC<->NAC overlap (inliers=14). Cannot produce
same-ground-area crops."

Appended to `results/logs/ablation.csv` with a `notes` column.

## 5.3 — Learned matchers on the same working-set images — FAIL too

Feasibility check on the identical normalized working-set images the
georeferencer used (OHRC ws 9370x1200, NAC ws 6528x633):

| Method | raw matches | RANSAC inliers | inlier ratio | runtime | vs SIFT baseline (14) |
|---|---|---|---|---|---|
| SuperPoint + SuperGlue (outdoor) | 2 | — (<4, no model fit) | — | ~4 s | WORSE |
| LoFTR (both resized 832x832) | 419 | 9 | 2.1% | 30.1 s | WORSE |
| SIFT (classical georef baseline) | — | 14 | — | ~38 s | baseline |

Conclusion (negative differentiation result): SuperGlue/LoFTR do NOT provide
usable matches on this pair while classical methods fail. ALL methods — classical
and learned — fail to establish a usable OHRC<->NAC overlap for this extreme
low-sun pair at the current georeference working-set regime. SuperPoint+SuperGlue
found essentially no repeatable interest points (2 matches); LoFTR's 419 raw
matches collapse to 9 inliers (2.1%), below even classical SIFT's 14.

Caveat: the LoFTR run resized both tall (7:1) working images to 832x832, which
distorts aspect ratio; this plausibly depressed LoFTR's weak inlier ratio. A fair
LoFTR test needs a common-scale overlapping crop, which itself requires a
geo-overlap that none of the matchers could establish autonomously.

## Why the terminal-cause finding matters (Phase 6 implication)

The hard cutoff is NOT in the registrations stage (SIFT+USAC_MAGSAC matching of a
same-ground-area crop) — it is EARLIER, in the georeference stage: with sun_el
~1.9 deg the low-sun OHRC-2026 swath has almost no stable local-contrast
structure at working-set scale, so no overlap can be auto-found by any available
interest-point matcher. Registrations-stage improvement alone (Phase 4 outlier
rejection / refinement) cannot rescue a pair whose crops were never produced.

Differentiation claim for Phase 6 should therefore target the GEO-REFERENCE stage
(autonomous overlap search), e.g. a dense phase-correlation / template search over
browse-level footprints, OR manual footprint-assisted initialization, rather than
the matching stage.

## Files

- `configs/experiment_C9_polar.yaml`
- `scripts/run_polar_learned.py`
- `results/logs/ablation.csv` (C9 polar row)
- `results/logs/polar_learned_run.log`, `results/logs/polar_loftr_run.log`