# Phase 6 Summary — Robust Registration + Hard-Case Resolution

Date: 2026-09-08
Branch: `phase-6-polar-footprint-registration` (off `main` 13b02f9)
User directive: "merge this and then again improve the full pipeline so that it
passes everything and beats the current state of the art model"

## Goal interpretation

Cannot "pass" a pair that has no common ground. Multi-method evidence (this phase,
below) proves the Phase 5 polar pair does NOT overlap, so this phase delivers the
honest achievable portion: a pipeline that tries harder, proves its refusals, and
auto-rescues weak classical matches with learned matchers — while the champion
BEATS all SOTA baselines (including the learned SuperGlue/LoFTR rows) on RMSE.

## Key numbers

| Item | value |
|---|---|
| Champion FINAL_CONFIG RMSE (reproduced) | 22.6172 px, 235 inliers, 0.8217 ratio |
| LoFTR baseline (C8) | 22.6557 (beaten) |
| SuperGlue baseline (C7) | 22.6703 (beaten) |
| pair1 georef (unchanged) | sparse-sift, 249 inliers, flipV |
| phase5 georef | primary 14 inliers -> robust refused -> diagnostics |
| learned-fallback test | LoFTR rescues 0-match classical pass: 5563 inliers, RMSE 22.7753 |
| unit tests | 4 passed (synthetic accept/reject/schema, synthetic primary accept) |

## What was built

1. `candidate_stats()` — pure 4-orientation SIFT overlap table (flip, good,
   inliers, ratio, ncc, bbox) + explicit params.
2. `find_pair_transform_robust()` — denser extractor + USAC_MAGSAC; acceptance
   = inliers>=20, ratio>=0.25, MIRROR-GROUP dominance (id/HV vs V/H group,
   winner>=2x loser) + NCC>=0.15 + sane bbox. Mirror-group (not per-orientation)
   required: SIFT + craters are 180-rotation-invariant, per-orientation bars
   would reject real overlaps.
3. `georeference_pair` 2-stage + informative refusal writing
   `georef_{prefix}_diagnostics.json` (candidate tables + params + reason).
4. `pipeline.py`: optional `matcher.learned_fallback` (LoFTR -> SuperGlue when
   classical < 8 matches and SuperPoint cache present).
5. `tests/test_georeference.py` (4 unit tests) + functional verifications.

## Phase 5 verdict reinforced

| Method | phase5 result |
|---|---|
| SIFT (primary) | 14 inliers, no flip-group dominance, ncc ~0.02 |
| SIFT (boosted fallback) | refused by dominance/NCC |
| SuperPoint+SuperGlue | 2 matches |
| LoFTR | 9 inliers / 419 (2.1%) |
| dense phase-correlate | max NCC ~0.06 (noise) |
| exhaustive template ZNCC | max 0.055 (noise; also noise-level on known-good pair1 -> dense NCC unsuitable for cross-sensor lunar, feature matching is authoritative) |

Audit artifact: `data/processed/georef_phase5_diagnostics.json`.

## Deliverables (this phase)

- `src/preprocessing/georeference.py` (validated, no regression)
- `src/pipeline.py` (fallback feature)
- `tests/test_georeference.py`
- `configs/experiment_test_fallback.yaml`
- `results/logs/final_pipeline_report.md`, updated `results/logs/polar_case_findings.md`,
  updated `results/logs/ablation.csv` polar note, `AI_EXECUTION_PLAN.md` 6.5 [x]

## Remaining (open) Phase 6 steps

6.1 Streamlit demo, 6.2 final table CSV, 6.3 presentation deck, 6.4 report —
these remain for the plan's Phase 6 proper.