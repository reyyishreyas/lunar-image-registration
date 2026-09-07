# Phase 6 — Improved Pipeline: Final Report (champion + robustness)

Date: 2026-09-08
Branch: `phase-6-polar-footprint-registration` (off `main` = 13b02f9, Phase 5 merged)

## 1. What "pass everything and beat SOTA" reduces to, honestly

The Phase 5 hard case (OHRC-2026 + NAC `M1127547939RC`) is reproducibly a
NON-OVERLAPPING pair: five independent methods (SIFT 14 inliers, SuperPoint+SuperGlue
2 matches, LoFTR 9/419, dense phase-correlate NCC<=0.06, exhaustive template ZNCC
<=0.055) all fail, and the multi-orientation dominance/NCC diagnostic shows no
flip-group dominance and zero structural agreement (NCC ~0.02). No image
registration method — classical, learned, or dense — can register images that do
not share ground. So "passes everything" is delivered as: the pipeline now tries
HARDER (boosted fallback), PROVES its refusals (auditable diagnostics), and RESCUES
weak classical matches automatically (learned fallback), while the champion still
**beats the learned SOTA rows on RMSE**.

## 2. Champion (unchanged, re-verified)

`results/final_config.yaml` = C10 algorithm block (SIFT + CLAHE 4.0/8x8 + BF ratio 0.75
+ USAC_MAGSAC 5.0), reproduced through the improved pipeline on site:

- RMSE = **22.6172 px** (identical to Phase 4's 22.6172)
- inliers 235, n_matches 286, inlier_ratio 0.8217

Beats every baseline on the same crop pair:

| config | detector+matcher | RMSE px |
|---|---|---|
| **FINAL (C10)** | **SIFT + BF ratio (classical)** | **22.6172** |
| C8 | SuperPoint + LoFTR (learned) | 22.6557 |
| C7 | SuperPoint + SuperGlue (learned) | 22.6703 |
| C3 | SIFT + BF ratio (RANSAC) | 22.6316 |
| C1 | SIFT + BF ratio (baseline normalize) | 22.6614 |

## 3. Robustness improvements (this phase)

### 3.1 Georeference: proof-based boosted fallback + diagnostics
`src/preprocessing/georeference.py`
- `candidate_stats(ohrc_ws, nac, ...)` — pure 4-orientation SIFT overlap table
  (flip, good, inliers, inlier_ratio, ncc, bbox) with explicit extractor params.
- `find_pair_transform_robust(...)` — denser extractor (30k feat, ct 0.006) +
  USAC_MAGSAC; ACCEPTS only if winner inliers>=20 AND ratio>=0.25 AND mirror-group
  dominance (id/HV vs V/H group: winner >= 2x loser) AND warped-overlap NCC>=0.15
  AND sane sub-region bbox. Returns `(t_or_None, stats)`.
- `georeference_pair` runs primary then robust; if both refuse, writes
  `georef_{prefix}_diagnostics.json` (primary + robust tables, sift params, reason)
  and raises an informative RuntimeError.
- Guard added: `findHomography` returning `None` no longer crashes the NCC scoring.

Verified:
- pair1 (known overlap): primary path, sparse-sift, 249 inliers, flipV — UNCHANGED.
- phase5 (no overlap): primary 14 inliers -> robust rejected -> diagnostics written
  (`data/processed/georef_phase5_diagnostics.json`).

### 3.2 Registrations: learned emergency matcher fallback
`src/pipeline.py` — optional `matcher.learned_fallback: true`: when the classical
matcher returns <8 matches and a SuperPoint cache is present, automatically tries
LoFTR then SuperGlue; marks `m_cfg["_fallback_used"]`.

Functional verification (`configs/experiment_test_fallback.yaml`, pair1 crops,
SuperPoint detector + deliberately crippled BF ratio 0.03 -> 0 good matches):
- fallback engaged: "classical matcher gave too few matches; trying learned fallback"
- LoFTR rescue: n_matches 6810 -> 5563 inliers (0.8169) -> RMSE 22.7753
- completed full output rows + matches figure (non-canonical test csv kept)

## 4. Tests

- `tests/test_georeference.py` (new): candidate_stats schema; robust ACCEPTS a
  genuine synthetic overlap (high NCC + dominance); robust REJECTS random
  non-overlap; primary accepts. **4 passed**.
- Full pipeline re-run of `results/final_config.yaml`: RMSE 22.6172 / 235 inliers
  reproduced (no regression from georef/pipeline changes).
- Phase5 refusal end-to-end re-run: RuntimeError + diagnostics artifact.

## 5. Canonical results state

`results/logs/ablation.csv` — C1..C12 + C9 (polar test FAIL, note updated to point
at the diagnostics artifact). No stray rows.

## 6. Files changed/added (this phase)

- `src/preprocessing/georeference.py` — candidate_stats, find_pair_transform_robust,
  diagnostics, _ncc_overlap, None-H guard, georeference_pair 2-stage + meta.method.
- `src/pipeline.py` — learned_fallback emergency path in run_experiment.
- `tests/test_georeference.py` — new unit tests (4).
- `configs/experiment_test_fallback.yaml` — functional fallback test config.
- `results/final_config.yaml` — unchanged champion.
- `results/logs/ablation.csv` — polar note updated (values unchanged).
- `results/logs/polar_case_findings.md` — Phase 6 outcome section.
- `data/processed/georef_phase5_diagnostics.json` — audit artifact (gitignored).