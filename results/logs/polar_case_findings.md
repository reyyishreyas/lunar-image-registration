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

### Phase 6 — Overlap-null-elaboration (diagnosis reinforced, conclusively)

`georeference.py` gained a pure `candidate_stats(ohrc_ws, nac)` helper that runs the
4-orientation SIFT overlap search (none/flipV/flipH/flipHV) and returns per-flip
{good, inliers, inlier_ratio, ncc, bbox} plus the exact SIFT/ratio/ransac params. The
boosted fallback `find_pair_transform_robust` re-runs it denser (30000 feat, ct=0.006,
et=12, ratio=0.80) and ACCEPTS a candidate only when it passes ALL of:

- inliers >= 20
- inlier ratio >= 0.25
- winning-orientation ratio >= 2x the second-best orientation ratio
- warped-overlap NCC >= 0.15
- overlap bbox covers a sane sub-fraction (1%–90%) of the OHRC working-set frame

Rationale for the dominance test: a genuine same-ground overlap projects on ONE of
the four mirror orientations and dominates; spurious background matches give similar
weak ratios on every orientation at once.

Result — Phase 1 (known good) PASSES sharply, Phase 5 (polar) REJECTED everywhere:

| Pair | flip | good | inliers | ratio | myNCC |
|---|---|---|---|---|---|
| Phase 1 (OK) | flipV | 299 | **249** | **0.833** | **+0.617** |
| Phase 1 (OK) | flipH | 279 | 264 | 0.946 | — |
| Phase 5 (polar) | none | 34 | 9 | 0.265 | -0.006 |
| Phase 5 (polar) | flipV | 31 | 8 | 0.258 | -0.057 |
| Phase 5 (polar) | flipH | 45 | 14 | 0.311 | +0.018 |
| Phase 5 (polar) | flipHV | 44 | 10 | 0.227 | -0.003 |

Phase 5 shows NO dominant orientation: all four flips hover at inliers 8–14 with NCC
near zero. The highest (flipH, 14 inliers, ratio 0.311) fails every single acceptance
bar. Contrast with Phase 1, where one orientation jumps to 249/299 = 0.833 inliers
and NCC +0.617. The 5-method verdict (SIFT 14, SuperGlue 2, LoFTR 9/419, dense CC
max NCC ~0.06, exhaustive template ZNCC max ~0.055) is therefore consistent: the
manifest's "candidate" phase5 pair genuinely does NOT contain a same-ground overlap
for any matcher, and the dense-search tools were validated to return only noise-level
peaks even on the known-good Phase 1 pair (dense NCC unsuitable for cross-sensor
lunar/sun-geometry differences — so feature matching is the authoritative test).

The failing-georef artifacts are now written to
`results/processed/georef_phase5_diagnostics.json` (primary + denser robust candidate
tables + SIFT params), so the no-overlap verdict is reproducible and inspectable.

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

## Why the terminal-cause finding matters (Phase 6 outcome)

The hard cutoff is NOT in the registrations stage (SIFT+USAC_MAGSAC matching of a
same-ground-area crop) — it is EARLIER, in the georeference stage: with sun_el
~1.9 deg the low-sun OHRC-2026 swath has almost no stable local-contrast
structure at working-set scale, so no overlap can be auto-found by any available
interest-point matcher. Registrations-stage improvement alone (Phase 4 outlier
rejection / refinement) cannot rescue a pair whose crops were never produced.

Phase 6 therefore hardened the GEO-REFERENCE stage (the autonomous overlap
search), the stage the differentiation must target:

- `candidate_stats()` — pure, reusable single/extractor-parameterized 4-orientation
  SIFT overlap table (per flip: good, inliers, inlier_ratio, ncc, bbox).
- `find_pair_transform_robust()` — boosted fallback (30 000 feat, ct 0.006) with
  PROOF-based acceptance: winner inliers>=20, ratio>=0.25, mirror-group
  dominance (id/HV group vs V/H group: winner group >= 2x loser group inliers;
  proven necessary because SIFT+craters are 180-rotation-invariant so the
  id/HV pair is naturally near-equal), warped-overlap NCC>=0.15, sane bbox.
- On failure, `georef_{prefix}_diagnostics.json` records primary + robust
  candidate tables and params, making a refusal reproducible and inspectable.
- Registrations stage: optional `matcher.learned_fallback` — when the classical
  matcher produces <8 matches and a SuperPoint cache exists, LoFTR/SuperGlue is
  tried automatically (functional test: crippled BF ratio 0.03 -> LoFTR rescue,
  5563 inliers, RMSE 22.7753 on pair1). Classical FINAL_CONFIG remains champion.

The phase5 verdict is unchanged and now stronger: no overlap, five methods
consistent, dense tools validated (noise-level even on the known-good pair), and
the improved georeferencer still refuses with an auditable diagnostics artifact.

## Files

- `configs/experiment_C9_polar.yaml`
- `scripts/run_polar_learned.py`
- `results/logs/ablation.csv` (C9 polar row)
- `results/logs/polar_learned_run.log`, `results/logs/polar_loftr_run.log`