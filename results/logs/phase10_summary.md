# Phase 10 Summary — Match/No-Match Decision Engine (PS-26166 "matches or not")

## Objective

"Give it any images from OHRC / TMC / IIRS / NAC and it says MATCH or NO MATCH,
and produces the best final aligned product." This phase turns the five
register_* modules' honest verdicts into a single machine-readable decision the
user can act on, and demonstrates it on four *different* real sensor pairs.

## What was delivered

### 1. Decision layer — `src/evaluation/decision.py`
- `classify(report)` reduces any registration report to a definitive decision:
  - `matched: True`   cause=`content_correspondence`  (inliers + RMSE evidence)
  - `matched: False`  cause among:
    - `dark_or_featureless`  a staged crop is low-contrast / very dark
    - `cross_sensor_mismatch`  IIRS (IR) vs a visible reference
    - `frame_disagreement`  both crops texture-rich yet common-grid overlap NCC
      ≈ 0 -> the two geodetic frames genuinely disagree (ISRO CD vs NAC SPICE)
    - `error`  un-registerable / broken input, raw note preserved
- Every decision carries a human `explanation` plus a checkable `evidence` block
  (overlap_ncc, overlap_px, low-contrast, means, inliers, RMSEs) and a
  `best_product` path — always evidence-based, never fabricated.
- `best_product()` returns the best aligned artefact (aligned/diff → checkerboard
  → montage → overlay → src).

### 2. "Drop anything" CLI — `--sensor any`
- `detect_pair(src, ref)` infers each file's sensor from its name
  (`ch2_ohr*`=OHRC, `ch2_tmc*`=TMC, `ch2_iir*`=IIRS, `M\d{10}(RC|LC|LE).IMG`=NAC)
  and dispatches to the right register.
- New same-sensor / cross combos wired: `ohrc-ohrc`, `tmc-tmc`, `iirs-iirs`,
  `ohrc-tmc` (all through `register_ch2_pair` / IIRS body).

### 3. Honest evidence on the geometry route
- Phase-8 geometry fallback now computes overlap NCC + per-crop
  low-contrast/means on the staged pair (`src/auto_pipeline._geometry_fallback`),
  so `geometry_registered` never ships a blank explanation.

### 4. Best aligned product
- Content-registered pairs write `{prefix}_aligned.png` (source warped onto the
  reference frame) + `{prefix}_diff.png` (Turbo difference map), surfaced as
  `decision.best_product`.

## Demonstration — four different real per-sensor pairs

| Pair | Files | Decision | Key evidence |
|---|---|---|---|
| **OHRC 2021 ↔ NAC** | ch2_ohr_ncp_20210405T1606536730 + M1469248775LC | **MATCH** (content_correspondence) | inliers=123, self-RMSE 0.967 px, **RMSE vs GT 0.549 px**, aligned product written |
| **TMC 2026 ↔ NAC** | ch2_tmc_nca_20260607T2319176707 + M1127547939RC | **NO MATCH** (frame_disagreement) | overlap_ncc −0.0032 over 134,139 px, both crops rich (low_contrast 0.0/0.0) → ISRO CD vs NAC SPICE frames disagree ~10 km |
| **OHRC 2026 ↔ NAC** | ch2_ohr_ncp_20260331T1105235288 + M1127547939RC | **NO MATCH** (frame_disagreement) | overlap_ncc −0.0196 over 157,674 px; same conclusion on an independent OHRC over the SAME footprint |
| **IIRS 2021 ↔ NAC** | ch2_iir_nri_20211221T0324126144 + M1127547939RC | **NO MATCH** (cross_sensor_mismatch) | 4 bands (32/87/142/197) all fail; IR vs visible |

All four ran through `scripts/run_auto.py --sensor any` (auto-detected pair) with
the honest decision printed, and a full JSON report + artefacts written.

## Test results

- Full suite: **76 passed, 0 failed** (including new `tests/test_decision.py`:
  registered→MATCH, dark-crop→dark_or_featureless, frame_disagreement, iirs→
  cross_sensor_mismatch).

## Files created/modified
- `src/evaluation/decision.py` (new)
- `src/auto_pipeline.py` (`classify` wiring, `detect_pair`, geometry-route
  evidence, aligned-product writer, extra sensor combos, decision in `summarize`)
- `scripts/run_auto.py` (`--sensor any`, expanded choices)
- `tests/test_decision.py` (new, 4 tests)
- `results/logs/phase9_summary.md`, `results/logs/ps26166_compliance.md`
- `data/processed/auto/pair1_nac/`, `tmc_nac/`, `ohrc2026_nac/`, `iirs_nac/`
  (real-data verification runs)