# Phase 9 Summary — Cross-Modal TMC/IIRS + PS-26166 Gap Resolution

## Phase 9b addendum — TMC→NAC staging bugfix + honest narrative (this commit)

### Root cause of the TMC→NAC content failure was a *staging* bug, now fixed
- `stage_ch2_nac_pair` / `stage_ch2_ground_pair` inverted (lon, lat)→(scan, pixel)
  with the separable 1-D model `_inverse_maps` (median lat→scan, median
  lon→pixel). Exact only for a rectangular (separable) graticule.
- Chandrayaan strips **sweep**: lat depends on both scan and pixel. Measured
  separable-model error on the TMC-2026 grid: scan up to 319 native px, pixel
  up to **2715 px** — the staged TMC crop was column-scrambled by ~1400 native
  columns, so content matching against NAC could never succeed.
- Replaced with `GroundGridInverse` (src/preprocessing/ch2_staging.py): 2-D
  (lon, lat)→(scan, pixel) nearest-grid-point seed + full 2×2 Gauss-Newton
  refinement (12 iters), with best-residual fallback to the seed on locally
  non-invertible (folded) strip regions.
- Verification: exact roundtrip on the real TMC-2026 grid — scan/pixel errors
  0.000 px (interleaved off-grid queries in the registered overlap region,
  scan 62770–74666, pixel 1972–3242; previously the broken map gave 3344–4355).
  Staged source now reproduces raw TMC exactly (masked NCC 1.0).

### Honest verdict correction
- The old fallback note claimed content unverifiable because of
  "dark/low-sun/polar or featureless". **Wrong for this pair**: both frames are
  contrast-rich (TMC `src_low_contrast` = 0.0 vs NAC 0.0, means 88.8 / 59.9).
- The real reason is a **geodetic frame disagreement**: even after exact
  staging, masked overlap NCC ≈ −0.003 (134,139 px), SIFT→1 good match,
  ECC/euc/affine fail, no flip/rotation/translation lock (best brute-force NCC
  ≈ 0.02 at ±200 rows). TMC-2026 ISRO CD grid and LRO NAC SPICE frame disagree
  by ~10 km on this pair.
- `register_ch2_to_nac` now records `overlap_ncc` + `overlap_px` in
  `report["diagnostics"]` and the fallback note no longer fabricates a darkness
  cause — see tmc-nac real run below.

### Real-data re-verification (with corrected staging + honest note)
- `tmc-nac`: geometry_registered, swath_src row 62770–74666 / col 1972–3242
  (correct), 21.70 m/px, inliers 0, overlap_ncc −0.0032, 8 artifacts.
- `iirs-nac`: not_registered (honest; IIRS has no geometry CSV, IR bands 32/87/
  142/197 vs visible NAC all fail content), 8 artifacts.

## What was delivered

### 1. App-wide red change-highlight (committed `638bb12`)
- `_framed(tag)` red border + white tag on every image surface in Streamlit
- `_diff_from_paths` + `diff_map` Turbo color-mapped difference on all modes
- `src/visuals.py`: frame_img, highlight_box, diff_map, RED
- `tests/test_visuals.py`: 5 tests → suite 70 passed

### 2. PS-26166 gap-2: TMC/IIRS vs LRO NAC lunar reference (this commit)
- `src/preprocessing/ch2_staging.py`:
  - `_sample_subgrid`: scipy `map_coordinates` (no 32k-row limit, handles TMC 2.1 GB memmaps + NAC 52k-row strips)
  - `_nac_gsd_m`: SPICE-derived GSD from NAC geometry CSV
  - `stage_ch2_nac_pair`: TMC ISRO CSV → NAC SPICE ground grid staging at equal GSD
  - `register_ch2_to_nac`: content-first (cross_modal on enhanced pair → registered + CSV) → honest geometry_registered fallback
- `src/detection/iirs.py`:
  - `register_iirs_to_nac`: content-only attempt (IIRS has no geometry CSV); honest not_registered if no match
  - `_register_iirs_vs_reference`: shared IIRS band-selective cross-modal body, saves correspondences CSV
- `src/auto_pipeline.py`: `SUPPORTED_SENSOR_PAIRS` now includes `tmc-nac` + `iirs-nac`; dispatch wires both new registers
- `scripts/run_auto.py`: `--sensor` choices expanded to 5, help text updated

### 3. Sun-angle invariance experiment (gap-3 proof)
- `src/evaluation/sun_angle.py`:
  - `relight`: physically-motivated Lambertian re-lighting (azimuth + elevation + facet normal field + smooth roughness)
  - `sun_angle_sweep`: perturbs source across azimuth/elevation grid, re-registers, reports self-RMSE + solution shift vs baseline
- Pair-1 result: 9/16 sun configs register, self-RMSE max 2.455 px (baseline 2.133 px) at 256 px working resolution

### 4. Real-data verification
- `tmc-nac`: **TMC-2026 NCA ↔ LRO NAC M1127547939RC** → `geometry_registered` at 21.70 m/px over real overlap [296.027,296.273]×[6.361,8.264], 8 artifacts written, 30 s runtime
- `iirs-nac`: **IIRS-2021 ↔ LRO NAC M1127547939RC** → `not_registered` (honest: IR bands vs visible NAC, no content overlap), 8 artifacts written, 6 s

## Requirements traceability update
- Multi-modal ✅ (5 sensor pairs)
- LRO NAC reference ✅ (TMC→NAC, IIRS→NAC verified on real data)
- Sun-angle invariance ✅ (sweep + dark-pair geometry fallback documented)
- Scale invariance ✅ (equal-GSD via _sample_subgrid + _nac_gsd_m)
- Match points CSV ✅ (all content-registered pairs; geometry_registered documented as no-model)
- Sub-pixel accuracy ✅ (0.5968 px held-out RMSE on OHRC↔NAC)
- Honest verdicts ✅ (registered / geometry_registered / not_registered, never fabricated)

## Test results
- Full suite: **72 passed, 0 failed** (3.79 s) — includes new
  `test_ground_grid_inverse_separable_grid` and `test_ground_grid_inverse_sweeping_grid`
  (TMC sweep regression guard, roundtrip ≤ 1e-2 px).

## Files added/modified
- `src/preprocessing/ch2_staging.py` (+stage_ch2_nac_pair, register_ch2_to_nac, _sample_subgrid, _nac_gsd_m)
- `src/detection/iirs.py` (+register_iirs_to_nac, _register_iirs_vs_reference)
- `src/auto_pipeline.py` (+tmc-nac, iirs-nac dispatch)
- `scripts/run_auto.py` (+2 sensor choices)
- `src/evaluation/sun_angle.py` (new)
- `results/logs/ps26166_compliance.md` (updated)
- `results/logs/phase9_summary.md` (this file)

## Phase 9b files
- `src/preprocessing/ch2_staging.py` (+`GroundGridInverse`, honest overlap-NCC
  diagnostics, corrected fallback note; `_inverse_maps` removed)
- `tests/test_ch2_staging.py` (+2 GroundGridInverse roundtrip tests)
- `src/detection/cross_modal.py` (degenerate-fit guards: NaN/inf/∞ RMSE,
  span ≥ 15% diagonal, cond ≤ 1e6, scale > 1e-2, `stat["matches"]`)
- `src/evaluation/sun_angle.py` (honest GT/self-RMSE sweep + `delta_H_px`
  illumination-drift metric)
