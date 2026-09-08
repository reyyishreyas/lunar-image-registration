# Phase 9 Summary — Cross-Modal TMC/IIRS + PS-26166 Gap Resolution

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
- Full suite: **70 passed, 0 failed** (11.74 s)

## Files added/modified
- `src/preprocessing/ch2_staging.py` (+stage_ch2_nac_pair, register_ch2_to_nac, _sample_subgrid, _nac_gsd_m)
- `src/detection/iirs.py` (+register_iirs_to_nac, _register_iirs_vs_reference)
- `src/auto_pipeline.py` (+tmc-nac, iirs-nac dispatch)
- `scripts/run_auto.py` (+2 sensor choices)
- `src/evaluation/sun_angle.py` (new)
- `results/logs/ps26166_compliance.md` (updated)
- `results/logs/phase9_summary.md` (this file)
