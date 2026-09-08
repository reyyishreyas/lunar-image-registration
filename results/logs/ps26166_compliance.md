# PS 26166 — Alignment of the delivered solution

Problem: *Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

## Requirement → implementation map

| PS 26166 requirement | Where implemented | Evidence |
|---|---|---|
| **Multi-modal**: OHRC, TMC-2, IIRS vs lunar reference | `src/detection/cross_modal.py` (radiometric-invariant front-ends), `src/preprocessing/ch2_staging.py` (TMC/OHRC ground grid + `register_ch2_to_nac` for TMC→NAC), `src/detection/iirs.py` (`register_iirs_to_ohrc` + `register_iirs_to_nac`) | `results/logs/phase9_summary.md`; 5 sensor presets in `demo/app.py` |
| **Reference images: LRO NAC** | `src/preprocessing/geometry.py` (NacInverse, SPICE lon/lat→sample/line), `src/preprocessing/georeference.py` (rasterio reader), pair-1/pair-2/TMC→NAC/IIRS→NAC flows | TMC→NAC: geometry_registered at 21.70 m/px over real overlap [296.027,296.273]×[6.361,8.264] (30 s); IIRS→NAC: honest not_registered (IR vs visible content mismatch, 6 s) |
| **Sun-illumination invariance** | `src/evaluation/sun_angle.py` (synthetic Lambertian relight across sun-azimuth/elevation grid + `sun_angle_sweep`); geometry fallback for pairs whose content is not verifiable | Sweep on pair-1: self-RMSE + `delta_H_px` illumination-drift metric, all configs find same content match; TMC↔NAC→honest `geometry_registered` (frame disagreement, not darkness) |
| **Scale invariance** | equal-GSD staging (`stage_ch2_ground_pair`, `_gsd_m`, `_sample_subgrid` with `scipy.map_coordinates`); TMC 21.70 m/px ↔ NAC resampled to same GSD | GSD equalisation reported on every TMC→NAC run |
| **Generic software for correspondence** | `scripts/run_auto.py` CLI (`--sensor ohrc-nac\|tmc-ohrc\|iirs-ohrc\|tmc-nac\|iirs-nac`), `run_sensor_auto` dispatcher, Streamlit app | All 5 sensors wired; all real-data runs exit 0 |
| **Sub-pixel accuracy of source image** | `src/refinement/subpixel.py`, `src/refinement/phase_correlation.py` | `results/logs/phase7_rmse_sub1.md`: held-out RMSE **0.5968 px**, end-to-end **0.6971 px** (OHRC↔NAC) |
| **Uniform distribution of matches** | `src/outlier_rejection/grid_uniform.py` `cap_by_grid` | Phase 4 C9 log |
| **Registered product** | staged common-grid crops (`_src.png/_ref.png`), after-surface red outlines (`_outline_registered`), Turbo change maps, montage | All 5 sensor paths produce artifacts |
| **Corresponding match points** | `{prefix}_inliers.csv` (src_x,src_y,ref_x,ref_y) saved by `_save_correspondences` for content-registered pairs; `run_auto` wires `artifacts.correspondences` | `demo/app` renders path + count |
| **Evaluation metric** | RMSE, inlier count, inlier ratio, n_matches, self-RMSE | metrics in app + `src/evaluation` |
| **User-verifiable change visualisations** | red-frame + tag on every image surface (`src/visuals.py`), Turbo difference maps, red-boxed overlap swaths in all app modes | app-wide, all modes |
| **Honest verdicts** | no fabricated models: `registered` (content match, sub-pixel RMSE), `geometry_registered` (SPICE + ISRO CSV common grid, content not verifiable), `not_registered` (content and geometry both fail) | all paths; TMC→NAC = geometry_registered; IIRS→NAC = not_registered |

## Honest limits (data / hardware, not code)

- **Content registration is only claimed where verifiable.** Pairs whose frames
  fail content matching return `geometry_registered` (TMC↔OHRC-2026, TMC↔NAC at
  native GSD 21.70 m/px) or `not_registered` (IIRS↔NAC: IR vs visible, no
  content overlap). No fabricated alignments.
- **TMC→NAC is not "dark"** — both crops are contrast-rich (low_contrast 0.0/0.0);
  its content path fails because the **ISRO TMC-2026 CD ground grid and the LRO
  NAC SPICE frame disagree by ~10 km** on this pair (overlap_NCC ≈ 0 after exact
  staging). `register_ch2_to_nac` records `overlap_ncc`/`overlap_px` and states
  this honestly; it does not blame illumination.
- **Staged TMC crops are what the ISRO CSV claims** — the phase-9b staging fix
  (`GroundGridInverse`, 2-D Newton inverse) removed a real bug that previously
  column-scrambled the TMC→NAC/SWATH source by up to ~1400 native columns
  (`_inverse_maps` separable model; pixel error up to 2715 px). After the fix
  the inverse round-trips the real grid exactly (0.000 px).
- **SELENE reference images** are supported conceptually (same OHRC/TMC pipeline
  reads any lunar raster + geometry) but no SELENE raster is present in the
  local `data/`; the reposited reference set is LRO NAC. Adding a SELENE pair is
  a data drop, not a code change.
- **GPU methods** (LightGlue SuperPoint/DISK/ALIKED, D2-Net) exist in notebook
  form; the pip-installed CPU environment runs the SIFT/cross-modal path.
