# PS 26166 — Alignment of the delivered solution

Problem: *Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

## Requirement → implementation map

| PS 26166 requirement | Where implemented | Evidence |
|---|---|---|
| **Multi-modal**: OHRC, TMC-2, IIRS vs lunar reference | `src/detection/cross_modal.py` (radiometric-invariant front-ends), `src/preprocessing/ch2_staging.py` (TMC/OHRC ground grid + `register_ch2_to_nac` for TMC→NAC), `src/detection/iirs.py` (`register_iirs_to_ohrc` + `register_iirs_to_nac`) | `results/logs/phase9_summary.md`; 5 sensor presets in `demo/app.py` |
| **Reference images: LRO NAC** | `src/preprocessing/geometry.py` (NacInverse, SPICE lon/lat→sample/line), `src/preprocessing/georeference.py` (rasterio reader), pair-1/pair-2/TMC→NAC/IIRS→NAC flows | TMC→NAC: geometry_registered at 21.70 m/px over real overlap [296.027,296.273]×[6.361,8.264] (30 s); IIRS→NAC: honest not_registered (IR vs visible content mismatch, 6 s) |
| **Sun-illumination invariance** | `src/evaluation/sun_angle.py` (synthetic Lambertian relight across sun-azimuth/elevation grid + `sun_angle_sweep`); dark/low-sun geometry fallback for unverifiable pairs | Sweep on pair-1: 9/16 configs register, self-RMSE 2.1–2.5 px at 256 px resolution, all configs find same content match; dark pair TMC↔NAC→honest `geometry_registered` |
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

- **Content registration is only claimed where verifiable.** Dark/low-sun
  pairs (TMC↔OHRC-2026, TMC↔NAC) return `geometry_registered` at native GSD
  (21.70 m/px); IIRS↔NAC returns `not_registered` (IR vs visible, no content
  overlap). No fabricated alignments.
- **SELENE reference images** are supported conceptually (same OHRC/TMC pipeline
  reads any lunar raster + geometry) but no SELENE raster is present in the
  local `data/`; the reposited reference set is LRO NAC. Adding a SELENE pair is
  a data drop, not a code change.
- **GPU methods** (LightGlue SuperPoint/DISK/ALIKED, D2-Net) exist in notebook
  form; the pip-installed CPU environment runs the SIFT/cross-modal path.
