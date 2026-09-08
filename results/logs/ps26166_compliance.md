# PS 26166 — Alignment of the delivered solution

Problem: *Multi-modal, Sun angle and scale invariant image correspondence using
Chandrayaan-2 optical images (OHRC, TMC and IIRS).*

## Requirement → implementation map

| PS 26166 requirement | Where implemented | Evidence |
|---|---|---|
| **Multi-modal**: OHRC, TMC-2, IIRS vs lunar reference | `src/detection/cross_modal.py` (radiometric-invariant front-ends), `src/preprocessing/ch2_staging.py` (TMC/OHRC/IIRS ground grid), `src/detection/iirs.py` (IIRS ENVI reader) | `results/logs/phase9_summary.md`; sensor presets in `demo/app.py` |
| **Reference images: LRO NAC** | `src/preprocessing/geometry.py` (SPICE + ISRO CSV), pair-1/pair-2 flows | RMSE vs `data/ground_truth/pair1_gt_v2.csv` |
| **Sun-illumination invariance** | dark/low-sun handling: `enhance_dark`, `low_contrast_score`; geometry fallback when content is not verifiable | TMC↔OHRC-2026: geometry_registered at 21.70 m/px; OHRC-2026 mean ~5/255 |
| **Scale invariance** | equal-GSD staging (`stage_ch2_ground_pair`, `_gsd_m`), self-calibrated NAC GSD factor | GSD equalisation reported on every run |
| **Generic software for correspondence** | `scripts/run_auto.py` / `run_sensor_auto` / `run_auto` end-to-end CLI + Streamlit | CLI exit 0 on all three pairs |
| **Sub-pixel accuracy of source image** | `src/refinement/subpixel.py`, `src/refinement/phase_correlation.py` | `results/logs/phase7_rmse_sub1.md`: held-out RMSE **0.5968 px**, end-to-end **0.6971 px** (OHRC↔NAC) |
| **Uniform distribution of matches** | `src/outlier_rejection/grid_uniform.py` `cap_by_grid` | Phase 4 C9 log in `results/logs/` |
| **Registered product** | staged common-grid crops (`_src.png/_ref.png`), ortho/overlay artifacts, montage | rendered, framed "delivered product" in the app |
| **Corresponding match points** | inlier correspondence pairs saved to `{prefix}_inliers.csv` (`src/preprocessing/ch2_staging.py::_save_correspondences`, `src/auto_pipeline.py`) | `demo/app` shows count + path |
| **Evaluation metric** | RMSE, inlier count, inlier ratio, n_matches, self-RMSE | metrics in the app + `src/evaluation` |
| **User-verifiable change visualisations** | red-frame + tag on every image surface, red-boxed overlap swaths, Turbo difference maps (`src/visuals.py`) | app-wide, all modes |

## Honest limits (data / hardware, not code)

- **Content registration is only claimed where verifiable.** Dark/low-sun
  pairs (TMC↔OHRC-2026, polar pair-2) return `geometry_registered`; IIRS ships
  no geometry CSV so it is content-only and may legitimately return
  `not_registered`. No fabricated alignments.
- **SELENE reference images** are supported conceptually (same OHRC/TMC pipeline
  reads any lunar raster + geometry) but no SELENE raster is present in the
  local `data/`; the reposited reference set is LRO NAC. Adding a SELENE pair is
  a data drop, not a code change.
- **GPU methods** (LightGlue SuperPoint/DISK/ALIKED, D2-Net) exist in notebook
  form; the pip-installed CPU environment runs the SIFT/cross-modal path.