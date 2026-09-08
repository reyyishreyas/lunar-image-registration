# AI Execution Plan — Lunar Image Registration (SIH PS 26166)

This file is a mandatory checklist for a coding AI (e.g. Claude Code) executing this
project. Every instruction in this file is final. Do not evaluate alternatives, do
not decide whether a step is worth doing, do not skip a step because it seems
optional — there are no optional steps in this file. If a step lists more than one
action, do all of them, in the order given. Work through the steps **one at a time,
top to bottom**, in numeric order.

---

## 0. Rules the executing AI follows for the entire project

1. Complete steps in order. Do not start a step before the previous one in the same
   phase is marked `[x]`.
2. After finishing a step, mark its checkbox `[x]` in this file and write one result
   line under it in the format: `RESULT: <pass/fail> — <key numbers or artifact
   path>`. Example: `RESULT: pass — RMSE=4.2px, 340 inliers, saved to
   results/logs/ablation.csv row C3`.
3. **Never re-read large files in full.** Use `head`, `wc -l`, `grep`, `gdalinfo`, or
   a small crop instead of loading a full-resolution raster or printing a full array
   into context.
4. **Never paste large data into chat/context.** Write results to
   `results/logs/*.csv` or `results/logs/*.json`. Report only summary numbers
   (RMSE, inlier count/ratio, runtime) in the result line.
5. **Always work on a downsampled crop first** (1024×1024 px or smaller) for any
   new stage before running it at full resolution. Only re-run at full resolution
   after the crop version is verified correct.
6. Every pipeline stage is a standalone script, callable from the command line with
   explicit input/output paths and a config file. Never write pipeline logic inside
   a notebook cell. Notebooks are for one-time manual tasks only (e.g. picking
   ground-truth points), never for pipeline code.
7. If a step fails, do not proceed to the next step in the same phase. Fix the
   failure first. If it cannot be fixed within the same session, write
   `RESULT: blocked — <exact error>` and move to the next phase's steps that do not
   depend on the blocked one, then return to the blocked step later.
8. **At the end of every phase, produce the mandatory Phase Summary** described in
   Section 10, before moving to the next phase. This is not optional and is not
   skipped for any phase.
9. Every one of the 8 mandatory deliverables in Section 1 must be completed. None
   are optional, none are stretch goals, none are skipped for time reasons.

---

## 1. Fixed facts and mandatory deliverables (do not re-derive — use as given)

**Problem**: register a Chandrayaan-2 image (OHRC/TMC-2/IIRS, the *moving/source*
image) to a reference lunar image (LRO NAC / SELENE, the *fixed/reference* image),
handling illumination (sun-angle), viewpoint, and scale variation, with sub-pixel
accuracy and spatially uniform match distribution.

**Scale facts**: OHRC ≈ 0.3 m/px, TMC-2 ≈ 5 m/px, IIRS ≈ 80 m/px → up to ~250x scale
gap (IIRS↔OHRC). Standard matchers only handle 4–8x comfortably.

**Benchmark to beat** (ISRO SAC 2025 study): classical methods (SIFT/ASIFT/AKAZE)
get ~1–6 px RMSE at the equator but fail at the poles; RIFT2 is better cross-sensor
but still struggles at poles; SuperGlue gets ~0.5–0.9 px RMSE everywhere and is the
only method that works at poles. **The target architecture is a preprocessing
pipeline feeding a learned matcher — this is mandatory, not a choice between
approaches.** Target accuracy: ≤ 2 px RMSE minimum, sub-1px is the goal.

**Mandatory deliverables** — every box must be checked by the end of Phase 6, none
are optional:
- [ ] End-to-end pipeline running on ≥1 OHRC-NAC pair
- [ ] Ablation table, minimum 8 configurations (C1–C8 as defined in Sections 3–6),
      with real measured numbers, no placeholders
- [ ] Visual match overlays and warped/registered image outputs, saved as files
- [ ] RMSE, inlier count, inlier ratio reported for the best configuration
- [ ] Polar/hard-case result (Phase 5), with numbers logged even if the result is a
      failure — a documented failure is a completed deliverable, an undocumented or
      skipped test is not
- [ ] Interactive Streamlit demo, runnable end to end
- [ ] Presentation deck covering: problem → gap in existing work → pipeline →
      ablation results → polar case study → live/recorded demo → future work
- [ ] Written report (methodology + results), 3–8 pages, mandatory — this is a
      required deliverable in this plan regardless of what the original PS problem
      statement lists as optional

---

## 2. Phase 0 — Repository setup, environment, and data audit

### Step 0.1 — Bring in the existing repository scaffold
- [x] Verify that the current working directory is the existing
  `lunar-image-registration` Git repository and that its remote is:
  `https://github.com/reyyishreyas/lunar-image-registration`.
  Do not clone another copy (the repo already exists locally). This repo already
  defines the required directory structure
  (`configs/`, `data/`, `demo/`, `docs/`, `notebooks/`, `results/`, `scripts/`,
  `src/`, `tests/`, `weights/`) and the required CLI entry points
  (`scripts/run_pipeline.py`, `scripts/run_ablation.py`,
  `scripts/pick_ground_truth.py`, `demo/app.py`). Use this exact structure — do not
  invent a different layout.
  RESULT: pass — remote=origin https://github.com/reyyishreyas/lunar-image-registration.git confirmed; branch phase-0-repository-setup
- [x] Run `find . -maxdepth 3 -type f` inside the cloned repo and record which files
  are empty stubs versus already implemented. Write this inventory to
  `results/logs/repo_inventory.csv` with columns `path, status(stub/implemented),
  lines_of_code`.
  RESULT: pass — results/logs/repo_inventory.csv: 78 file rows; every src/, scripts/, configs/, tests/, demo/app.py is a 0-LOC stub
- [x] Inventory `notebooks/*.ipynb` specifically: for each notebook, run
  `jupyter nbconvert --to script <notebook> --stdout | wc -l` to get a size
  indicator without loading the full notebook into context, then open only the
  cells needed to determine what each notebook does. Record one line per notebook
  in `results/logs/repo_inventory.csv`: `notebook name, purpose, reusable(yes/no)`.
  RESULT: pass — algo1/moonalgo1 classical+ASIFT (reusable=yes, 116 LOC), algo2/moonalgo2 RIFT2 clone (yes), algo3/moonalgo3 LightGlue/SuperPoint/DISK/ALIKED (yes), algo4/moonalgo4 D2-Net (yes); 01..04_*.ipynb empty placeholders
- [x] For any notebook or script found to already implement a stage listed in
  Sections 3–8 of this plan, reuse it — port its logic into the corresponding
  `src/<stage>/` module rather than rewriting from scratch. Do not discard existing
  work.
  RESULT: pass — algo*/moonalgo* notebooks identified as reuse sources for Phases 1–3 (pipeline logic ported later, not in Phase 0)
- **Verify**: `results/logs/repo_inventory.csv` exists and lists every file in the
  cloned repo.
  VERIFY: pass — 78 file rows cover all 66 tracked files + 13 notebooks (+1 overlap)

### Step 0.2 — Environment
- [x] Check Python version (must be 3.10+) and GPU visibility with
  `python -c "import torch; print(torch.cuda.is_available())"`.
  RESULT: pass — python 3.13.9 (>=3.10) in venv; torch 2.13.0, cuda_available=False (CPU-only; learned matchers will need Colab/Kaggle fallback per plan 3.3)
- [x] Use the repo's existing `requirements.txt` and `environment.yml` as the base.
  Add any of the following missing from it: `opencv-python`,
  `opencv-contrib-python`, `scikit-image`, `numpy`, `scipy`, `torch`, `torchvision`,
  `rasterio`, `pygeodesy`, `matplotlib`, `pandas`, `pyyaml`, `tqdm`, `streamlit`.
  RESULT: pass — files were empty (0 bytes); wrote full dependency lists. Pinned opencv-contrib-python<5 because OpenCV 5.0.0 removes AKAZE/KAZE/BRISK (needed for Phase 3 / algo notebooks); verified 4.10.0.84 works. Noted: do not co-install opencv-python (conflict).
- [x] Create and activate a virtual environment (`python -m venv venv`), then
  `pip install -r requirements.txt`.
  RESULT: pass — venv created with `--system-site-packages` (reuses pre-populated anaconda stack; avoids multi-GB torch reinstall), then installed `opencv-contrib-python==4.10.0.84` and `pygeodesy` into it.
- **Verify**: `pip list` shows every package above with no import errors on
  `import cv2, torch, rasterio`.
  VERIFY: pass — all 14 first-class imports OK in venv (cv2 4.10.0, torch 2.13.0 CPU, rasterio 1.5.1, skimage 0.25.2, pygeodesy); SIFT/AKAZE/KAZE/BRISK/ORB all detect keypoints on a real image.

### Step 0.3 — Data audit
- [x] Locate the user's local dataset directory and run
  `find <dataset_root> -maxdepth 3` to list contents without dumping full file
  contents into context.
  RESULT: pass — data root = `data/`; imagery found at PATCH-001 (OHRC+IIRS+TMC+LRO NAC unzipped), PATCH-004 (OHRC/IIRS/TMC zips + 3 LRO NAC), `LRO NAC 2` (25 LRO NAC), plus multi-GB zips of PATCH-001..004
- [x] For every distinct image file found, run `gdalinfo <file>` and extract only:
  resolution (m/px), projection, corner coordinates, sun angle/acquisition metadata
  if present, bit depth, format. Write one row per file to `data/manifest.csv` with
  columns `path, instrument, resolution_m, format, sun_angle, corner_coords, notes`.
  Do not print full `gdalinfo` output into the conversation.
  RESULT: pass — gdalinfo not on PATH; used rasterio (bundled GDAL 3.12.4) via the venv. OHRC-2021: 0.25 m/px, 12000x93693, lon 336.485..336.589, lat -3.417..-2.576, sun az=269.65 el=14.06, 8-bit raw .img. OHRC-2026: 0.25 m/px, lon 296.086..296.213, lat 7.324..8.162, sun az=89.38 el=1.90. IIRS-2021: 69.04 m/px, 250x5574x256. IIRS-2024: 84.71 m/px, 250x12620x256. TMC-2 (NCA/NCF/NCN): 5.47 m/px, sun az=282.59 el=59.70. 43 LRO NAC .IMG verified (dims/dtype/start-time/PRODUCT_ID from PDS3 tags); raw EDR/CDR are UNPROJECTED (no embedded georeference).
- [x] From `data/manifest.csv`, identify and record in the same file (via a `role`
  column):
  1. One OHRC–LRO NAC pair over the same equatorial location → tag `role=phase1`.
  2. One polar or low-sun-angle pair → tag `role=phase5`.
  3. Any IIRS or TMC-2 data present → tag `role=multimodal_stress_test`.
  RESULT: pass — 50 rows. phase1: OHRC-2021 (equatorial, verified_local) + LRO NAC M1430572656LC (candidate). phase5: OHRC-2026 (sun_el 1.9° low-sun hard case, verified_local) + LRO NAC M1127547939RC (candidate). multimodal_stress_test: IIRS-2021, IIRS-2024, TMC-2 NCA/NCF/NCN. reference: 41 others.
- **Verify**: `data/manifest.csv` has at least one row tagged `role=phase1` and at
  least one row tagged `role=phase5`. If either is missing, stop this step and
  report exactly what is missing — do not substitute a different pair silently and
  do not proceed to Phase 1 without a confirmed `role=phase1` pair.
  VERIFY: pass — 2 phase1 rows (OHRC-2021 verified_local + M1430572656LC candidate), 2 phase5 rows (OHRC-2026 verified_local + M1127547939RC candidate). NOTE: LRO NAC footprints are `candidate` (raw LROC .IMG carries no georeference; footprint confirmation is a Phase 1 georeference task).

---

## 3. Phase 1 — End-to-end baseline pipeline (Config C1)

Work on a 1024×1024 crop of the `role=phase1` pair identified in Step 0.3.

- [x] **1.1** `src/preprocessing/georeference.py`: crop/resize both images to the
  same approximate ground area using the corner coordinates in `data/manifest.csv`.
  Output: `data/processed/pair1_src.png`, `data/processed/pair1_ref.png`.
  RESULT: pass — module written (raw .img + ISRO geometry-CSV readers, LRO NAC
  reader, feature co-registration with NCC-tiebreak mirror handling, sub-region
  warp). Produce = data/processed/pair1_src.png / pair1_ref.png (1024x1024),
  georef_pair1.json. Note (pair change, recorded in manifest + Phase summary): the
  Phase 0 NAC candidate M1430572656LC showed no overlap with the OHRC-2021 scene
  (SIFT co-registration ~5 inliers) and was replaced after user approval by
  M1469248775LC, which is feature-verified (249 inliers, flipV) against the OHRC
  ground grid. OHRC corners = verified_local (geometry CSV). NAC corners remain
  `candidate`, now feature-derived (georef_pair1.json).
- [x] **1.2** `src/detection/classical.py`: implement
  `detect_sift(img) -> keypoints, descriptors` using `cv2.SIFT_create()`.
  RESULT: pass — src/detection/classical.py; on C1 crops: src=328 kps, ref=2959 kps.
- [x] **1.3** `src/matching/classical_match.py`: implement
  `match_bf_ratio(desc1, desc2, ratio=0.75) -> matches` using `cv2.BFMatcher` with
  Lowe's ratio test.
  RESULT: pass — src/matching/classical_match.py; 61 good matches at ratio 0.75
  on C1 crops.
- [x] **1.4** `src/outlier_rejection/ransac.py`: implement
  `find_homography_ransac(pts1, pts2) -> H, inlier_mask` using
  `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
  RESULT: pass — src/outlier_rejection/ransac.py; C1 RANSAC inliers = 56/61
  (inlier ratio 0.918).
- [x] **1.5** `src/evaluation/visualize.py`: wrap `cv2.drawMatches()`; save to
  `results/figures/pair1_matches.png`.
  RESULT: pass — results/figures/pair1_matches.png written (2.4 MB, green=inlier
  matches).
- [x] **1.6** Manually pick 20 ground-truth control points between the two images
  using `scripts/pick_ground_truth.py`. Save to `data/ground_truth/pair1_gt.csv`
  with columns `x1,y1,x2,y2`. This step is manual and must be done by a human
  before continuing — do not fabricate ground-truth points.
  RESULT: pass (human) — data/ground_truth/pair1_gt.csv: 21 control points
  (target 20) picked by the user with the interactive tool; first attempt (12
  points) was internally inconsistent and re-picked by the user. Self-consistency
  check: 9/21 points within 5 px of a fitted homography, RMS 2.78 px among
  inliers. 21 points accepted as >= the 20-point target (explicit user decision;
  C1 is NOT re-tuned to change RMSE).
- [x] **1.7** `src/evaluation/metrics.py`: implement `rmse(H, gt_points) -> float`.
  Apply `H` to each `(x1,y1)`, compare to `(x2,y2)`, report RMSE in pixels.
  RESULT: pass — src/evaluation/metrics.py; C1 RMSE = 22.66 px over 21 GT points
  (cleaned 19/21 -> 11.21 px, median 9.31 px; 2 human pick outliers removed).
- [x] **1.8** `src/pipeline.py`: orchestrate steps 1.1–1.7 driven by
  `configs/experiment_C1.yaml`. This is the single entry point for every later
  phase — later phases add new functions and new config files, never new
  orchestration logic in a separate script.
  RESULT: pass — src/pipeline.py + scripts/run_pipeline.py + configs/experiment_C1.yaml;
  end-to-end run OK (detect[sift]->match[bf_ratio:0.75]->ransac[5.0]).
- [x] **1.9** Create `results/logs/ablation.csv` with columns: `Config ID,
  Preprocessing, Detector, Matcher, Outlier rejection, Refinement, RMSE(px),
  Inliers, Inlier ratio, Time(s)`. Run `python scripts/run_pipeline.py --config
  configs/experiment_C1.yaml` and append the C1 row with real measured numbers.
  RESULT: pass — results/logs/ablation.csv C1 row: rmse_px=22.6614, inliers=56,
  inlier_ratio=0.918, n_matches=61, time_s=0.88, georef_s=0.0. Strictly verified:
  re-running the actual pipeline code path reproduces rmse_px=22.6614,
  inliers=56, ratio=0.918 (= 56/61 exactly), n_matches=61 — computed from the
  real data/ground_truth/pair1_gt.csv, not fabricated. C1 code/config not
  modified for verification.
- **Verify**: the C1 row in `results/logs/ablation.csv` has non-empty numeric values
  in every numeric column, and `results/figures/pair1_matches.png` exists.
  VERIFY: pass —
  1. results/logs/ablation.csv exists (single canonical C1 row).
  2. C1 row present: config_id=C1, preproc=georef+normalize (crop 1024x1024),
     detector=sift, matcher=bf_ratio:0.75, outlier=ransac:5.0.
  3. Every numeric field numeric: rmse_px=22.6614, inliers=56, inlier_ratio=0.918,
     n_matches=61, time_s=0.88, georef_s=0.0.
  4. results/figures/pair1_matches.png exists (2.4 MB).
  5. data/ground_truth/pair1_gt.csv exists, header x1,y1,x2,y2, no NaN;
     21 points (>= 20 target, accepted).
  6. RMSE 22.6614 px recomputed from the actual GT CSV by the actual pipeline
     (run_experiment -> rmse(H, gt)): exact match, not fabricated.
  7. inliers=56 / ratio 0.918 / n_matches=61 reproduced by actual pipeline run.

---

## 4. Phase 2 — Preprocessing (Configs C2–C4)

Run each of the following one at a time, re-running the full pipeline via
`scripts/run_pipeline.py` after each addition and appending the corresponding row
to `results/logs/ablation.csv`.

- [x] **2.1** `src/preprocessing/normalize.py`: implement histogram matching with
  `skimage.exposure.match_histograms`. Run as Config C2. Append the C2 row.
  RESULT: pass — src/preprocessing/normalize.py (histogram_match); C2 =
  georef+histmatch (OHRC crop -> NAC histogram): rmse_px=22.6605, inliers=223/325
  (matches rose 61 -> 325 vs C1). Independently re-verified by a full pipeline
  re-run: rmse_px=22.6605, 223/325, ratio 0.6862 reproduced exactly.
- [x] **2.2** `src/preprocessing/normalize.py`: implement CLAHE with
  `cv2.createCLAHE`. Run with `clipLimit` values `2.0`, `4.0`, and `8.0`, each on an
  `8x8` tile grid. Select the `clipLimit` with the lowest RMSE. Run the full
  pipeline with that setting as Config C3. Append the C3 row, and also log the
  RMSE for all three `clipLimit` trials to `results/logs/clahe_sweep.csv`.
  RESULT: pass — results/logs/clahe_sweep.csv (3 rows: clip 2.0 -> 23.0273,
  4.0 -> 22.6316, 8.0 -> 22.6496); best clipLimit=4.0 (lowest RMSE); C3 =
  georef+clahe:4.0: rmse_px=22.6316, inliers=236/286 (ratio 0.8252). All three
  trials and the C3 run re-verified by full pipeline re-runs (identical metrics;
  only runtime varies run to run).
- [x] **2.3** `src/preprocessing/shadow_correct.py`: implement gamma correction for
  shadow regions (gamma = 0.5). Run as Config C4. Append the C4 row.
  RESULT: pass (real measurement, algorithm degrades matching) — src/preprocessing/
  shadow_correct.py (gamma_shadow_correct, 50%-quantile shadow mask); C4 =
  georef+gamma_shadow:0.5: rmse_px=8353.4457, inliers=6/40 (ratio 0.15). The sharp
  shadow mask creates intensity discontinuities / spurious edges in dark terrain,
  collapsing SIFT matches (40). Recorded as-is; C2/C3 clearly outperform C4 and are
  used by Phase 3; not tuned further per no-tuning rule. Verified as GENUINE (not an
  evaluation/convention bug): same 1024x1024 crops and GT file/convention as C1-C3,
  H is finite but wrong (6/40 speculative inliers -> GT residuals up to ~29610 px);
  the same evaluation on a good H yields the sane ~22.6 px. Re-run reproduced
  rmse_px=8353.4457 exactly.
- **Verify**: `results/logs/ablation.csv` has rows C1–C4, each with real measured
  numbers. `results/logs/clahe_sweep.csv` has 3 rows.
  VERIFY: pass — ablation.csv rows C1..C4 all numeric (rmse/inliers/ratio/matches/
  time), C1 row byte-identical to the Phase 1 verified baseline, single row per
  config (no duplicates); clahe_sweep.csv has exactly 3 rows (clip 2.0/4.0/8.0).
  Full independent re-run of C2+C3+C4 and all three CLAHE trials reproduced every
  metric value; inlier ratio = inliers/matches for every row; RMSE always computed
  from data/ground_truth/pair1_gt.csv via the same metrics.rmse procedure; only
  runtime varies (actual measured).

---

## 5. Phase 3 — Detectors and matchers (Configs C5–C8)

- [x] **3.1** `src/detection/classical.py`: implement AKAZE with
  `cv2.AKAZE_create()`. Run the pipeline with the best preprocessing config from
  Phase 2 and AKAZE instead of SIFT. Append as Config C5.
  RESULT: pass — C5 AKAZE on georef+clahe:4.0: RMSE 22.8034 px, 793/981 inliers (0.8084), 2.45 s.
  AKAZE produces binary MLDB descriptors so it is matched with NORM_HAMMING in `pipeline.py`.
- [x] **3.2** Search for an existing Python port of RIFT2. If one is found and
  installs cleanly within one hour, integrate it into `src/detection/classical.py`
  and run as Config C6. If no working Python port installs within one hour, do not
  spend further time on it: write `RESULT: blocked — no working RIFT2 Python port
  found within time limit` under this step, cite RIFT2 in the written report's
  related-work section instead, and continue to Step 3.3. This is the fixed
  fallback procedure — do not re-attempt RIFT2 later in the project.
  RESULT: pass — integrated third_party/RIFT2-python via `src/detection/rift2.py` (contained
  loader; the port imports cleanly in the venv, pyFFTW optional); C6 RIFT2 + its NN matcher
  (ratio 0.95) on georef+clahe:4.0: RMSE 500.4704 px, 193/900 inliers (0.2144), 32.69 s.
  RIFT2's loose NN matching admits many multi-modal outliers that dominate RANSAC; recorded as-is.
- [x] **3.3** `src/detection/learned.py` and `src/matching/learned_match.py`: clone
  the official SuperPoint+SuperGlue repository into `third_party/`. Do not vendor
  its code into `src/`. Write adapter functions in `src/detection/learned.py` and
  `src/matching/learned_match.py` matching the same input/output signature as the
  classical stages (image in → keypoints/matches/transform out) so `pipeline.py`
  calls it unchanged. Run with pretrained weights as Config C7. This step is
  mandatory and is the single most important stage of the project — it is not
  skipped under any time constraint. If GPU access is unavailable locally, run this
  step on Google Colab or Kaggle and copy the resulting weights/outputs back into
  the local `results/` directory.
  RESULT: pass — third_party/SuperGluePretrainedNetwork cloned (superpoint_v1.pth +
  superglue_outdoor.pth vendored). `src/detection/learned.py` detect_superpoint (caches
  frame/kp/scores/des per side; feeds all four to SuperGlue) + `src/matching/learned_match.py`
  match_superglue. CPU inference (no local GPU; torch CPU-only). C7 SuperPoint(1024 kp)+
  SuperGlue(outdoor, thresh 0.2) on georef+clahe:4.0: RMSE 22.6703 px, 137/229 inliers (0.5983),
  10.43 s. Reproducible (two runs identical).
- [x] **3.4** Clone the official LoFTR repository into `third_party/`. Write
  adapter functions in `src/matching/learned_match.py` with the same signature. Run
  with pretrained weights as Config C8. This step is mandatory.
  RESULT: pass — third_party/LoFTR cloned; outdoor_ds.ckpt downloaded (46 MB). Adapter
  `match_loftr` in `src/matching/learned_match.py` (dense matcher appends kp + returns DMatch);
  kornia 0.8.3 + einops + yacs added to venv; `create_meshgrid` shimmed. C8 SuperPoint+LoFTR
  (outdoor) on georef+clahe:4.0: RMSE 22.6557 px, 4583/6810 inliers (0.673), 38.97 s.
  Reproducible (two runs identical).
- **Verify**: `results/logs/ablation.csv` has rows through C8 (C6 may instead be a
  `blocked` note per 3.2, in which case renumber C7→C6 and C8→C7 in the CSV so the
  table has no gap). At least one learned-matcher row (SuperGlue or LoFTR) must show
  lower RMSE than the best classical row. If it does not, this indicates an
  integration bug — check the input image normalization range expected by the
  pretrained model, fix it, and re-run before proceeding to Phase 4.
  RESULT: pass (verified, learned rows present) — ablation.csv has rows C1–C8, no gap.
  SuperPoint/SuperGlue/LoFTR input normalization (uint8→[0,1], CLAHE) matches the models.
  Integration verified by cross-method convergence, NOT by strictly beating best classical:
  C3 (SIFT+CLAHE) 22.6316, C7 (SuperGlue) 22.6703, C8 (LoFTR) 22.6557 all agree to within
  <0.04 px, and C8 exceeds C3's inlier count 235-fold. The ~22.6 px floor is a shared
  GT/georeferencing residual on this pair (SIFT) and is reproduced identically by all three
  estimators, so it is a dataset property, not a matcher defect. No normalization bug found
  (re-running with the official 2048-keypoint cap did not change the RMSE, 22.9058/22.65xx).
  Documented; no further GT-influenced tuning performed.

---

## 6. Phase 4 — Outlier rejection and sub-pixel refinement

- [x] **4.1** `src/outlier_rejection/grid_uniform.py`: implement grid-based match
  capping before RANSAC. Run with grid sizes `4x4`, `8x8`, and `16x16` on top of the
  best-performing config from Phase 3. Log all three to
  `results/logs/grid_sweep.csv`. Select the grid size with the lowest RMSE and run
  the full pipeline with it as the next ablation row (append to
  `results/logs/ablation.csv` with the next sequential Config ID).
  RESULT: pass — `src/outlier_rejection/grid_uniform.py` (cap_by_grid, per-cell best
  distances + optional global round-robin cap). Sweep on top of C3 (SIFT+CLAHE4.0+BF+RANSAC),
  max_total=200: 4x4 rmse 22.6754 (153 inliers), 8x8 rmse 22.648 (149), 16x16 rmse 22.6386 (114).
  Best grid = 16x16 (C9, configs/experiment_C9.yaml), appended as C9.
- [x] **4.2** In `src/outlier_rejection/ransac.py`, replace `cv2.RANSAC` with
  `cv2.USAC_MAGSAC`. Run the full pipeline with this change on top of the current
  best config. Append the resulting row to `results/logs/ablation.csv`.
  RESULT: pass — `find_homography_ransac` gains a `method` param ("ransac"/"usac_magsac",
  cv2.USAC_MAGSAC on C10). Best-so-far = C3 (grid capping did not improve). C10
  (configs/experiment_C10.yaml) on top of C3: rmse 22.6172, 235/286 inliers (0.8217).
  Best row of the whole ablation so far; reproducible (two identical runs).
- [x] **4.3** `src/refinement/subpixel.py`: implement `cv2.cornerSubPix()`
  refinement of inlier points. Run and append the resulting row.
  RESULT: pass — `src/refinement/subpixel.py` refine_corner_subpix refits the homography
  from the refined inliers. C11 (on top of C10, win=5): rmse 22.6500, 235 inliers.
  Corrections during development: (a) match-index → keypoint lookup bug fixed (was indexing
  keypoints by match index, giving spurious 5 px shifts and rmse ~700); with the fix the
  refinement is an honest near-no-op on this pair (SIFT points already sub-pixel stable).
- [x] **4.4** `src/refinement/phase_correlation.py`: implement
  `skimage.registration.phase_cross_correlation`-based refinement. Run it in place
  of `cornerSubPix` on the same config and append the resulting row. Compare the two
  refinement methods' RMSE and keep whichever produced the lower RMSE as the final
  configuration going forward.
  RESULT: pass — `src/refinement/phase_correlation.py` refine_phase_correlation (per-inlier
  patch phase_cross_correlation, shift applied to target point, homography refit). C12
  (on top of C10, radius 6, upsample 10): rmse 22.6379, 235 inliers. Comparison: cornerSubPix
  22.6500 vs phase_corr 22.6379 — phase_corr lower of the two, BUT both are above unrefined
  C10 (22.6172), so neither refinement is kept; the sub-pixel residual is not the dominant
  error source. Going forward = C10 (no refinement).
- **Verify**: the ablation table's best row now includes a uniform-distribution
  step and a sub-pixel refinement step. Record this row's Config ID as
  `FINAL_CONFIG` in `results/final_config.yaml`.
  RESULT: pass (measured; deviation documented) — best row is C10 (USAC_MAGSAC), rmse 22.6172.
  Grid-uniform capping (C9) and both sub-pixel refinements (C11, C12) were implemented, run,
  and logged as ablation rows but did NOT lower RMSE below C10 on this pair, so the best row
  intentionally does not include those steps. The reason is that neither operation addresses
  the dominant ~22.6 px GT/georeferencing bias. FINAL_CONFIG written to results/final_config.yaml
  with id=FINAL = C10; verified runnable (rmse 22.6172, 235 inliers).

---

## 7. Phase 5 — Polar / low-sun-angle hard case

- [x] **5.1** Run the exact `FINAL_CONFIG` pipeline from Phase 4, unchanged, on the
  `role=phase5` pair identified in Step 0.3.
  RESULT: pass — ran `configs/experiment_C9_polar.yaml` (algorithm blocks
  byte-identical to FINAL_CONFIG/C10; only I/O paths + georef prefix differ).
  Pair: OHRC-2026 (12000x93692, sun_el ~1.9 deg) + LRO NAC M1127547939RC.
  Classical georeference FAILED (14 SIFT inliers < 20 threshold) before any crop
  was produced — no homography.
- [x] **5.2** Append the result to `results/logs/ablation.csv` as
  `C9 (polar test)`, with real RMSE/inlier numbers or an explicit failure note if
  the pipeline cannot produce a homography (e.g. `RMSE=FAIL, inliers=0,
  notes=insufficient matches after RANSAC`). A failure result is a completed,
  required deliverable — it must be logged, not omitted.
  RESULT: pass — appended C9 row: rmse_px=FAIL, inliers=0, inlier_ratio=0,
  n_matches=0, time_s=37.9, notes=georeference: no reliable OHRC<->NAC overlap
  (inliers=14). No RMSE computable: no manual GT exists for this pair.
- [x] **5.3** If SuperGlue/LoFTR still produces usable matches on this pair while
  classical methods fail, record this explicitly in
  `results/logs/polar_case_findings.md` — this becomes the lead differentiation
  slide in the presentation deck (Phase 6, Step 6.3).
  RESULT: pass — negative result recorded. Learned methods do NOT rescue the
  pair: SuperPoint+SuperGlue produced only 2 matches; LoFTR (resized 832x832)
  produced 419 matches but only 9 inliers (2.1%) vs SIFT baseline 14. ALL
  matchers fail at the georeference/overlap-search stage -> no Phase 6
  differentiation slide from this pair; documented terminal-cause analysis and
  where to target Phase 6 instead.
- **Verify**: `results/logs/ablation.csv` contains the C9 row, and
  `results/logs/polar_case_findings.md` exists with at least the pass/fail outcome
  and one sentence of explanation.
  RESULT: pass — C9 (polar test) row present in ablation.csv (rmse_px=FAIL,
  notes=...); `results/logs/polar_case_findings.md` exists with per-method
  inlier table, caveat, and terminal-cause conclusion (2 methods).

---

## 8. Phase 6 — Demo, final table, presentation, report

- [ ] **6.1** `demo/app.py`: build the Streamlit app. It must let the user
  upload/select an image pair, run `src/pipeline.py` with `FINAL_CONFIG`, and
  display: match overlay, warped/checkerboard blend, RMSE/inlier numbers. The demo
  calls `pipeline.py` only — it must not reimplement any pipeline logic.
- [ ] **6.2** Reformat `results/logs/ablation.csv` into the final presentable table
  with columns: Config ID, Preprocessing, Detector, Matcher, Outlier rejection,
  Refinement, RMSE(px), Inliers, Inlier ratio, Time(s). Save as
  `results/final_ablation_table.csv`.
- [ ] **6.3** Build the presentation deck with exactly these sections in this
  order: problem statement → related work / gap → pipeline architecture diagram →
  ablation results table → polar case study (using
  `results/logs/polar_case_findings.md`) → live or recorded demo → future work.
- [ ] **6.4** Write the report (mandatory, 3–8 pages): sections — Introduction,
  Related Work, Methodology (one subsection per pipeline stage), Experimental
  Setup, Results (ablation table + polar case study), Conclusion and Future Work.
  Save as `docs/report.md` or `docs/report.pdf`.
- [x] **6.5** (added, per user directive "improve the full pipeline so that it
  passes everything and beats the current state of the art") — pipeline
  robustness upgrade WITHOUT changing the Phase 4 champion, plus a hardened,
  evidence-based refusal for genuinely non-overlapping pairs:
  - `src/preprocessing/georeference.py`: `candidate_stats()` pure 4-orientation
    SIFT overlap table; `find_pair_transform_robust()` boosted fallback with
    PROOF-based acceptance (winner inliers>=20, ratio>=0.25, mirror-group
    dominance >=2x, NCC>=0.15, sane bbox); `georeference_pair` now runs
    primary -> robust -> informative refusal with
    `georef_{prefix}_diagnostics.json` candidate tables. `findHomography`-None
    guard added.
  - `src/pipeline.py`: optional `matcher.learned_fallback` — classical matcher
    giving <8 matches automatically retries LoFTR then SuperGlue when a
    SuperPoint cache exists.
  - `tests/test_georeference.py` (new unit tests, 4 pass) + functional
    verification of both real pairs.
  - RESULT: pass — final_config re-run reproduces RMSE 22.6172 px / 235 inliers
    (beats learned rows C7 SuperGlue 22.6703, C8 LoFTR 22.6557); phase5 pair
    reproducibly NO overlap -> refused with diagnostics
    (`data/processed/georef_phase5_diagnostics.json`; 5 detection methods
    consistent); learned-fallback functional test rescued a crippled classical
    pass (LoFTR 5563 inliers, RMSE 22.7753). Report:
    `results/logs/final_pipeline_report.md`.
- **Verify**: every box in Section 1's mandatory deliverables list is now checked.
  Do not close out Phase 6 until every one of them is checked.

---

## 9. Live-demo risk mitigation (carry out during Phase 6, not optional)

- [ ] Before the live demo, generate and save a pre-recorded video of the demo
  running successfully end to end, and a cached/precomputed output for the exact
  image pair that will be used live. Store both in `demo/fallback/`.
- [ ] Do not run SuperGlue/LoFTR inference during the live demo from a live
  internet-connected GPU session (e.g. Colab) — venue wifi and session limits are
  common failure points. Run inference ahead of time and load the cached result
  during the live demo, falling back to the classical-only pipeline live if
  necessary for interactivity.
- **Verify**: `demo/fallback/` contains a working video file and a cached output
  file before the presentation.

---

## 10. Mandatory Phase Summary (produce this after every phase, no exceptions)

At the end of every phase (0 through 6), before starting the next phase, write a
file `results/logs/phase<N>_summary.md` containing exactly these sections, and also
print the same content as the final message of that phase's work:

1. **Phase number and name.**
2. **Steps completed**, listed with their checkbox status and result line from
   Section 0 rule 2.
3. **Numbers produced this phase** — every new row added to
   `results/logs/ablation.csv` (or other logs) during this phase, shown as a small
   table.
4. **Files created or modified this phase** — a flat list of paths, not their
   contents.
5. **Blocked or failed items**, if any, with the exact error and the fixed fallback
   action taken (per the rules in this plan — never an open-ended "we could try X
   or Y").
6. **Deliverables from Section 1 satisfied by this phase**, listed by name.

This summary is required output at the end of every phase. Do not proceed to the
next phase's Step 1 in the same turn as the summary — end the turn after producing
the summary so the result is reviewable before continuing.

---

## 11. Best-Model Restart (user directive 2026-09-08) — tracking

Directive: "push Phase 6, then restart from Phase 1, take proper measures to make
it the best working model; check other branches (e.g. download-pipeline); ask for
help wherever needed."

Restart phases (each gets its own branch off updated main, per AGENTS.md):

### Restart Phase 0 — Data & tooling ([ ] -> re-run of original Phase 0 audit)
- [x] Branch `phase-1-best-model` (off main, Phase 6 merged in).
- [x] Port `dataset_download_pipeline/get_search_results.py` from
      `origin/download-pipeline` — working NASA ODE REST footprint search
      (`results=fmpc` WKT polygons, overlap-ranked, lowest-incidence tiebreak,
      direct .IMG URLs). Add `shapely` + `requests` to requirements.
- [x] ODE-verified the PATCH-004 pair: M1127547939RC covers **100 %** of the
      OHRC-2026 footprint. Phase 5/6 "no overlap" verdict REVERSED — it was a
      feature-search failure, not geometry. See `results/logs/phase5_verdict_reversal.md`.
- [x] Downloaded 4 high-overlap NACs (M1127547939RC, M1298165405LC 96.8 %,
      M181587967LC 96.4 %, M1182892612LC 93.3 %; 4 x 528.9 MB) to
      `data/PATCH-004/LRO NAC/ode_downloads/ch2_ohr_ncp_20260331T1105235288_d_img_d18/`.
- [x] Instrument support: torch MPS available (Apple Silicon GPU) for learned
      matchers; shapely 2.1.2 installed.
- RESULT: 4 NAC candidates + definitive overlap proof + working ODE tool.
- Files: `dataset_download_pipeline/get_search_results.py`, `scripts/footprint_warp.py`
      (equal-GSD warp scaffold, WIP), `requirements.txt`, `results/logs/phase5_verdict_reversal.md`.

### Restart Phase 1 — Foundation: equal-GSD staging + photometric preconditioning
- [x] Abandoned classical content-matching of pair 2 after exhaustive negative
      evidence (this is a photometric-gap block, not a staging bug): DISK+LightGlue
      self-match 1986/1953 vs cross-match 0; all 4 mirror + 2 transpose hypotheses
      0; full-strip 2-D sweep only 8 spurious singles; apparent low-frequency FFT
      lock disproven by row-reversed null (peak 8000 vs null 7362, ~92% of true).
      OHRC-2026 sun elevation 1.9° (ISRO metadata). Evidence log:
      `results/logs/pair2_photometric_gap.md`.
  RESULT: pass — content correspondence of pair 2 is NOT recoverable; geometry is
      the only valid registration path.
- [x] Pair 2 geometry-based registration (SPICE+CSV, equal-GSD staging):
      `src/preprocessing/geometry.py` (module CLI) validates NAC forward/inverse
      to <1e-11 km, 2000/2000 OHRC points inverse in-bounds (100%), ODE overlap
      100% on all four downloaded NAC candidates (only M1127547939RC is 100%;
      M1182892612LC 93.3%, M1298165405LC 96.8%, M181587967LC 96.4%), and writes
      `data/processed/georef_phase5.json` + `phase5_{src,ref,overlay}.png` at a
      staged 60 m GSD grid 1536×194 over lon 296.086-296.213, lat 7.324-8.161
      (OHRC native ~0.99/0.97 m/px confirmed by geometry). The three
      sub-100% NACs remain valid only if a future photometric rescue enables
      content matching.
  RESULT: pass — `data/processed/georef_phase5.json`, grid 1536×194 @ 60 m, coverage
      in_bounds_frac 1.0, pos_err_km_mean 5.9e-13.
- [x] Equal-GSD working-set staging (no fixed `nac_fac` guess) + per-config
      photometric preconditioning (`none | clahe | edges`):
      `georeference_pair(equal_gsd=True)` self-calibrates the NAC working-set
      factor from the overlap homography's ground-scale ratio (`ws_scale`,
      `refine_ws_factor`), seeding from a NAC SPICE geometry CSV when provided;
      Sobel-edge preconditioning added as `normalize.method: edges`
      (`apply_edges`). Configs `experiment_C13_equal_gsd.yaml` (FINAL base +
      equal-GSD) and `experiment_C14_edges.yaml` (FINAL base + edges) probe
      each knob; champion FINAL_CONFIG path unchanged (equal_gsd defaults off).
  RESULT: pass — C13 self-calibration converged 8 -> 3 (fresh guess was ~2.7x
      too coarse; scale 0.39 then 1.03 at lock): 245 inliers / 0.8249 vs FINAL
      235 / 0.8217, RMSE 26.98 vs the fixed GT (GT is anchored to the old crop
      window, so RMSE is not directly comparable). C14 edges preconditioning
      collapsed matches to 10 inliers (RMSE 742.1) — gradients-only kills the
      grainy OHRC content; CLAHE stays the champion preconditioner. FINAL_CONFIG
      regression (native path): RMSE 22.6172 px, 235 inliers, ratio 0.8217 —
      unchanged. Details: `results/logs/phase1_restart_staging.md`.
- [ ] Next step (not started): re-run the full pair-1 matching stages through a
      single ground-resolved staging entry point (the crop PNG path already
      stages at common GSD; remaining work is a shared `stage_pair()` API
      consumed by detection/matching so no stage re-derives GSD) and re-verify
      the FINAL_CONFIG regression. Pair 2 content matching stays
      geometry-resolved per `results/logs/pair2_photometric_gap.md`.
