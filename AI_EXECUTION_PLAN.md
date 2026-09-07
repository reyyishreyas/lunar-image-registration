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
- [ ] - [ ] Verify that the current working directory is the existing
  `lunar-image-registration` Git repository and that its remote is:
  `https://github.com/reyyishreyas/lunar-image-registration`.
  Do not clone another copy.  into the
  working directory. This repo already defines the required directory structure
  (`configs/`, `data/`, `demo/`, `docs/`, `notebooks/`, `results/`, `scripts/`,
  `src/`, `tests/`, `weights/`) and the required CLI entry points
  (`scripts/run_pipeline.py`, `scripts/run_ablation.py`,
  `scripts/pick_ground_truth.py`, `demo/app.py`). Use this exact structure — do not
  invent a different layout.
- [ ] Run `find . -maxdepth 3 -type f` inside the cloned repo and record which files
  are empty stubs versus already implemented. Write this inventory to
  `results/logs/repo_inventory.csv` with columns `path, status(stub/implemented),
  lines_of_code`.
- [ ] Inventory `notebooks/*.ipynb` specifically: for each notebook, run
  `jupyter nbconvert --to script <notebook> --stdout | wc -l` to get a size
  indicator without loading the full notebook into context, then open only the
  cells needed to determine what each notebook does. Record one line per notebook
  in `results/logs/repo_inventory.csv`: `notebook name, purpose, reusable(yes/no)`.
- [ ] For any notebook or script found to already implement a stage listed in
  Sections 3–8 of this plan, reuse it — port its logic into the corresponding
  `src/<stage>/` module rather than rewriting from scratch. Do not discard existing
  work.
- **Verify**: `results/logs/repo_inventory.csv` exists and lists every file in the
  cloned repo.

### Step 0.2 — Environment
- [ ] Check Python version (must be 3.10+) and GPU visibility with
  `python -c "import torch; print(torch.cuda.is_available())"`.
- [ ] Use the repo's existing `requirements.txt` and `environment.yml` as the base.
  Add any of the following missing from it: `opencv-python`,
  `opencv-contrib-python`, `scikit-image`, `numpy`, `scipy`, `torch`, `torchvision`,
  `rasterio`, `pygeodesy`, `matplotlib`, `pandas`, `pyyaml`, `tqdm`, `streamlit`.
- [ ] Create and activate a virtual environment (`python -m venv venv`), then
  `pip install -r requirements.txt`.
- **Verify**: `pip list` shows every package above with no import errors on
  `import cv2, torch, rasterio`.

### Step 0.3 — Data audit
- [ ] Locate the user's local dataset directory and run
  `find <dataset_root> -maxdepth 3` to list contents without dumping full file
  contents into context.
- [ ] For every distinct image file found, run `gdalinfo <file>` and extract only:
  resolution (m/px), projection, corner coordinates, sun angle/acquisition metadata
  if present, bit depth, format. Write one row per file to `data/manifest.csv` with
  columns `path, instrument, resolution_m, format, sun_angle, corner_coords, notes`.
  Do not print full `gdalinfo` output into the conversation.
- [ ] From `data/manifest.csv`, identify and record in the same file (via a `role`
  column):
  1. One OHRC–LRO NAC pair over the same equatorial location → tag `role=phase1`.
  2. One polar or low-sun-angle pair → tag `role=phase5`.
  3. Any IIRS or TMC-2 data present → tag `role=multimodal_stress_test`.
- **Verify**: `data/manifest.csv` has at least one row tagged `role=phase1` and at
  least one row tagged `role=phase5`. If either is missing, stop this step and
  report exactly what is missing — do not substitute a different pair silently and
  do not proceed to Phase 1 without a confirmed `role=phase1` pair.

---

## 3. Phase 1 — End-to-end baseline pipeline (Config C1)

Work on a 1024×1024 crop of the `role=phase1` pair identified in Step 0.3.

- [ ] **1.1** `src/preprocessing/georeference.py`: crop/resize both images to the
  same approximate ground area using the corner coordinates in `data/manifest.csv`.
  Output: `data/processed/pair1_src.png`, `data/processed/pair1_ref.png`.
- [ ] **1.2** `src/detection/classical.py`: implement
  `detect_sift(img) -> keypoints, descriptors` using `cv2.SIFT_create()`.
- [ ] **1.3** `src/matching/classical_match.py`: implement
  `match_bf_ratio(desc1, desc2, ratio=0.75) -> matches` using `cv2.BFMatcher` with
  Lowe's ratio test.
- [ ] **1.4** `src/outlier_rejection/ransac.py`: implement
  `find_homography_ransac(pts1, pts2) -> H, inlier_mask` using
  `cv2.findHomography(..., cv2.RANSAC, 5.0)`.
- [ ] **1.5** `src/evaluation/visualize.py`: wrap `cv2.drawMatches()`; save to
  `results/figures/pair1_matches.png`.
- [ ] **1.6** Manually pick 20 ground-truth control points between the two images
  using `scripts/pick_ground_truth.py`. Save to `data/ground_truth/pair1_gt.csv`
  with columns `x1,y1,x2,y2`. This step is manual and must be done by a human
  before continuing — do not fabricate ground-truth points.
- [ ] **1.7** `src/evaluation/metrics.py`: implement `rmse(H, gt_points) -> float`.
  Apply `H` to each `(x1,y1)`, compare to `(x2,y2)`, report RMSE in pixels.
- [ ] **1.8** `src/pipeline.py`: orchestrate steps 1.1–1.7 driven by
  `configs/experiment_C1.yaml`. This is the single entry point for every later
  phase — later phases add new functions and new config files, never new
  orchestration logic in a separate script.
- [ ] **1.9** Create `results/logs/ablation.csv` with columns: `Config ID,
  Preprocessing, Detector, Matcher, Outlier rejection, Refinement, RMSE(px),
  Inliers, Inlier ratio, Time(s)`. Run `python scripts/run_pipeline.py --config
  configs/experiment_C1.yaml` and append the C1 row with real measured numbers.
- **Verify**: the C1 row in `results/logs/ablation.csv` has non-empty numeric values
  in every numeric column, and `results/figures/pair1_matches.png` exists.

---

## 4. Phase 2 — Preprocessing (Configs C2–C4)

Run each of the following one at a time, re-running the full pipeline via
`scripts/run_pipeline.py` after each addition and appending the corresponding row
to `results/logs/ablation.csv`.

- [ ] **2.1** `src/preprocessing/normalize.py`: implement histogram matching with
  `skimage.exposure.match_histograms`. Run as Config C2. Append the C2 row.
- [ ] **2.2** `src/preprocessing/normalize.py`: implement CLAHE with
  `cv2.createCLAHE`. Run with `clipLimit` values `2.0`, `4.0`, and `8.0`, each on an
  `8x8` tile grid. Select the `clipLimit` with the lowest RMSE. Run the full
  pipeline with that setting as Config C3. Append the C3 row, and also log the
  RMSE for all three `clipLimit` trials to `results/logs/clahe_sweep.csv`.
- [ ] **2.3** `src/preprocessing/shadow_correct.py`: implement gamma correction for
  shadow regions (gamma = 0.5). Run as Config C4. Append the C4 row.
- **Verify**: `results/logs/ablation.csv` has rows C1–C4, each with real measured
  numbers. `results/logs/clahe_sweep.csv` has 3 rows.

---

## 5. Phase 3 — Detectors and matchers (Configs C5–C8)

- [ ] **3.1** `src/detection/classical.py`: implement AKAZE with
  `cv2.AKAZE_create()`. Run the pipeline with the best preprocessing config from
  Phase 2 and AKAZE instead of SIFT. Append as Config C5.
- [ ] **3.2** Search for an existing Python port of RIFT2. If one is found and
  installs cleanly within one hour, integrate it into `src/detection/classical.py`
  and run as Config C6. If no working Python port installs within one hour, do not
  spend further time on it: write `RESULT: blocked — no working RIFT2 Python port
  found within time limit` under this step, cite RIFT2 in the written report's
  related-work section instead, and continue to Step 3.3. This is the fixed
  fallback procedure — do not re-attempt RIFT2 later in the project.
- [ ] **3.3** `src/detection/learned.py` and `src/matching/learned_match.py`: clone
  the official SuperPoint+SuperGlue repository into `third_party/`. Do not vendor
  its code into `src/`. Write adapter functions in `src/detection/learned.py` and
  `src/matching/learned_match.py` matching the same input/output signature as the
  classical stages (image in → keypoints/matches/transform out) so `pipeline.py`
  calls it unchanged. Run with pretrained weights as Config C7. This step is
  mandatory and is the single most important stage of the project — it is not
  skipped under any time constraint. If GPU access is unavailable locally, run this
  step on Google Colab or Kaggle and copy the resulting weights/outputs back into
  the local `results/` directory.
- [ ] **3.4** Clone the official LoFTR repository into `third_party/`. Write
  adapter functions in `src/matching/learned_match.py` with the same signature. Run
  with pretrained weights as Config C8. This step is mandatory.
- **Verify**: `results/logs/ablation.csv` has rows through C8 (C6 may instead be a
  `blocked` note per 3.2, in which case renumber C7→C6 and C8→C7 in the CSV so the
  table has no gap). At least one learned-matcher row (SuperGlue or LoFTR) must show
  lower RMSE than the best classical row. If it does not, this indicates an
  integration bug — check the input image normalization range expected by the
  pretrained model, fix it, and re-run before proceeding to Phase 4.

---

## 6. Phase 4 — Outlier rejection and sub-pixel refinement

- [ ] **4.1** `src/outlier_rejection/grid_uniform.py`: implement grid-based match
  capping before RANSAC. Run with grid sizes `4x4`, `8x8`, and `16x16` on top of the
  best-performing config from Phase 3. Log all three to
  `results/logs/grid_sweep.csv`. Select the grid size with the lowest RMSE and run
  the full pipeline with it as the next ablation row (append to
  `results/logs/ablation.csv` with the next sequential Config ID).
- [ ] **4.2** In `src/outlier_rejection/ransac.py`, replace `cv2.RANSAC` with
  `cv2.USAC_MAGSAC`. Run the full pipeline with this change on top of the current
  best config. Append the resulting row to `results/logs/ablation.csv`.
- [ ] **4.3** `src/refinement/subpixel.py`: implement `cv2.cornerSubPix()`
  refinement of inlier points. Run and append the resulting row.
- [ ] **4.4** `src/refinement/phase_correlation.py`: implement
  `skimage.registration.phase_cross_correlation`-based refinement. Run it in place
  of `cornerSubPix` on the same config and append the resulting row. Compare the two
  refinement methods' RMSE and keep whichever produced the lower RMSE as the final
  configuration going forward.
- **Verify**: the ablation table's best row now includes a uniform-distribution
  step and a sub-pixel refinement step. Record this row's Config ID as
  `FINAL_CONFIG` in `results/final_config.yaml`.

---

## 7. Phase 5 — Polar / low-sun-angle hard case

- [ ] **5.1** Run the exact `FINAL_CONFIG` pipeline from Phase 4, unchanged, on the
  `role=phase5` pair identified in Step 0.3.
- [ ] **5.2** Append the result to `results/logs/ablation.csv` as
  `C9 (polar test)`, with real RMSE/inlier numbers or an explicit failure note if
  the pipeline cannot produce a homography (e.g. `RMSE=FAIL, inliers=0,
  notes=insufficient matches after RANSAC`). A failure result is a completed,
  required deliverable — it must be logged, not omitted.
- [ ] **5.3** If SuperGlue/LoFTR still produces usable matches on this pair while
  classical methods fail, record this explicitly in
  `results/logs/polar_case_findings.md` — this becomes the lead differentiation
  slide in the presentation deck (Phase 6, Step 6.3).
- **Verify**: `results/logs/ablation.csv` contains the C9 row, and
  `results/logs/polar_case_findings.md` exists with at least the pass/fail outcome
  and one sentence of explanation.

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
