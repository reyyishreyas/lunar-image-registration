# Team Guide — Lunar Image Registration Project

This document is for **our team only**. It explains how the repo is organized, who owns what, how to add your work without breaking anyone else's, and the exact workflow to follow day-to-day. Read this fully before writing any code.

---

## Table of Contents

- [Before You Start](#before-you-start)
- [Repo Walkthrough — What Goes Where](#repo-walkthrough--what-goes-where)
- [Role Assignments](#role-assignments)
- [Git Workflow](#git-workflow)
- [How to Add a New Experiment](#how-to-add-a-new-experiment)
- [How to Add a New Pipeline Function](#how-to-add-a-new-pipeline-function)
- [Coding Conventions](#coding-conventions)
- [Logging Results](#logging-results)
- [Daily/Weekly Checklist](#dailyweekly-checklist)
- [Communication Norms](#communication-norms)
- [Common Mistakes to Avoid](#common-mistakes-to-avoid)
- [Milestone Timeline](#milestone-timeline)

---

## Before You Start

1. Clone the repo and set up your environment exactly as described in the main `README.md` (`pip install -r requirements.txt`).
2. Read `docs/methodology.md` fully — this explains *why* the pipeline is structured the way it is. Don't skip this; it'll save you from redoing work.
3. Join the team group chat/channel and confirm your assigned role below.
4. Make sure you can run the baseline pipeline end-to-end on the sample data **before** touching any new code:
   ```bash
   python scripts/run_pipeline.py --config configs/experiment_sift_baseline.yaml
   ```
   If this doesn't run for you, fix your environment first — don't start building on a broken setup.

---

## Repo Walkthrough — What Goes Where

| Folder | What lives here | Who touches it |
|---|---|---|
| `configs/` | One YAML file per experiment — defines which detector/matcher/preprocessing combo to run | Everyone, whenever testing a new combination |
| `data/raw/` | Original downloaded images, untouched | Data lead downloads; nobody edits these files directly |
| `data/processed/` | Output of preprocessing (georeferenced, resampled, normalized images) | Generated automatically by the pipeline — don't hand-edit |
| `data/ground_truth/` | CSV files of manually picked control points, used for RMSE | Evaluation lead maintains this |
| `weights/` | Pretrained model weights (SuperPoint, SuperGlue, LoFTR) | Deep learning lead downloads and documents these |
| `src/io_utils/` | Reading image formats, extracting metadata (sun angle, resolution, corners) | Data & preprocessing lead |
| `src/preprocessing/` | Georeferencing, resampling, normalization, shadow correction | Data & preprocessing lead |
| `src/detection/` | All feature detectors — classical, cross-modal, learned | Classical CV lead + Deep learning lead (separate files, no overlap) |
| `src/matching/` | All feature matchers | Classical CV lead + Deep learning lead |
| `src/outlier_rejection/` | RANSAC variants + our custom grid-uniformity logic | Classical CV lead |
| `src/refinement/` | Sub-pixel refinement methods | Classical CV lead |
| `src/evaluation/` | RMSE, inlier ratio, visualization code | Evaluation lead |
| `scripts/` | Command-line entry points that tie everything together | Whoever needs a new automation script — coordinate before adding |
| `notebooks/` | Scratch work, exploration, quick visual checks | Anyone — but nothing here counts as "final" code |
| `results/logs/` | The shared ablation results CSV | Evaluation lead owns updating this, everyone can read it |
| `results/figures/` | Saved plots/images for the PPT | Demo/presentation lead pulls from here |
| `demo/` | The Streamlit app | Demo/presentation lead |
| `tests/` | Unit tests | Whoever writes a function should also write its test |
| `docs/` | Methodology, related work, diagrams | Research/documentation lead, but everyone can contribute notes |

**Golden rule:** if you're not sure whether a file belongs to you, ask in the group chat before editing it. Two people editing the same file at the same time is the #1 cause of merge conflicts and lost work.

---

## Role Assignments

Fill in names here and keep it updated:

| Role | Name | Owns |
|---|---|---|
| Data & Preprocessing Lead | | `data/`, `src/io_utils/`, `src/preprocessing/` |
| Classical CV Lead | | `src/detection/classical.py`, `src/detection/cross_modal.py`, `src/matching/classical_match.py`, `src/outlier_rejection/`, `src/refinement/` |
| Deep Learning Lead | | `src/detection/learned.py`, `src/matching/learned_match.py`, `weights/` |
| Evaluation & Metrics Lead | | `src/evaluation/`, `results/logs/`, ground-truth point picking |
| Demo & Presentation Lead | | `demo/`, `results/figures/`, the PPT itself |
| Research/Documentation Lead (if 6th member) | | `docs/`, paper drafting, literature notes |

Even with clear ownership, **everyone should be able to explain every stage** — SIH judges frequently quiz random team members on parts they didn't personally build.

---

## Git Workflow

1. **Never commit directly to `main`.** `main` should always be in a working, demo-able state.
2. Create a feature branch for whatever you're working on:
   ```bash
   git checkout -b feature/shadow-correction
   ```
3. Commit often, with clear messages:
   ```bash
   git commit -m "Add CLAHE-based shadow correction with configurable clip limit"
   ```
4. Before opening a pull request, pull the latest `main` and resolve conflicts locally:
   ```bash
   git pull origin main
   ```
5. Open a pull request into `main`. At least one other teammate should skim it before merging — even a quick glance catches obvious bugs.
6. After merging, confirm the pipeline still runs end-to-end:
   ```bash
   python scripts/run_pipeline.py --config configs/experiment_sift_baseline.yaml
   ```
7. Delete the feature branch once merged to keep things tidy.

**Branch naming convention:** `feature/<short-description>`, `fix/<short-description>`, `experiment/<short-description>`.

---

## How to Add a New Experiment

You should **never** need to write new pipeline code just to try a different combination of existing components — that's what configs are for.

1. Copy an existing config file:
   ```bash
   cp configs/experiment_sift_baseline.yaml configs/experiment_akaze_clahe.yaml
   ```
2. Edit the fields you want to change (detector, matcher, preprocessing steps, RANSAC threshold, etc.).
3. Run it:
   ```bash
   python scripts/run_pipeline.py --config configs/experiment_akaze_clahe.yaml
   ```
4. Confirm the result got logged to `results/logs/ablation_results.csv` with a unique config ID.
5. If you need to test many configs at once, add them all to `configs/` and run:
   ```bash
   python scripts/run_ablation.py --configs configs/
   ```

If the combination you want to test **isn't possible with existing components** (e.g. a totally new detector), see the next section.

---

## How to Add a New Pipeline Function

Every stage follows the same contract so pieces are swappable. Before adding a new function:

1. Check the existing files in that stage's folder (e.g. `src/detection/classical.py`) to see the expected function signature.
2. Write your function following the same input/output pattern — for example, a detector should take an image and return `(keypoints, descriptors)`, regardless of whether it's SIFT or SuperPoint underneath.
3. Register it in `src/pipeline.py` so it can be selected via the config file (usually a simple dictionary lookup by name).
4. Add a basic test in `tests/` confirming it runs on a sample image without crashing.
5. Add a short docstring explaining what it does and where it came from (paper/library reference).
6. Add an entry to `docs/related_work.md` if it's based on a specific paper, so we don't lose track of citations for the report/paper later.

---

## Coding Conventions

- **Python style:** follow PEP 8. Use `black` for auto-formatting if possible (`pip install black`, then `black src/`).
- **Function signatures:** keep them consistent within a stage — don't invent a new return format just for your function.
- **No hardcoded file paths.** Use config values or function arguments instead of hardcoding `"data/raw/ohrc1.tif"` inside a function.
- **No secrets or API keys committed.** If you need credentials for a data portal, keep them in a local `.env` file (already gitignored).
- **Comment the "why," not the "what."** Code should be readable enough to show *what* it does; comments should explain *why* a particular threshold, method, or parameter was chosen.
- **One function, one responsibility.** If a function is doing preprocessing *and* matching, split it.

---

## Logging Results

Every experiment run should append a row to `results/logs/ablation_results.csv` with, at minimum:

| Column | Description |
|---|---|
| `config_id` | Matches the config filename |
| `preprocessing` | Short description of steps applied |
| `detector` | Detector used |
| `matcher` | Matcher used |
| `outlier_rejection` | Method used |
| `refinement` | Method used |
| `rmse_px` | RMSE in pixels |
| `inliers` | Inlier count |
| `inlier_ratio` | Inlier ratio |
| `runtime_sec` | End-to-end runtime |
| `notes` | Anything unusual (e.g. "polar image, partial failure") |

This CSV becomes both our results slide and the experiments section of the paper — keep it accurate and don't overwrite old rows.

---

## Daily/Weekly Checklist

**Every work session:**
- [ ] Pull latest `main` before starting
- [ ] Work on a feature branch, not `main`
- [ ] Run the baseline pipeline once before and after your changes to confirm nothing broke

**Every few days:**
- [ ] Sync with the team on progress (quick standup — what you did, what's blocked, what's next)
- [ ] Update `results/logs/ablation_results.csv` with any new runs
- [ ] Check `docs/methodology.md` and `docs/related_work.md` are still accurate if your approach changed

**Before any milestone/demo:**
- [ ] Confirm `demo/app.py` runs end-to-end without errors
- [ ] Confirm the current best config is copied to `results/final_config.yaml`
- [ ] Confirm the README's usage commands still match reality

---

## Communication Norms

- Post in the team chat **before** starting major changes to shared files (e.g. `pipeline.py`, config schema) so others aren't blindsided by breaking changes.
- If you're stuck for more than ~30-45 minutes on a setup/environment issue, ask the team instead of silently grinding — someone may have already solved it.
- Flag any dead ends early (e.g. "RIFT2 has no usable Python port, I'm blocked") rather than late — we can reroute effort while there's still time.
- Be honest about partial results. A documented failure case (e.g. "our pipeline also struggles at the poles, here's why") is more valuable for the presentation than silence.

---

## Common Mistakes to Avoid

- **Editing files inside `data/processed/` by hand.** These should always be regenerated by the pipeline — hand edits get silently overwritten and cause confusing bugs.
- **Putting real logic only in a notebook.** Notebooks are for exploration; if it works, move it into `src/` so the whole team can use it.
- **Hardcoding a specific image pair inside a pipeline function.** Pipeline code must work generically — the PS explicitly requires a generic solution, not one hand-tuned to a single pair.
- **Committing large data/weight files to git.** These belong in `data/` and `weights/`, which are gitignored — share large files via a drive link documented in `data/README.md` instead.
- **Skipping the ground-truth step.** Without manually verified control points, you cannot compute a real RMSE — don't wait until the last day to build this.
- **Only testing on easy equatorial images.** The polar/low-sun-angle case is where our project's real value gets demonstrated — don't skip it for time reasons if at all avoidable.

---

## Milestone Timeline

*(Update dates based on your actual SIH deadlines)*

| Milestone | Target date | Owner(s) |
|---|---|---|
| Environment set up, baseline pipeline runs | | Everyone |
| First working end-to-end registration (any accuracy) | | Classical CV Lead |
| Preprocessing improvements integrated | | Data & Preprocessing Lead |
| Deep learning matcher integrated | | Deep Learning Lead |
| Ablation table with 5+ configurations complete | | Evaluation Lead |
| Polar/hard-case result obtained | | Whole team |
| Streamlit demo functional | | Demo Lead |
| PPT draft complete | | Demo Lead + Research Lead |
| Full rehearsal | | Everyone |

---

Questions about anything in this guide? Ask in the team chat rather than guessing — an incorrect assumption here can cost hours of duplicated or wasted work.