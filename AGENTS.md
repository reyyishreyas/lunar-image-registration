# OpenCode Instructions — Lunar Image Registration

## Source of Truth

The file `AI_EXECUTION_PLAN.md` is the mandatory execution plan for this project.

Treat it as authoritative.

Do not skip, reorder, reinterpret, remove, or silently alter its required steps.

The execution plan is divided into Phase 0 through Phase 6.

Work through the phases sequentially.

---

## Verified Repository State (as of Phase 0)

These facts were confirmed by inspection. They are hard-earned context that will
otherwise cost time or cause mistakes.

- **The codebase is a scaffold of empty stubs.** Every `src/<stage>/*.py`, every
  `scripts/*.py`, every `configs/*.yaml`, every `tests/*.py`, and
  `demo/app.py` is **0 bytes**. The plan's entry points (`scripts/run_pipeline.py`,
  `src/pipeline.py`) do not yet exist as code. Phase 1 effectively starts from
  scratch; do not assume any stage is already implemented.
- **`requirements.txt` and `environment.yml` are empty (0 bytes).**
  Execution Plan Step 0.2 says "use the repo's existing `requirements.txt` /
  `environment.yml` as the base" — running `pip install -r requirements.txt` will
  install **nothing**. Write real dependency lists yourself during Phase 0 before
  relying on them. Step 0.2's "Add any of the following missing from it" list is
  effectively the full install list.
- **All real algorithm logic lives in notebooks only.**
  `notebooks/algo*.ipynb` and `notebooks/moonalgo*.ipynb` contain working code for
  SIFT/ORB/AKAZE/BRISK/KAZE + ASIFT (algo1), a RIFT2 Python port clone (algo2),
  SuperPoint/DISK/ALIKED via LightGlue (algo3), and D2-Net (algo4). Per Phase 0
  Step 0.1's reuse rule, port this logic into `src/<stage>/` rather than rewriting.
  The `01..04_*.ipynb` files are empty placeholders.
- **Local environment (host) is NOT clean-install compatible.** The active
  interpreter is conda Python **3.13** with **CPU-only torch 2.13** (`cuda=False`),
  cv2 5.0.0, rasterio 1.5.1, skimage 0.25.2, pandas, yaml, streamlit present. But
  `gdalinfo` is **not on PATH** (Phase 0 Step 0.3 and the data audit rely on it).
  SuperGlue/LoFTR/SuperPoint need GPU or Colab fallback (plan Step 3.3).
- **Data lives directly under `data/`, NOT in the gitignored subdirs.** Actual images
  are in `data/LRO NAC 2/`, `data/PATCH-001/`, ... `data/PATCH-004/` (multi-GB; some
  `.zip`). `PATCH-004/` contains `IIRS/`, `LRO NAC/`, `OHRC/`, `TMC/`. The plan's
  Step 0.3 assumes data under a gitignored `data/raw/`; instead most data sits in
  untracked top-level dirs. `data/ground_truth/` and `data/processed/` are empty.
- **Paths contain spaces** (e.g. `"data/LRO NAC 2"`) — always quote them.
- **This volume is a `.mounty` FUSE mount and is case-sensitive.** macOS produces
  `._*` AppleDouble metadata files that are untracked; `AGENTS.md` and `agents.md`
  are distinct filenames. Ignore the `._*` noise in `git status`.
- **`agents.md` was renamed to `AGENTS.md`** (this file, canonical). Git workflow,
  phase-stop, and commit/push rules in later sections are binding.

---

## Existing Repository

This is an EXISTING Git repository.

GitHub remote:

https://github.com/reyyishreyas/lunar-image-registration

The repository is already present locally.

DO NOT clone the repository again.

DO NOT create a second repository.

DO NOT replace the existing working directory.

DO NOT discard existing local work.

Phase 0 Step 0.1 originally mentions cloning the repository. Since the repository already exists locally, interpret that step as:

"Verify that the current working directory is the existing lunar-image-registration repository and verify its GitHub remote."

Do not perform another clone.

---

## Git Workflow

Every phase must be developed on its own Git branch.

Before beginning a phase:

1. Make sure the current branch is clean enough to safely begin work.
2. Update main:

   git checkout main
   git pull origin main

3. Create a new branch:

   git checkout -b phase-X-short-description

Example:

   git checkout -b phase-0-repository-setup

Never develop directly on main.

---

## Phase Execution

Execute exactly one phase at a time.

Within a phase, execute its steps in numeric order.

Do not begin a step until the previous step has been completed and marked `[x]`.

After completing a step:

1. Mark its checkbox `[x]` in `AI_EXECUTION_PLAN.md`.
2. Add the required:

   RESULT: pass — <key numbers or artifact path>

3. If the step fails, follow the failure rules in the execution plan.
4. Do not falsely mark a failed step as complete.

---

## Testing Requirement

Writing code does NOT mean a step is complete.

Every implementation must be tested.

Where applicable:

- Unit tests
- Integration tests
- Pipeline execution
- Real input/output verification
- Numerical metrics
- Visual output verification
- Runtime verification

For this project, actual pipeline execution is especially important.

Do not claim that a computer-vision or registration stage works merely because imports succeed or unit tests pass.

---

## Crop-First Requirement

Follow the execution plan's crop-first rule.

For every new pipeline stage:

1. Test using a crop of 1024×1024 pixels or smaller.
2. Verify correctness.
3. Only then run the stage at full resolution.

Never immediately perform an expensive full-resolution experiment when the crop verification has not passed.

---

## Data Handling

Never dump large datasets, rasters, arrays, notebook contents, model outputs, or logs into the conversation.

Use commands such as:

- head
- wc
- grep
- gdalinfo
- file
- identify
- small crops

Write detailed results to:

results/logs/

Only report summary metrics in the conversation.

Important metrics include:

- RMSE
- MAE where applicable
- Inlier count
- Inlier ratio
- Runtime
- Image dimensions
- Resolution
- Number of matches

---

## Notebooks

Notebooks are not pipeline implementations.

If useful logic already exists inside a notebook:

1. Inspect the relevant cells.
2. Reuse the logic.
3. Port it into the appropriate `src/` module.
4. Keep the notebook for manual/exploratory work.

Do not place production pipeline logic inside notebooks.

---

## Pipeline Architecture

The main pipeline entry point is:

scripts/run_pipeline.py

which calls:

src/pipeline.py

Later phases must extend the existing architecture.

Do not create independent competing pipeline implementations unless explicitly required by `AI_EXECUTION_PLAN.md`.

The Streamlit application must call the pipeline rather than duplicate its logic.

---

## Existing User Work

Before modifying files:

git status

Treat existing user modifications as intentional unless proven otherwise.

Never run:

git reset --hard

Never use:

git clean -fd

Never force push.

Never delete existing work without explicit user approval.

Do not overwrite unrelated modifications.

---

## Git Commit Rules

Only commit after:

1. The current phase's required implementation is complete.
2. Required tests pass.
3. Functional verification passes where applicable.
4. The final `git diff` has been inspected.
5. Only relevant files are included.

Use meaningful commit messages.

Example:

git add <relevant files>
git commit -m "Complete Phase 1 baseline registration pipeline"

Do not use meaningless messages such as:

- changes
- update
- fixes
- final
- stuff

---

## Git Push Rules

After successful phase verification:

git push -u origin <current-phase-branch>

Never force push.

Confirm that the push succeeded.

Then stop.

Do NOT automatically merge the branch.

The user will review and merge the phase branch into main.

---

## Mandatory Phase Stop

At the end of EVERY phase:

1. Complete the required Phase Summary.
2. Create:

results/logs/phase<N>_summary.md

3. Print the same summary in the final response.
4. Commit the summary and all phase changes.
5. Push the phase branch.
6. Clearly report that the branch is ready for merge.
7. STOP.

Do not begin the next phase in the same turn/session after producing the Phase Summary.

Wait until the user starts the next phase after merging.

---

## Required Phase Completion Format

The final response after a successful phase MUST contain:

========================================
PHASE X COMPLETED
========================================

Phase:
<phase number and name>

Steps Completed:
- [x] Step X.X
  RESULT: pass — ...

Numbers Produced:
<table or concise list>

Files Created/Modified:
- path
- path
- path

Blocked or Failed Items:
- None
OR
- <exact failure and required fallback>

Deliverables Satisfied:
- <deliverable>

Testing:
- <command>
- <result>

Functional Verification:
- <command>
- <result>

Git:
- Branch: <branch>
- Commit: <commit hash>
- Push: SUCCESS

Status:
READY FOR MERGE

========================================

STOP AFTER THIS REPORT.

---

## Failed Phase Format

If the phase cannot be completed:

========================================
PHASE X FAILED
========================================

Problem:
<description>

Exact Error:
<error>

Tests:
<tests and results>

Action:
<fix or mandatory fallback from AI_EXECUTION_PLAN.md>

Git:
NO PUSH

========================================

Do not claim completion.

---

## Merge Workflow

After the user merges a phase:

The next phase must begin from the updated main branch.

Run:

git checkout main
git pull origin main

Then create the next phase branch:

git checkout -b phase-X-short-description

Never start the next phase from an old phase branch.

---

## Final Rule

A phase is complete ONLY when:

IMPLEMENTED
+
TESTED
+
FUNCTIONALLY VERIFIED
+
RESULTS LOGGED
+
EXECUTION PLAN UPDATED
+
PHASE SUMMARY CREATED
+
COMMITTED
+
PUSHED
=
PHASE COMPLETED

The branch is then READY FOR MERGE.

Never proceed to the next phase automatically.