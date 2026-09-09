# Phase 11 Summary — Streamlit app surfaces the Match/No-Match decision + "any image pair"

## Objective

Phase 10 delivered the decision engine on the CLI. Phase 11 puts it in front of
the user: the demo app (demo/app.py) now (1) shows the definitive **MATCH /
NO MATCH** banner with reason + evidence for every run, (2) renders the **best
aligned product** (aligned + diff) when a content match produced one, and
(3) gains a first-class **"Any image pair (auto-detect)"** input mode — drop any
two OHRC/TMC/IIRS/NAC files, the app detects each file's sensor from its
filename and runs the right register with the honest verdict.

## What was delivered

### 1. Decision banner — `_decision_banner(rep)`
- Renders `report["decision"]` as a big **✅ MATCH** / **⛔ NO MATCH** headline
  with the machine explanation; colour follows the cause
  (frame_disagreement → info, cross_sensor/dark → warning, content → success).
- Shows the `best_product` path caption.
- Wired as the first element of `_render_sensor`, which serves the multi-sensor
  presets, the CLI-style builds, and the new "Any image pair" mode.

### 2. Best aligned product — `_render_best_product(rep)`
- When a content-registered run wrote `artifacts.best_aligned` (or
  `decision.best_product`), renders the warped-source deliverable
  (`BEST ALIGNED` frame) plus the `DIFF` map.
- Appended after the existing What-changed / Check-it sections in `_render_sensor`.

### 3. "Any image pair (auto-detect)" input mode
- New sidebar radio entry (now the default first option) with its own
  `_execute_any` handler that:
  - offers four real on-disk example pairs (*OHRC 2021→MATCH*,
    *TMC 2026→frame disagreement*, *OHRC 2026→frame disagreement*,
    *IIRS 2021→cross-sensor*) plus free-form source/ref image + geometry paths;
  - calls `detect_pair(src, ref)` to auto-detect the sensor pair, shows it, then
    runs `run_sensor_auto` and attaches a decision if the report lacks one;
  - reuses the same `_render_sensor` path so decisions/products render uniformly.
- `_sensor_labels` generalised to every supported pair
  (`tmc-nac`, `iirs-nac`, `ohrc-tmc`, same-sensor …) instead of hard-coded triples.
- Fixed an invalid conditional expression left over from an early draft of the
  banner (would have been a SyntaxError at import).

## Bugfix round (same phase)

Two issues reported after the first push of Phase 11:

1. **`ImportError: cannot import name 'detect_pair' from 'src.auto_pipeline'`**
   Root cause: the repo lives on a `.mounty` FUSE mount whose timestamps make
   Python's `.pyc` invalidation unreliable; `src/__pycache__/auto_pipeline.cpython-313.pyc`
   (shared by the venv and the conda 3.13 interpreter that runs
   `streamlit`) was the pre-Phase-10 bytecode, so the app loaded an old module
   without `detect_pair`. Fix: removed all project `__pycache__` trees (incl.
   AppleDouble `._*.pyc`), and set `sys.dont_write_bytecode = True` at the top
   of `demo/app.py` and `scripts/run_auto.py` so no stale cache can ever be
   written again on the mount.

2. **`[Errno 21] Is a directory` on the "drop anything" path (and IIRS preset)**
   When a run had no reference geometry (`ref_geom=""`),
   `os.path.join(root, "")` returned the *repo root*, which was then opened as
   a file → `IsADirectoryError` masquerading as `registration_failed`. Fix:
   `run_sensor_auto` now maps empty paths to `""` (`j = join(p) if p else ""`)
   instead of `os.path.join(root, "")`; a missing reference geometry now yields
   the honest `registration_failed`/`error` note, and the IIRS preset verifies
   `cross_sensor_mismatch` with an empty ref geometry.

3. **"Automatic registration not working" — verified it works**
   Ran the exact `_execute_auto(PAIR1)` call the app makes: `registered`,
   RMSE vs reference **0.5493 px**, 123 inliers, decision
   `MATCH / content_correspondence`, elapsed 100.7 s. Also verified
   TMC→NAC (`geometry_registered`) and IIRS→NAC (`not_registered →
   cross_sensor_mismatch`) through the fixed empty-geometry logic. The "not
   working" impression was the stale cache coupled with the ~100 s runtime.

## Verification

- `ast.parse` OK on `demo/app.py` and `scripts/run_auto.py`; all `src` imports
  resolve (no stale bytecode).
- `streamlit.testing.v1.AppTest` boots the app with **0 exceptions**;
  the new radio option `Any image pair (auto-detect)` is present and default;
  the "Example pair" selectbox lists the four real presets; photometric
  preconditioning selectbox still renders.
- `detect_pair` smoke-checked on the three non-OHRC reference pairs
  (ohrc-nac / tmc-nac / iirs-nac) and `classify` on a synthetic registered
  report → MATCH with evidence. All preset file/geometry paths confirmed on disk.
- Empty-reference-geometry regression: tmc-nac without `--ref_geom` now fails
  honestly (`registration_failed`, no `Is a directory`); iirs-nac with empty
  ref geometry → `cross_sensor_mismatch`.
- Real Automatic-registration run exactly as the app fires it (PAIR1):
  `registered`, RMSE vs reference 0.5493 px, 123 inliers,
  decision `MATCH / content_correspondence`, 100.7 s.

## Test results

- Full suite: **76 passed, 0 failed** (demo touches no pipeline behaviour —
  it only renders; core unchanged).

## Functional verification

- `AppTest.from_file('demo/app.py').run()` -> radio shows the new default mode,
  0 exceptions, preset selectbox populated from `_any_presets()`.
- `venv/bin/python -c "from src.auto_pipeline import detect_pair"` passes after
  cache clearing.

## Files created/modified

- `demo/app.py` (decision banner, best-product panel, "Any image pair" mode,
  generalised `_sensor_labels`, syntax fix, `sys.dont_write_bytecode`)
- `scripts/run_auto.py` (path bootstrap + `sys.dont_write_bytecode`, dedup
  ROOT block)
- `src/auto_pipeline.py` (empty-path guard in `run_sensor_auto`: `j()` keeps
  empty reference geometry empty instead of resolving to the repo root)
- `results/logs/phase11_summary.md` (this file)