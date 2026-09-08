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

## Verification

- `ast.parse(demo/app.py)` OK; module imports clean with all 76 tests passing.
- `streamlit.testing.v1.AppTest` boots the app with **0 exceptions**;
  the new radio option `Any image pair (auto-detect)` is present and default;
  the "Example pair" selectbox lists the four real presets; photometric
  preconditioning selectbox still renders.
- `detect_pair` smoke-checked on the three non-OHRC reference pairs
  (ohrc-nac / tmc-nac / iirs-nac) and `classify` on a synthetic registered
  report → MATCH with evidence. All preset file/geometry paths confirmed on disk.

## Test results

- Full suite: **76 passed, 0 failed** (demo touches no pipeline behaviour —
  it only renders; core unchanged).

## Functional verification

- `AppTest.from_file('demo/app.py').run()` -> radio shows the new default mode,
  0 exceptions, preset selectbox populated from `_any_presets()`.

## Files created/modified

- `demo/app.py` (decision banner, best-product panel, "Any image pair" mode,
  generalised `_sensor_labels`, syntax fix)
- `results/logs/phase11_summary.md` (this file)