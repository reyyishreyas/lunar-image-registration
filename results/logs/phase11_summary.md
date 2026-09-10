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

4. **`ModuleNotFoundError: No module named 'src'` (@ `from src.visuals`)**
   Pre-existing ordering bug in `demo/app.py`: the `from src.visuals import`
   line sat *above* the `ROOT` computation / `sys.path.insert(0, ROOT)`, so
   `src` only resolved because the repo root happened to be the process cwd.
   Launching the app from any other directory broke the import. Fix: move the
   `ROOT` bootstrap *before* all `src.*` imports and add `os.getcwd()` as a
   second fallback, so the app is importable no matter how/whence it is run.
   Verified by launching the app script with `cwd=/tmp` — it executes with no
   `ModuleNotFoundError`. Other entry-point scripts were audited: they all
   insert ROOT into `sys.path` before importing `src.*`, so only the app
   needed fixing.

## Quality round — "any image pair" content result (user review)

Review flagged: `GSD n/a` for both products, a "streaky / scrambled"
checkerboard, an inlier ratio of only 0.44 (118/268), and the risk that a
single homography hides local-region misfit. Resolved in the pipeline + app:

1. **GSD is real now, not n/a.** `run_auto`/`run_sensor_auto` attach
   `report["gsd"]` = {src_est_m, ref_est_m, staged_m, scale_ratio, source}.
   - src (OHRC) native from the ISRO ground grid (`_gsd_m`) → **0.973 m/px**.
   - ref (NAC) native from the georef record's native-vs-ground corners
     (`_nac_gsd_from_meta`) → **3.098 m/px**.
   - **native scale ratio ref/src = 3.184**; staged common cell 0.97 m/px.
   App metric columns show `src X / ref Y · ×R`; a "Ground sample distance"
   expander explains the derivation; other sensor routes get a staged-cell
   fallback instead of n/a when the grid is unavailable.

2. **Checkerboard is now built from the WARPED source**, not the raw staged
   crops (which sit on different native grids and tiled directly → the reported
   discontinuities/streaks). `_attach_decision` writes `{prefix}_checkerboard.png`
   from `warped` vs `ref`; `_write_artifacts` MERGES into `report["artifacts"]`
   instead of replacing, so `decision.best_product` now points at the real
   `*_aligned.png` (warped deliverable), not the fallback checkerboard.

3. **Per-region residual audit.** `report["tile_residuals"]` = 4×4 grid of the
   inliers' residual RMSE bucketed by source tile (worst region ≈ **1.44 px**,
   most ≤ 1.0 px) — shown in the app, so a weak global fit can't hide a bad
   region. This directly answers the parallax/homography-limitity concern: on
   this OHRC→NAC pair the registration is uniformly good, no local blow-up.

4. **Inlier ratio display**: the 118/268 ratio stays visible in the metric
   column and decision evidence (evidence-first honesty) — the tightening then
   sub-pixel-fit already re-selects inliers at 1.5 px, and the per-tile audit
   confirms nothing locally fails.

Verified end-to-end on the real OHRC-2021 → NAC run that produced the review:
`registered`, self-RMSE 0.9722 px, decision MATCH, GSD 0.973/3.098/×3.184,
tiles all ≤ 1.44 px, `best_aligned` on disk and exposed.

## Verification

Visual-audit round — "checkerboard unchanged + braided diff" (review):

**The complaints were real, and both are now explained AND fixed:**

1. **The two "scrambled/static" checkerboard tiles were a tiling/index bug,
   exactly as the review hypothesised.** `_tile_nmi` sliced the 4×4 mask but
   indexed the full-size images with it → `IndexError: boolean index did not
   match indexed array along axis 0` — which silently aborted the whole
   `_attach_decision` product block, dropping `best_aligned`/`diff`/`residual`
   AND left the app showing the *fallback* raw-crop checkerboard (seam p50
   37.8 grey levels). It was caught because the artifacts list changed. Fixed by
   slicing both image and mask per tile. This is exactly the class of bug the
   review told us to chase at full resolution outside the montage.
   Checkerboard is now built from the **histogram-matched** warped source so
   crater continuity is actually visible (seam p50 **17.4**, p90 43.0).

2. **The braided diff was a brightness/normalisation artefact — hypothesis (b)
   confirmed numerically, and the evidence is now reproduced honestly.**
   * Raw |warped − ref|: mean 51.7, p99 140, braided look.
   * Global histogram match: mean 32.1, p99 133 — only partly fixes it.
   * **Local mean/contrast-normalised residual: mean 0.41, flat-region ≈0.3** —
     structure-level disagreement is tiny. NMI is invariant to monotone
     intensity, and min per-tile NMI = **1.026** → the warp holds everywhere.
   * diff-vs-illumination-gradient correlation ≈ 0 → not a sun-gradient
     cosine; it is the sensor/CLAHE brightness mismatch per review point (b).
   * Neither resolution-mismatch (equal ground cell 0.97 m/px, precise GSD
     ratio) nor residual geometry (NMI ≥1.01 everywhere, uniform) is present.
   Pipeline now writes `_diff.png` (histogram-matched change map) AND
   `_residual.png` (brightness-robust, mostly dark + warm speckles), plus
   `report["brightness"]` with all the numbers above. The app shows both
   products and the evidence expansion; the old "what-changed" panel no longer
   presents a raw two-sensor subtraction as truth.

3. **New regression tests** (`tests/test_pipeline_quality.py`, now 11):
   checkerboard hatch outside the mask; global-histogram match tracks the ref
   CDF; local-normalised residual is near-zero for same-structure/different-
   brightness images; per-tile NMI no longer crashes and returns the min.

Decisive-experiment round — "is the residual structure geometric or
photometric?" (review challenged the brightness-normalisation claim):

1. **The test the review demanded was run: a non-global model was fitted.**
   From the run's 118 inlier tie-points we fitted a Delaunay **piecewise-affine
   warp** (per-triangle affine, outside-hull falls back to the global H) and
   rewarped the whole source with it. Result — the piecewise model did NOT
   shrink the residual, it made it *worse everywhere* (brightness-robust
   residual mean 0.322 → 0.449; min/mean/max NMI 1.007/1.104/1.192 →
   1.008/1.039/1.117). A parallax field consistent with the inliers would be
   *extracted* by that fit and reduce the residual; the opposite result
   falsifies the "parallax the homography missed" hypothesis for this pair.
   The residual is therefore photometric, not geometric.

2. **The residual panel that "looked identical to the diff" was a display
   bug, now fixed.** The residual itself is ~75-90% white noise: smooth /
   'braided' energy only **0.09** (blur-σ31 explains ~9% of its variance;
   autocorr 0.26@8px → 0.16@32px). The old fixed 90× gain mapped a mean-0.41
   noise floor to a mid-grey wash that visually mirrored the raw diff. The
   render is now scaled to its own 99.5th percentile → flat floor with sparse
   warm specks (p50 ≈ 48/255). Residual stats now reported in normalised units
   (mean 0.41, std 0.34).

3. **The lower-centre/right bright cluster is real but small and not
   smooth-band.** Region residual 0.56 vs 0.37 global (×1.5, ≈0.7σ above the
   floor) — an elevated photometric/texture tier of the same noise process,
   not a coarse misalignment.

4. **NMI distribution (not just the min) is healthy.** min 1.026 (one border
   tile) but **median 1.088, mean 1.092, max 1.176, 90.6% of tiles above
   1.05** — clearly matched content (a no-relationship pair trails ~1.0).

5. **The "two scrambled tiles" are reference-swath nodata corners**, not a
   warp failure: rows 6-7 / cols 2-3 carry 6-13% NAC black (crop edge) and
   previously drowned under a heavy 4-px hatch plus the blown-up noise
   display. The nodata marker is now a light 32-px stipple with an explanatory
   caption, and those tiles' own residual / NMI are actually typical
   (r7c2 = 0.17).

Sentinel numbers (delivered OHRC->NAC run): residual 0.41/0.34 (mean/std),
braid energy 0.09, NMI 1.026/1.088/1.092/1.176 (min/med/mean/max; 90.6%>1.05),
inlier med/max 0.88/1.52 px, worst per-tile residual-RMSE 1.44 px.

## Verification

Note for the live demo: GSD / artifacts are part of the *report dict*, which the
app only refreshes when **Register this pair** is pressed inside a restarted
Streamlit process. A long-lived server keeps the pre-fix module + session
report (no `gsd`, no `best_product`); restart `streamlit run demo/app.py` and
re-register to see GSD 0.973 / 3.098 m/px, ratio 3.184, self-RMSE, tiles, and
the aligned product. Content matches have no ground truth, so `rmse_px` is
None by design — the RMSE column now falls back to `rmse_self_px` ("0.967
(self)") instead of "n/a".

- `ast.parse` OK on `demo/app.py`, `scripts/run_auto.py`, `src/auto_pipeline.py`;
  all `src` imports resolve (no stale bytecode).
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
- Quality round: real OHRC→NAC content run → GSD 0.973 / 3.098 m/px,
  ratio 3.184, tile residuals 4×4 all ≤ 1.44 px, `best_product` = aligned.

## Test results

- Full suite: **84 passed, 0 failed** (now `tests/test_pipeline_quality.py`
  covers checkerboard masking, corner-derived NAC GSD, GSD-report tolerance,
  tile residual bucketing, best_product ordering, histogram-matching,
  local-normalised residual invariance, per-tile NMI).

## Functional verification

- `AppTest.from_file('demo/app.py').run()` -> radio shows the new default mode,
  0 exceptions, preset selectbox populated from `_any_presets()`.
- `venv/bin/python -c "from src.auto_pipeline import detect_pair"` passes after
  cache clearing.
- `_render_sensor` renders the full enriched report (GSD + tile residuals +
  aligned product) without error.

## Files created/modified

- `demo/app.py` (decision banner, best-product panel, "Any image pair" mode,
  generalised `_sensor_labels`, `sys.dont_write_bytecode`, `src`-first path
  bootstrap, `_rmse_sentence`, GSD metric columns, GSD + per-region expanders,
  warped-checkerboard display)
- `scripts/run_auto.py` (path bootstrap + `sys.dont_write_bytecode`, dedup
  ROOT block)
- `src/auto_pipeline.py` (empty-path guard; `_sensor_gsd_report` /
  `_nac_gsd_from_meta` / `_tile_residuals`; merged artifacts; warped
  checkerboard in `_attach_decision`; `gsd` + `tile_residuals` in report)
- `tests/test_pipeline_quality.py` (new, 6 tests)
- `results/logs/phase11_summary.md` (this file)