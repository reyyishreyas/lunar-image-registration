# Phase 9 Summary — Cross-modal multi-sensor coverage (PS 26166: OHRC + TMC + IIRS)

Date: 2026-09-08
Branch: `phase-9-cross-modal-tmc-iirs`
Base: `main` (= `phase8-ok` at `5a599f0`)

## Objective
PS 26166 names three Chandrayaan-2 sensors — OHRC, TMC, IIRS. Phase 8 implemented
only the OHRC↔NAC auto path. This phase adds the TMC and IIRS registration paths
plus a cross-modal (radiometric-invariance) matcher to bridge the large radiometric
gaps these sensors present, **without disturbing the working Phase 8 code**
(rollback: git tag `phase8-ok` + fs backup).

## Deliverables
| File | Purpose |
|------|---------|
| `src/detection/cross_modal.py` | Cross-modal matcher (front-ends: gradient_structure, histogram_match, log_ratio, clahe, none; proof-based robust acceptance) |
| `src/preprocessing/tmc.py` | CH2 raw reader (width auto-inferred, memmap), geometry grid, GSD, `register_tmc_to_ohrc` |
| `src/detection/iirs.py` | ENVI BSQ band-selective reader, representative band selection, `register_iirs_to_ohrc` |
| `src/auto_pipeline.py` | `run_sensor_auto` dispatch (`ohrc-nac`/`tmc-ohrc`/`iirs-ohrc`) |

### Follow-up fix (9.6–9.8): dark/polar handling
- `src/preprocessing/ch2_staging.py` (new): ground-grid equal-GSD stager for two
  CH2 geometry CSVs + layered `register_ch2_pair` decision
  (enhance → content → geometry → not_registered) + `low_contrast_score`.
- `tests/test_ch2_staging.py` (new, 9 tests).
- `src/detection/iirs.py`: enhancement + low-contrast diagnostics + honest
  no-geometry note on every band.
- `src/auto_pipeline.py` / `scripts/run_auto.py`: TMC path now uses the robust
  layered `register_ch2_pair`; `summarize` handles `gsd_m`/`staged_gsd_m`.
| `scripts/run_auto.py` | `--sensor` flag |
| `tests/test_cross_modal.py`, `tests/test_tmc.py`, `tests/test_iirs.py` | 21 new unit tests |

## Numbers Produced
- Full test suite: **64 passed** (34 prior + 21 + 9 dark/polar).
- TMC width inference: 4000 px (real product, 551946 × 4000, 2.2 GB).
- OHRC width inference preserved: 12000 px (both 2021 pair-1 and 2026 PATCH-004).
- TMC GSD: 21.70 m/px (stride-aware); OHRC-2026 GSD 0.982 m/px, OHRC-2021 GSD 0.25 m/px.
- Real TMC swath self-registration: front=clahe, inliers=1314, ratio=0.996, RMSE=0.253 px.
- Real IIRS band 116 read (band-selective, near-instant); self-registration: front=clahe, inliers=794, ratio=0.994, RMSE=0.252 px.
- Content path regression (bright pair): `register_ch2_pair` → `registered`, content:gradient_structure, 2170 inliers, RMSE 0.0 px.
- **Real overlap found**: TMC-2026 NCA/NCF/NCN all overlap OHRC-2026 (lon 296.09–296.21, lat 7.32–8.16).

## Dark / polar (low-sun) handling
- **OHRC-2026 is globally dark**: mean ~4–8/255, max ~28–205, p98~19 across all rows → a genuine low-sun hard case. After percentile-stretch + CLAHE, SIFT still finds 0 matches vs TMC (no recoverable structure) → content registration is mathematically infeasible on this data.
- **Proper pipeline** (`register_ch2_pair` layered decision): enhance → content → if content fails AND footprints overlap → **geometry registration on a common ground grid**; if no overlap → honest `not_registered`. Never fabricates a model.
- **TMC↔OHRC-2026 result**: `GEOMETRY REGISTERED (no content): GSD=21.70 m, method=geometry`, aligned src/ref ground crops written, exit 0. UI/CLI report notes "content correspondence NOT verifiable (dark/low-sun/polar or featureless)".
- **IIRS**: no ISRO geometry CSV / ENVI map-info → geometry registration impossible; content-only per-band with enhancement + low-contrast diagnostics + explicit no-geometry note. IIRS-2021↔OHRC-2021 real attempt → honest `not_registered` (no fabricated model).

## Blocked / Not fully verified
- **IIRS content cross-sensor registration**: IIRS is IR hyperspectral with no
  ground geometry; only a hypothetical bright overlapping IIRS↔OHRC acquisition
  could yield a content `registered`. Currently honest `not_registered`.
- **Content TMC↔OHRC on this data**: not achievable for the OHRC-2026 pair
  (globally dark); handled via geometry fallback instead (verified on real data).

## Testing / Functional verification
- `venv/bin/python -m pytest tests/ -q` → **64 passed**.
- Real TMC: memmap reader shape (551946, 4000); swath read in 0.15 s; cross-modal self-register OK.
- Real IIRS: header parse + band-116 read; cross-modal self-register OK.
- Real TMC↔OHRC-2026 (dark): `--sensor tmc-ohrc` → geometry_registered, GSD 21.70 m, exit 0.
- Real IIRS-2021↔OHRC-2021: `register_iirs_to_ohrc` → honest not_registered with no-geometry note.
- CLI: `scripts/run_auto.py --help` shows all three `--sensor` choices; bogus sensor → `input_error`.

## Git
- Branch: `phase-9-cross-modal-tmc-iirs`
- Rollback anchor (untouched): git tag `phase8-ok` at `5a599f0`, pushed.
- Status after this summary: commit + push pending (completed at end of phase).

## Streamlit multi-sensor UI (Phase 9)

- demo/app.py: added "Multi-sensor registration (TMC / IIRS)" one-click mode.
- Two disambiguooted presets: TMC-2026 <-> OHRC-2026 (dark/low-sun, geometry
  fallback) and IIRS-2021 <-> OHRC-2021 (content-only, honest verdict).
- Auto-extracts the TMC .img/.csv from its zip on first run.
- Friendly verdict banners (registered / geometry-registered / not-registered),
  plain-language explanations for dark/low-sun geometry fallback, footprint +
  image-quality captions, ground images, expandable technical notes.
- Uses run_sensor_auto / run_auto / run_experiment (no logic duplicated).
- Verified: AppTest one-click TMC flow -> "Registered by geometry", GSD 21.70 m/px,
  zero exceptions, zero deprecation warnings; suite 64 passed.

## PS-26166 review follow-up

- Fact-checked an external PS review against repo state:
  - "Sub-pixel accuracy not achieved" -> STALE: Phase 7 achieved RMSE 0.5968 px
    (held-out) / 0.6971 px (end-to-end) on pair-1 GT v2.
  - "Uniform match distribution not implemented" -> STALE: grid_uniform.py
    cap_by_grid (Phase 4, C9).
  - "22.66 px" -> correct, but that is the Phase 1 C1 classical baseline,
    superseded by Phase 4-7 pipeline (idem 22.6172 on the old noisy GT whose
    noise floor is ~15 px).
- Verified "suspicious" displayed images are real data, not decode bugs:
  TMC-2026 has native periodic column striping (~3.5 px period, amplified on a
  154 px wide overlap crop); OHRC-2026 reference mean 4.9/255 = genuinely dark
  low-sun frame.
- UI now renders an expandable evidence trace per result: source instrument,
  reference instrument, native GSDs, common-grid GSD, preprocessing,
  matcher/transformation, accuracy, verdict. (demo/app.py _render_pipeline_trace)

## Before -> After previews in the app (UI evidence)

- ch2_staging.stage_ch2_ground_pair now also writes native raw-strip "before"
  previews (overlap window, bin-downsampled <=640px wide/1600px tall,
  display-stretched) as original_src/original_ref artifacts; register_ch2_pair
  and register_iirs_to_ohrc attach them to the report.
- demo/app._render_before_after shows Before (raw strips) vs After (common-grid
  registered pair) with instrument names and adaptive captions
  (registered/geometry_registered -> "registered on the same ground grid";
  not_registered -> "processed working crops (enhanced)").
- Verified: TMC pair -> original_src 1290x103 / original_ref 1587x203 valid PNGs;
  IIRS pair artifacts valid; suite 64 passed; AppTest renders before/after.

## Highlighted before/after (visible change markers)

Users found the before/after panels too similar (both are display-stretched
re-projections). Added explicit red highlight markers:
- _make_original_preview now builds a CONTEXT window around the overlap swath
  and returns the inner-overlap box in preview coords.
- _draw_highlight_box draws a red rectangle on each raw strip marking the exact
  swath extracted; _outline_registered draws a red outline on the registered
  product (ch2_staging + iirs emit before_*/after_* PNGs).
- demo/app._render_before_after uses these and adds a caption explaining "red
  box = overlap swath extracted; red outline = the same data re-projected on
  the common grid".
- New test test_original_preview_box_and_highlights; suite 65 passed; AppTest
  renders the legend.

## Visible, labeled before/after (v3 — "red boxes STILL not visible" fix)

Root causes found and fixed:
- Panels were >10:1 tall slivers; on screen the frame was far off-screen / an
  invisible sliver. _make_original_preview now decimates along-track MORE than
  across-track (per-axis integer binning; uniform binning preserves aspect) so
  every preview is a <=3.5:1 wide panel.
- Labels: every Before panel now carries a red 'OVERLAP SWATH' box + text and a
  hard-dimmed context; After panels carry a red 'REGISTERED' outline + tag.
- New 'what changed' proof: a per-pixel |src-ref| difference map (TURBO
  colormap) written as change_map, and the After pair + difference are stacked
  into one wide 2.2:1 montage so nothing renders as a sliver.
- Verified: TMC/OHRC-2026 real run -> before panel aspect 2.7/2.5 (was 16/7.8),
  ~45-46k red marker pixels per Before, montage 1024x462 with 75k red px; suite
  65 passed; AppTest renders montage branch (else-branch marker absent) with no
  exceptions. register test now asserts all six evidence artifacts exist.

## Guided, verifiable results view (v4 — "confusing, users can't verify")

Reordered the sensor results page into a narrative a first-time user can follow:
1. Verdict banner
2. 'What the pipeline did' — numbered plain-language steps matched to the actual
   verdict (content-registered vs geometry-registered vs not_registered), with
   the key numbers inline (RMSE, inliers, GSD).
3. Before (red-boxed overlap swath) -> After montage (After|After|Difference).
4. 'Before vs after — by the numbers' table: native strip sizes, the exact
   red-box swath in native pixels, and the common-grid size + GSD.
5. 'How to check it''s correct' — a short per-verdict check script (eyeball the
   After panels at equal zoom; check RMSE under 1 px; or for geometry pairs why
   no RMSE exists).
Evidence trace + technical notes collapse into expanders.
The stage now records dimensions in the report
(native_src/native_ref, swath_src/swath_ref, grid_along/across, gsd_m) — IIRS
too. Real TMC/OHRC-2026: TMC 551,946x4,000 px, OHRC 93,692x12,000 px, TMC swath
rows 63,406-68,557 x cols 3,590-4,113, common grid 1,024x154 @ 21.70 m/px.
Suite 65 passed; AppTest renders all four sections.

## App-wide red/change-highlight frames (v5) + PS-26166 compliance

Applied red-box / change-highlight markings to **every** image surface in the
Streamlit app (not just the sensor panels), per the user directive.

- New `src/visuals.py`: `RED`, `frame_img(tag)` (red outline + tag),
  `highlight_box(box, tag)` (red box + dimmed context), `diff_map(a, b)`
  (Turbo |src−ref| per-pixel difference). ch2_staging now delegates its
  underline/outline drawing to these shared helpers (`RECT_COLOR` removed).
- Inline pair-1 / pair-2 / upload results in `main()` now render a **framed**
  matches figure (`MATCHES`), a **framed** checkerboard (`ALIGNED`) and a new
  `what changed` difference map. `_execute` computes `diff` via `_diff_from_paths`
  for all three inline branches (pair-1, pair-2 content, upload).
- `_render_phase5` orthos + overlay framed with `OHRC ORTHO` / `NAC ORTHO` /
  `50/50 OVERLAY` tags.
- Auto mode (`_render_auto`) and ground artifacts (`_render_ground_artifacts`)
  were already framed in the previous step (delivered-products framing).
- New `tests/test_visuals.py` (5 tests): red border pixels present, box dims
  interior vs context, identical frames -> near-dark diff, different frames ->
  bright diff, shape-mismatch -> None. Suite **70 passed**.
- AppTest: pair-1, TMC sensor, and auto modes all run with zero exceptions and
  metrics rendered; pixel-level check confirms framed fig/checker + 3-channel
  diff file all present on disk.

### PS-26166 compliance mapping

`results/logs/ps26166_compliance.md` maps every requirement (multi-modal
OHRC/TMC/IIRS, sun-angle invariance, scale invariance, sub-pixel accuracy,
uniform match distribution, registered product, corresponding match points,
evaluation metrics) to implementation + evidence. Honest limits documented:
dark/low-sun pairs report geometry_registered (no fabricated content alignment),
IIRS ships no geometry CSV so it stays content-only, and the reposited reference
is LRO NAC (SELENE supported conceptually but not present in local `data/`).
