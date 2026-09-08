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
