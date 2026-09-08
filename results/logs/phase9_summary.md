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
| `scripts/run_auto.py` | `--sensor` flag |
| `tests/test_cross_modal.py`, `tests/test_tmc.py`, `tests/test_iirs.py` | 21 new unit tests |

## Numbers Produced
- Full test suite: **55 passed** (34 prior + 21 new).
- TMC width inference: 4000 px (real product, 551946 × 4000, 2.2 GB).
- OHRC width inference preserved: 12000 px.
- TMC GSD: 21.70 m/px (stride-aware; fixes earlier 6×/missed-stride bug).
- Real TMC swath self-registration: front=clahe, inliers=1314, ratio=0.996, RMSE=0.253 px, recovered offset to 0.02 px.
- Real IIRS-2024 header parsed: samples=250, lines=12620, bands=256, int16, BSQ.
- Real IIRS band 116 read (band-selective, near-instant); self-registration: front=clahe, inliers=794, ratio=0.994, RMSE=0.252 px, recovered offset to 0.03 px.

## Cross-modal validation
- Synthetic radiometric inversion: `best_front_name` = `gradient_structure`, ~467
  inliers, RMSE 0.218 px, homography recovered within 0.01/1.0 px of truth.

## Blocked / Not fully verified
- **Content TMC↔OHRC and IIRS↔OHRC registration on overlapping acquisitions:**
  the available TMC (2026-06-07, lon 294.7–296.8, lat −26.5 to 18.5) and IIRS
  (2024-05-03) products do **not overlap** the OHRC pair-1 footprint (lon 336.5,
  lat −3.4 to −2.6). A content run against that reference would legitimately fail
  at ground level (no shared terrain), not from a coding defect. Verified the two
  readers + the cross-modal chain on the real products via self-registration
  instead; a true cross-sensor run requires an overlapping acquisition (new data).

## Testing / Functional verification
- `venv/bin/python -m pytest tests/ -q` → 55 passed.
- Real TMC: memmap reader shape (551946, 4000); swath read in 0.15 s; cross-modal self-register OK.
- Real IIRS: header parse + band-116 read; cross-modal self-register OK.
- CLI: `scripts/run_auto.py --help` shows all three `--sensor` choices; bogus
  sensor → verdict `input_error`.

## Git
- Branch: `phase-9-cross-modal-tmc-iirs`
- Rollback anchor (untouched): git tag `phase8-ok` at `5a599f0`, pushed.
- Status after this summary: commit + push pending (completed at end of phase).
