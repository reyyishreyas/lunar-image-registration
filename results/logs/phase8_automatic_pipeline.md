# Phase 8 — Fully-Automatic Registration Pipeline — Summary

**Branch:** `phase-8-automatic-pipeline`
**Date:** 2026-09-08

## Objective
Single entry point that takes the three inputs — OHRC `.img`, ISRO geometry CSV,
NAC `.IMG` — and does **everything** automatically, with no manual mode
selection. Delivered via **both** a CLI (`scripts/run_auto.py`) and a one-click
Streamlit automatic page (`demo/app.py` → "Automatic registration").

## What was built
- `src/auto_pipeline.py` — `run_auto()`:
    1. Stages OHRC<->NAC to a common ground grid (reuses `stage_pair` /
       `georeference_pair`; equal-GSD optional, Champion default off).
    2. Detects/match/robust-fits the champion config (SIFT + CLAHE +
       USAC_MAGSAC), then iteratively tightens to a sub-pixel set
       (`_lsq_homography`, `_apply_homography`, `_tighten`, `_rmse`).
    3. Automatically decides: **content registration** | **geometry-only**
       (SPICE + ISRO CSV fallback for the pair-2 photometric gap) | explicit
       non-registration. Never silently returns garbage.
    4. Deterministic (`cv2.setRNGSeed`/`np.random.seed`) and `fresh` (clears
       stale staged crops so a previous equal-GSD run can't leak its flip into
       the result).
    5. Emits one JSON report + match overlay + checkerboard.
- `scripts/run_auto.py` — CLI; accepts the three inputs (optional NAC geometry
  and ground-truth reference), writes `report.json` + figures; exit code 0 on
  a completed registration.
- `demo/app.py` — new "Automatic registration" mode (`_execute_auto`,
  `_render_auto`) that calls `src.auto_pipeline.run_auto` (no reimplemented
  logic) and renders the verdict, metrics, and figures.
- `tests/test_auto_pipeline.py` — 10 unit tests for the pure helpers.
- `AI_EXECUTION_PLAN.md` — new Section 12 (Phase 8) with completed steps.

## Numbers produced
| Case | Verdict | RMSE vs ref (px) | Self-RMSE (px) | Inliers / ratio | Matches | GSD (m/px) |
|------|---------|------------------|----------------|-----------------|---------|------------|
| Pair-1 (OHRC-2021 + M1469248775LC) | registered (content) | **0.5493** | 0.967 | 123 / 0.4301 | 286 | — |
| Pair-2 (polar OHRC-2026 + M1127547939RC) | geometry_registered (no content) | n/a | n/a | fallback | — | 60 |

Pair-1 sub-pixel (0.5493 < 1.0 px) and reproduces the champion RMSE~0.55;
pair-2 auto-detects no content and falls back to SPICE+ISRO geometry.

## Determinism / reproducibility
- Seeded RNG makes the 4-orientation flip decision reproducible (flipV for the
  champion pair-1 path, RMSE 0.5493).
- `fresh=True` deletes the prefix's staged crops before re-staging, so any
  prior equal-GSD/flip variant cannot contaminate the run (this was a real
  source of apparent RMSE drift — 0.55 vs 14.66 — from stale cached crops).

## Files created / modified
- `src/auto_pipeline.py` (new)
- `scripts/run_auto.py` (new)
- `demo/app.py` (added Automatic registration mode)
- `tests/test_auto_pipeline.py` (new, 10 tests)
- `AI_EXECUTION_PLAN.md` (new Section 12, checks marked [x])

## Testing
- `venv/bin/python -m pytest tests/ -q` → **34 passed** (24 + 10 new).
- CLI pair-1 end-to-end: `REGISTERED (content): RMSE=0.5493 px` ✓
- CLI pair-2 end-to-end: `GEOMETRY REGISTERED (no content): GSD=60 m` ✓
- Demo app `py_compile` clean.

## Git
- Branch: `phase-8-automatic-pipeline`
- Ready for merge.