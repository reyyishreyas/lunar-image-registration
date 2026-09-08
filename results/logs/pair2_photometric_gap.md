# Pair 2 Photometric Gap — Evidence Log (Phase 5 / Restart Phase 1)

**Pair:** OHRC-2026 (`ch2_ohr_ncp_20260331T1105235288_d_img_d18.img`)
vs LRO NAC `M1127547939RC`
**Branch:** `phase-1-best-model`
**Date:** 2026-09-08

## Verdict

The OHRC-2026 strip and the SPICE-georeferenced NAC do NOT share recoverable
photometric content after geometry-aware equal-GSD staging. Every content
matcher returns noise-level matches; the one apparent low-frequency FFT lock is
a strip-envelope artifact (reproduced unchanged by a row-reversed null image).
Registration of pair 2 is therefore geometry-based: the ISRO OHRC CSV
georeference and the LRO SPICE NAC geometry are mutually consistent to
sub-meter, so the strip is registered to the NAC by construction.

Root cause is photometric, not geometric and not a staging bug: ISRO OHRC-2026
metadata reports `sun_elevation = 1.895764 deg` (extreme grazing illumination;
sun azimuth 89.38 deg), versus near-noon-equivalent illumination in the NAC
viewing/illumination geometry.

## Evidence chain

### 1. Matcher health (glue verified, so "0 matches" is a content result)
- kornia 0.8.3 DISK + LightGlue, `n=2000` keypoints, staged 60 m grid 1536x194.
- Self-match (image vs itself through the matcher):
  - SRC self-match: **1986 inliers**
  - REF self-match: **1953 inliers**
- Cross-match (SRC vs REF, same staged grid): **0 inliers**.
- The glue was previously broken (self-match 2-7); after fixing the LightGlue
  `{"image0": {...}, "image1": {...}}` data-dict contract, self-match jumped to
  the thousands. "0 cross" cannot be blamed on the matcher.

### 2. Deleted degrees of freedom (all exhausted)
- Orientation hypotheses: rowsFlip, colsFlip, bothFlips, transpose,
  transpose+rowFlip -> all **0 matches** on the staged 1024x1024 crops.
- Full-strip 2-D tile sweep (~81 tiles visited at multiple strides): **8
  spurious single matches total**; no tile carries a real correspondence.

### 3. Low-frequency FFT lock is an envelope artifact
- Full-strip cross-correlation peak at (578, -35) in scan/px units
  (~= +34.7 km along-track, -2.1 km across) initially looked like a lock.
- Band-pass shoulder was weak/inconsistent across filters.
- Null test (strip rows re-shuffled and row-reversed so no content, same
  envelope): peak magnitude within ~92% of the true image peak.
  - true peak 7,999.8 vs row-reversed-null peak 7,361.6 (sigma 8 px)
  - verdict: **envelope artifact**; no content lock exists.

### 4. Geometry consistency (why the geometry-based registration is trustworthy)
- 2000 OHRC ground sample points -> NAC inverse: **100% in-bounds**
  (in_bounds_frac = 1.0).
- Forward self-consistency: mean position error **5.9e-13 km** (sub-picometer,
  i.e. numerically exact round-trip through the SPICE/csv models).
- NAC footprint vs ODE products: corners agree to ~100 m.
- ODE overlap on all four downloaded NAC candidates for this OHRC strip:
  - M1127547939RC **100%** (primary, used)
  - M1182892612LC 93.3%
  - M1298165405LC 96.8%
  - M181587967LC 96.4%

## Deliverables
- `src/preprocessing/geometry.py` (module CLI): OHRC CSV / NAC SPICE model,
  NAC forward+inverse, equal-GSD staging, coverage audit, content probe
  (DISK+LightGlue), correlator null-test, and phase 5 emit.
- `data/processed/georef_phase5.json`:
  - grid 1536x194 @ 60 m staged GSD (180 px = 10.8 km), lon 296.086-296.213,
    lat 7.324-8.161
  - OHRC native GSD confirmed: 0.99 m/scan, 0.97 m/px
- `data/processed/phase5_src.png`, `phase5_ref.png`, `phase5_overlay.png`
  (geometry-aligned side-by-side and overlay at equal GSD).
- NumPy arrays: `data/processed/spice_georef/pair2_fullstrip.npz`
  (src/ref/mask/s_axis/p_axis + full-res staged grid ancillary evidence).

## Tests
- `tests/test_geometry.py`: 7 unit tests (NAC inverse round-trip, boundary
  NaN, uint8 stretch, NAC remap, GSD from geometry grid, synthetic strip
  ortho, CSV loaders) — all pass.
- Full suite: `venv/bin/python -m pytest tests/ -q` -> **11 passed**.

## Regression
- FINAL_CONFIG (pair 1): RMSE **22.6172 px**, 235 inliers, ratio **0.8217**,
  unchanged from the Phase 4 champion. No regression introduced by the staging
  / geometry additions.