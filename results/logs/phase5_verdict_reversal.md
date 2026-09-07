# Phase 5 Verdict REVERSED — OHRC-2026 DOES overlap its NAC (ODE evidence)

Date: 2026-09-08
Branch: phase-1-best-model

## TL;DR
The Phase 5 / Phase 6 conclusion "PATCH-004 OHRC-2026 vs M1127547939RC do not
overlap" was **WRONG**. Independent authoritative geometry from the NASA ODE
REST API shows the single NAC **covers 100 % of the OHRC-2026 footprint**.
The pair genuinely overlaps; our *feature-based search* failed to discover the
georeference (extreme cross-sensor illumination contrast + scale mismatch), and
we wrongly concluded geographic non-overlap.

## Evidence

### 1. ODE footprint search (get_search_results.py, results=fmpc)
Queried `https://oderest.rsl.wustl.edu/live2/` with our local
`ch2_ohr_ncp_20260331T1105235288_d_img_d18.xml` corner footprint
(lat 7.3241-8.1615, lon 296.0864-296.2134).

Top candidates (Overlap = intersection area, % cover = fraction of OHRC):
```
 1 nac.m1127547939rc   35.9km²  100.0%  inc 66.95°  2013-07-04   <- our phase-5 NAC
 2 nac.m1298165405lc   34.8km²   96.8%  inc 22.13°  2018-11-29
 3 nac.m181587967lc    34.6km²   96.4%  inc 61.45°  2012-01-19
 4 nac.m1182892612lc   33.5km²   93.3%  inc 46.76°  2015-04-05
 ...
11 nac.m1504651871lc   25.3km²   70.6%  inc  8.17°  2025-06-15
17 nac.m1420286205lc   20.6km²   57.2%  inc 27.96°  2022-10-13   <- on disk already
20 nac.m1394475435lc   18.9km²   52.7%  inc 70.43°  2021-12-18   <- on disk already
```

### 2. Exact footprint polygons
- M1127547939RC: POLYGON ((296.2 8.27, 296.27 6.37, 296.1 6.37, 296.02 8.26, 296.2 8.27))
  -> lon 296.02-296.27, lat 6.37-8.27. The OHRC strip (lon 296.0864-296.2134,
  lat 7.3241-8.1615) lies fully inside it.
- M1504651871LC (lowest incidence 8.17°): POLYGON ((295.91 6.27, 296.1 7.96, 296.25 7.94, 296.06 6.25, ...))
  -> overlaps too.

## Why phase-5/6 matching failed anyway (feature-level)
- The 4-orientation SIFT search maxed at 8-16 inliers (NCC ~ 0.02-0.05).
- Scale was NOT the dominant cause: produced same flat result for nac_fac 3-8.
- Root cause = radiometric contrast: OHRC-2026 sun elevation ~1.9° (grazing
  light, almost all terrain in shadow) vs NAC incidence 66.95° (sun elevation
  ~23°). Same terrain, opposite shadow geometry -> SIFT/LoFTR/SuperGlue
  descriptors do not agree. This is a true multi-modal registration problem.

## What the ODE data unlocks
- A real second overlapping pair now EXISTS (OHRC-2026 X M1127547939RC, plus
  96-100 % candidates nac.m1298165405lc / nac.m181587967lc).
- The right "proper measure" for the best model is a georeference that does
  NOT depend on descriptor matching alone:
  1. equal-GSD resampling (both strips onto a common ground grid),
  2. photometric preconditioning (ridge/edge/phase-congruency domain),
  3. footprint-initialized warp (native NAC geometry when available; the ODE
     4-corner bilinear model is too coarse (~ 1-2 km) for fine init alone).

## Status
- Branch `phase-1-best-model` carries this finding + the working ODE tool
  (dataset_download_pipeline/get_search_results.py) + equal-GSD warp scaffold
  (scripts/footprint_warp.py).
- Prior phase-5/6 summaries (polar_case_findings.md, final_pipeline_report.md)
  recorded only the matching-failure interpretation; this doc supersedes their
  "no overlap" conclusion.