# Phase 0 (Best-Model Restart) Summary

## 1. Phase number and name
Best-Model Restart — Phase 0: Data acquisition, ODE tooling, and overlap audit
(branch `phase-1-best-model`, created off main with Phase 6 merged in).

## 2. Steps completed
- [x] Create restart branch off main (+ merged `phase-6-polar-footprint-registration`).
  RESULT: pass — branch `phase-1-best-model` = main(13b02f9) + b3e110f.
- [x] Port ODE search/download tool from `origin/download-pipeline`.
  RESULT: pass — `dataset_download_pipeline/get_search_results.py` runs live
  (requests + shapely; shapely 2.1.2 installed; `requests`+`shapely` added to
  requirements.txt).
- [x] ODE audit of the PATCH-004 pair.
  RESULT: pass — **M1127547939RC covers 100 % of the OHRC-2026 footprint**
  (35.9 km²; lat 6.37-8.27, lon 296.02-296.27). The Phase 5/6 "no overlap"
  verdict is REVERSED; it was a descriptor-search failure under extreme
  illumination contrast, not a geometric non-overlap.
- [x] Download extra high-overlap NAC candidates.
  RESULT: pass — M1127547939RC (100 %), M1298165405LC (96.8 %, inc 22.13°),
  M181587967LC (96.4 %, inc 61.45°), M1182892612LC (93.3 %, inc 46.76°);
  4 x 528.9 MB downloaded to data/PATCH-004/LRO NAC/ode_downloads/... .
- [x] Hardware audit.
  RESULT: pass — torch MPS available and built (Apple Silicon GPU usable for
  learned matchers); CPU fallback intact.

## 3. Numbers produced this phase
| Product | Overlap (km²) | OHRC cover % | Incidence° |
|---|---|---|---|
| M1127547939RC | 35.9 | 100.0 | 66.95 |
| M1298165405LC | 34.8 | 96.8 | 22.13 |
| M181587967LC | 34.6 | 96.4 | 61.45 |
| M1182892612LC | 33.5 | 93.3 | 46.76 |
| (ref) M1504651871LC | 25.3 | 70.6 | 8.17 |

Key ODE footprints (WKT):
- M1127547939RC: POLYGON ((296.2 8.27, 296.27 6.37, 296.1 6.37, 296.02 8.26, 296.2 8.27))
- OHRC-2026 (XML Refined_Corner_Coordinates): lat 7.3241-8.1615, lon 296.0864-296.2134  -> fully inside the NAC.

Database: 4 NAC .IMG files x 528,929,736 B (2.0 GB) staged under data/.

## 4. Files created or modified this phase
- dataset_download_pipeline/get_search_results.py (ported, working)
- scripts/footprint_warp.py (equal-GSD footprint-warp scaffold, WIP)
- requirements.txt (+ requests, + shapely)
- results/logs/phase5_verdict_reversal.md (new)
- results/logs/phase0_restart_summary.md (this file)
- AI_EXECUTION_PLAN.md (restart tracking section added)

## 5. Blocked or failed items
- Pair-2 fine registration is NOT yet achievable: the ODE 4-corner NAC footprint
  is too coarse (~1-2 km) to blind-init a warp (bilinear/local-affine models give
  inconsistent cross-track offsets; SIFT/POC/profiles all flat). Fallback action:
  the foundation Phase 1 will add equal-GSD staging + photometric
  preconditioning + learned matchers (MPS) before attempting pair 2 again; this
  is designed work, not an unverifiable claim.

## 6. Deliverables from Section 1 satisfied by this phase
- Working data acquisition path from NASA ODE (unblocks "data under data/raw/"
  gap and gives a real second overlapping pair set).
- Corrected scientific verdict for the polar hard case (Phase 5/6 record updated:
  feature-search failure, not non-overlap).
- Restart tracking section in the authoritative execution plan.