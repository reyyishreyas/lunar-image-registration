# Lunar Image Registration — Presentation Deck

Chandrayaan-2 OHRC ⟷ LRO NAC co-registration
Ablation-best config: SIFT + CLAHE + BF + USAC_MAGSAC, **RMSE 22.6172 px, 235 inliers, 0.8217**

---

## 1. Problem statement

- Register pushbroom lunar images from **different missions (Chandrayaan-2 OHRC vs LRO NAC)** so features, time-tag exposure, and change are comparable at the same ground area.
- Requirement: an **end-to-end pipeline** — georeference → detection → matching → outlier rejection → evaluation — runnable on ≥1 OHRC–NAC pair with **reported RMSE / inlier count / inlier ratio**.
- Hard case in scope: near-Polar / grazing-sun products (sun elevation ≈ 1.9°) where classical radiometry assumptions break.

## 2. Related work / gap

- Classical pipelines: SIFT/ORB/AKAZE/BRISK/KAZE + RANSAC are the standard toolset but assume comparable illumination.

---

## 2. Related work / gap (cont.)

- Learned matchers: SuperPoint+SuperGlue/LoFTR help, but need **matching photometric content**.
- Lunar grazing-sun scenes differ massively between sensors → **a content-absence detector is required** — the pipeline must *refuse* confidently when no same-ground overlap exists, instead of emitting garbage.
- Georeferencing via sensor geometry (SPICE + ISRO CSV) is a reliable fallback where content matching is impossible.

## 3. Pipeline architecture

```
                     ┌────────────────────────────────────────────┐
   OHRC .img ───────▶│ staging (stage_pair)                      │
   ISRO geometry CSV ▶│  georeference_pair / SPICE ortho (equal-  │──▶ src crop
                     │  GSD, self-calibrated NAC factor)          │
   NAC .IMG ────────▶│  photometric precondition: none|clahe|edges│──▶ ref crop
                     └────────────────────────────────────────────┘
                                  │
                                  ▼
                 detection (SIFT/AKAZE/RIFT2/SuperPoint)
                                  │
                 matching (BF ratio / LoFTR / SuperGlue / RIFT2)
                                  │
                 outlier rejection (RANSAC / USAC_MAGSAC, optional grid cap)
                                  │
                 refinement (cornerSubPix / phase-correlation)
                                  │
                 evaluation (RMSE vs GT, overlay figure, ablation row)
```

## 4. Ablation results (pair 1: OHRC-2021 + NAC M1469248775LC)

| Config | Detector+Matcher | Outlier | Refine | RMSE px | Inliers | Ratio |
|---|---|---|---|---|---|---|
| **FINAL/C10** | **SIFT + BF 0.75** | **USAC_MAGSAC 5** | — | **22.6172** | **235** | **0.8217** |
| C12 | + phase_corr | | | 22.6379 | 235 | 0.8217 |
| C11 | + cornerSubPix | | | 22.65 | 235 | 0.8217 |
| C8 | SuperPoint+LoFTR | RANSAC | | 22.6557 | 4583 | 0.673 |
| C7 | SuperPoint+SuperGlue | | | 22.6703 | 137 | 0.598 |
| C3 | SIFT+BF | RANSAC | | 22.6316 | 236 | 0.825 |
| C2 | histmatch | | | 22.6605 | 223 | 0.686 |
| C1 | baseline normalize | | | 22.6614 | 56 | 0.918 |
| C5 | AKAZE | | | 22.8034 | 793 | 0.808 |
| C6 | RIFT2 | | | 500.5 | 193 | 0.214 |
| C4 | gamma_shadow | | | 8353.4 | 6 | 0.15 |
| C13 | equal-GSD staging | | | 26.977* | 245 | 0.825 |
| C14 | edges precond. | | | 742.1 | 10 | 0.588 |

\* GT anchored to the native crop window; restaged crop not strictly comparable.

```
16 configs run; best = classical SIFT+C LAHE (beats both learned rows on RMSE).
```

## 5. Polar case study (pair 2: OHRC-2026 sun_el 1.9° + NAC M1127547939RC)

- Georeference overlap search: **14 SIFT inliers < 20** → RuntimeError, no crops.
- Five-method consistency: SIFT 14, SuperGlue 2, LoFTR 9/419, dense NCC ≤0.06, template ZNCC ≤0.055.
- Multi-orientation dominance + NCC diagnostic: **no dominant flip**, all NCC ≈ 0 — the pair has **no shared content** (photometric gap, sun elevation 1.9°).
- Outcome: **geometry-based registration** — ISRO CSV + SPICE are mutually consistent (2000/2000 in-bounds, self-consistency < 1e-11 km) → `georef_phase5.json`, equal-GSD orthos.
- This is a *documented, evidence-based* refusal, not a silent skip.

## 6. Demo (live / recorded)

- Streamlit app `demo/app.py` drives `src/pipeline.py` with FINAL_CONFIG only (no reimplemented logic).
- Shows: match overlay, checkerboard of the aligned staged crops, RMSE/inliers/ratio metrics, staged GSD.
- Pair-2 mode demonstrates the hardened refusal with diagnostics path.
- Fallback: `demo/fallback/` cached output + pre-recorded video.

## 7. Future work

- Percentile-based cross-sensor radiometric harmonization for grazing-sun pairs.
- End-to-end SPICE-orthographic staging for ALL pairs (equal-GSD, no SIFT anchor).
- Sub-pixel refinement against SPICE-forward model (target <2 px; chase 1 px).
- Probe the 3 remaining ODE NAC candidates (93–97% overlap) once photometric rescue matures.
- Full-resolution (non-crop) registration at the staged pipeline's proven GSD.