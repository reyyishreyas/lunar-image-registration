# Multi-modal Lunar Image Registration

**Smart India Hackathon — Problem Statement 26166**
Multi-modal, Sun-angle and Scale-invariant Image Correspondence using Chandrayaan-2 Optical Images (OHRC, TMC, IIRS)

**Organization:** Indian Space Research Organisation (ISRO), Department of Space
**Category:** Software | **Theme:** Space Technology

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Our Approach](#our-approach)
- [Pipeline Overview](#pipeline-overview)
- [Hard Cases: Dark / Low-Sun / Polar Regions](#hard-cases-dark--low-sun--polar-regions)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
- [Dataset Layout](#dataset-layout)
- [Usage](#usage)
- [Datasets](#datasets)
- [Evaluation Metrics](#evaluation-metrics)
- [Results](#results)
- [Tech Stack](#tech-stack)
- [Team](#team)
- [References](#references)
- [License](#license)

---

## Problem Statement

Image registration is the process of aligning two or more images of the same scene — taken at different times, from different viewpoints, or by different sensors ,  into a common coordinate system. This project registers **Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)** against reference lunar imagery (e.g. **LRO NAC**), overcoming three core challenges:

| Challenge | Description |
|---|---|
| **Illumination variation** | Different sun azimuth/elevation angles drastically change surface appearance due to the Moon's lack of atmosphere — shadows are sharp-edged and near-black, and the same crater can look unrecognizable under different lighting. |
| **Viewpoint variation** | Different orbital passes and camera orientations cause geometric distortion — the same feature can appear shifted, rotated, or perspective-warped. |
| **Scale variation** | Chandrayaan-2 payloads operate at vastly different resolutions — OHRC (~0.3 m/pixel), TMC-2 (~5 m/pixel), IIRS (~80 m/pixel) — creating scale ratios far beyond what standard feature detectors are built to handle. |

### Expected Solution

A generic software solution that finds correspondence between Chandrayaan-2 images and lunar reference images with:

- **Sub-pixel accuracy**
- **Uniform spatial distribution of match points** across the image
- A **registered output product** with corresponding match points
- **Quantitative evaluation metrics**: RMSE, inlier count, inlier ratio

---

## Our Approach

1. Build a **modular pipeline** where every stage (preprocessing, detection, matching, outlier rejection, refinement) is a swappable, independently testable component.
2. Start with a simple, working end-to-end baseline (classical detector + RANSAC) before adding complexity.
3. Invest heavily in **preprocessing** (illumination/shadow correction) — prior research shows this has the single largest impact on lunar image matching quality.
4. Benchmark classical methods (SIFT, ASIFT, AKAZE, RIFT2) against deep-learning methods (SuperGlue, LoFTR) on identical image pairs.
5. Explicitly test the **hard case** — polar or low-sun-angle image pairs — since this is where most existing methods fail, and robustness here is our key differentiator.
6. Report every result with hard numbers (RMSE, inlier ratio, runtime), not just qualitative claims.
7. **Detect when content matching is not trustworthy and say so honestly** — never fabricate a registration model.

---

## Pipeline Overview

```
Input Images (OHRC/TMC/IIRS + LRO reference)
        │
        ▼
Preprocessing (georeference → resample → normalize → shadow-correct)
        │
        ▼
Feature Detection (multimodal keypoints & descriptors)
        │
        ▼
Feature Matching (nearest-neighbor or learned matcher)
        │
        ▼
Outlier Rejection (RANSAC + grid-based uniformity enforcement)
        │
        ▼
Sub-pixel Refinement (phase correlation / corner refinement)
        │
        ▼
Registered Output (aligned image + match points + metrics)
```

The **fully-automatic entry point** (`scripts/run_auto.py`) wraps this in a layered decision:

1. **Enhance** — lift contrast on both images (critical for dark/low-sun frames).
2. **Content match** — detect features and register from content. Success → report `registered` with RMSE + inliers.
3. **Geometry fallback** — if content is not verifiable but the two footprints overlap on the ground, both images are re-projected onto a common grid using the ISRO spacecraft geometry (no content needed). Success → report `geometry_registered`.
4. **Honest failure** — if neither path is trustworthy → report `not_registered`. The pipeline never invents an alignment.

See [`docs/methodology.md`](docs/methodology.md) for the full stage-by-stage explanation and rationale.

---

## Hard Cases: Dark / Low-Sun / Polar Regions

Polar and low-Sun lunar images (e.g. near 80°S/Malapert) can be almost black — real examples in this repository have a mean level of ~5/255. On such data, feature detectors produce no reliable matches, and forcing a content-based model would be wrong.

The pipeline handles these automatically:

- Detection scripts score every frame with a **low-contrast heuristic** and apply **contrast enhancement (percentile stretch + CLAHE)** before matching.
- If content registration still cannot be verified, the pipeline switches to **geometry registration** on a shared ground grid (equal GSD, aligned longitude/latitude box) using the ISRO geometry CSVs — no SPICE kernel required.
- The result is reported as `geometry_registered` with an explicit note, so downstream consumers know a geometric (not photometric) registration was used.
- If neither content nor geometry is usable (e.g. IIRS, which ships **no** ground-geometry CSV), the verdict is an honest `not_registered`.

---

## Repository Structure

```
lunar-image-registration/
├── configs/               # YAML configs, one per experiment/ablation run
├── data/                  # Raw products + processed output (gitignored, see below)
├── weights/               # Pretrained deep-matcher weights (optional, gitignored)
├── src/
│   ├── pipeline.py        # Orchestrates the full pipeline from a config file
│   ├── auto_pipeline.py   # Fully-automatic registration (layered decision + CLI dispatch)
│   ├── io_utils/          # Format readers, metadata extraction
│   ├── preprocessing/     # Georeferencing, resampling, normalization, shadow correction,
│   │                      # CH2 raw readers (tmc.py) and dark/polar ground-staging (ch2_staging.py)
│   ├── detection/         # Classical, cross-modal, and learned feature detectors
│   │                      # (cross_modal.py, iirs.py band-selective hyperspectral reader)
│   ├── matching/          # Classical and learned feature matchers
│   ├── outlier_rejection/ # RANSAC variants, grid-based uniformity capping
│   ├── refinement/        # Sub-pixel refinement methods
│   └── evaluation/        # Metrics computation and visualization
├── scripts/               # CLI entry points
│   ├── run_auto.py        # ★ Fully-automatic multi-sensor registration (primary CLI)
│   ├── run_pipeline.py    # Config-driven pipeline runner
│   ├── run_ablation.py    # Ablation sweeps
│   └── ...                # ground-truth, geometry, data-download helpers
├── notebooks/             # Exploratory analysis only — not the source of truth
├── results/               # Logs, figures, and the final chosen configuration
├── demo/                  # Streamlit end-user app (app.py)
├── tests/                 # Unit tests for each module
└── docs/                  # Methodology, related work, architecture diagrams
```

---

## Getting Started

### Prerequisites

- **Python 3.10+** (developed and verified on Python 3.13)
- **Git**
- A reasonable amount of disk space — raw products are multi-GB (`data/` is gitignored)
- **GPU not required.** The default path (classical + cross-modal + geometry) runs on CPU; PyTorch also runs on Apple MPS. Deep-learning matchers (SuperGlue/LoFTR) are optional and need a GPU or Colab.

### Installation

```bash
# Clone the repository
git clone https://github.com/reyyishreyas/lunar-image-registration.git
cd lunar-image-registration

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **Important — OpenCV pin:** `requirements.txt` pins `opencv-contrib-python<5`. OpenCV **5.x removed AKAZE/KAZE/BRISK** and the detectors used by the algorithm notebooks. Do **not** install `opencv-python` alongside it — the two packages conflict. This project is verified with `opencv-contrib-python 4.10.0`.

### Verify the install

```bash
python -c "import cv2, numpy, rasterio, skimage; print('ok', cv2.__version__)"
python -m pytest tests/ -q          # should end with "64 passed"
```

> `.dist-info` warnings from `python -m pip check` (leftover `-einops`/`-kornia`/`-gdown` dirs) are harmless; the suite runs clean regardless.

### (Optional) Pretrained weights for deep matchers

SuperPoint/SuperGlue/LoFTR need pretrained weights not shipped in this repo. Place the downloaded weights inside `weights/` (gitignored). On a CPU-only machine these run slowly; Colab/Kaggle is recommended for the learned decoder.

### (Optional) Data acquisition

See [`data/README.md`](data/README.md) for sources:

- **Chandrayaan-2 (OHRC, TMC-2, IIRS):** [ISSDC Chandrayaan Map Browse](https://chmapbrowse.issdc.gov.in/)
- **Reference imagery (LRO NAC/WAC):** [LROC QuickMap](https://quickmap.lroc.im-ldi.com/)

`scripts/download_data.sh` and the ODE REST search tool (in `dataset_download_pipeline/`) can fetch NASA ODE products; note that **ISRO products (OHRC/TMC/IIRS) must be retrieved manually from ISSDC**.

---

## Dataset Layout

`data/` is **gitignored** — products live directly under top-level subdirectories (not `data/raw/`). After acquisition, the expected layout is:

```
data/
├── PATCH-001/                       # 2021 Chandrayaan-2 + LRO NAC overlap
│   ├── data/OHRC/ch2_ohr_ncp_20210405.../   # .img strip + geometry CSV
│   ├── data/IIRS/.../                        # .qub + .hdr (hyperspectral)
│   └── data/LRO_NAC/LC/M1469248775LC.IMG     # NAC reference
├── PATCH-004/                       # 2026 Chandrayaan-2 + LRO NAC overlap (polar-adjacent)
│   ├── TMC/... .zip                 # TMC raw still zipped (app auto-extracts)
│   ├── OHRC/data/calibrated/20260331/...
│   ├── OHRC/geometry/calibrated/20260331/...
│   ├── LRO NAC/OHRC/M1127547939RC.IMG
│   └── IIRS/ ...
├── ground_truth/                    # manually marked control points (pair-1)
└── processed/                       # derived products (auto stages, uploads, crops)
```

> **Paths contain spaces** (e.g. `data/LRO NAC 2/`, `data/PATCH-004/LRO NAC/...`). Always quote them in the shell, exactly as the examples below do.

---

## Usage

### Launch the Streamlit end-user app (recommended for non-experts)

Everything is exposed through a one-click UI: pick a sensor pair, press **Register**, and read a plain-language result.

```bash
python -m streamlit run demo/app.py
```

Modes:

| Mode | What it does |
|---|---|
| **Automatic registration** | OHRC ⟷ LRO NAC, fully automatic (pair-1 bright, pair-2 polar/geometry) |
| **Multi-sensor registration (TMC / IIRS)** | TMC ⟷ OHRC and IIRS ⟷ OHRC presets; auto-extracts the 2 GB TMC archive on first run; dark/low-sun pairs fall back to geometry automatically |
| **Project pair-1 / pair-2** | Config-driven pipeline runs with georeference staging |
| **Upload aligned crops** | Score your own two grayscale PNGs against the FINAL config |

The app only orchestrates `src/` — it never reimplements matching, georeferencing, or outlier rejection.

### CLI: fully-automatic registration (all sensor pairs)

The primary CLI is `scripts/run_auto.py`. Three sensor pairs are supported: `ohrc-nac`, `tmc-ohrc`, `iirs-ohrc`.

```bash
python scripts/run_auto.py --help
```

**OHRC ⟷ LRO NAC (bright pair, sub-pixel content match + ground-truth check):**

```bash
python scripts/run_auto.py --sensor ohrc-nac \
  --src_img  "data/PATCH-001/data/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/data/calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_d_img_d18.img" \
  --src_geom "data/PATCH-001/data/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/geometry/calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv" \
  --ref_img  "data/PATCH-001/data/LRO_NAC/LC/M1469248775LC.IMG" \
  --ground_truth "data/ground_truth/pair1_gt_v2.csv" \
  --out_dir data/processed/auto --prefix ohrc_nac
```

**OHRC ⟷ LRO NAC (dark/low-sun pair → geometry registration fallback):**

```bash
python scripts/run_auto.py --sensor ohrc-nac \
  --src_img  "data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img" \
  --src_geom "data/PATCH-004/OHRC/geometry/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv" \
  --ref_img  "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG" \
  --out_dir data/processed/auto --prefix ohrc_nac_polar
```

Expected summary line: `REGISTERED BY GEOMETRY — Stage 2 (SPICE) required` or the equivalent `geometry_registered` verdict.

**TMC ⟷ OHRC (Chandrayaan-2 to Chandrayaan-2, same orbit family; dark OHRC handled by geometry):**

```bash
# TMC raw ships inside a zip — extract once (the Streamlit app does this for you):
unzip -o -j "data/PATCH-004/TMC/ch2_tmc_nca_20260607T2319176707_d_img_d18.zip" \
  "data/calibrated/20260607/ch2_tmc_nca_20260607T2319176707_d_img_d18.img" -d data/processed/tmc
unzip -o -j "data/PATCH-004/TMC/ch2_tmc_nca_20260607T2319176707_d_img_d18.zip" \
  "geometry/calibrated/20260607/ch2_tmc_nca_20260607T2319176707_g_grd_d18.csv" -d data/processed/tmc

python scripts/run_auto.py --sensor tmc-ohrc \
  --src_img  "data/processed/tmc/ch2_tmc_nca_20260607T2319176707_d_img_d18.img" \
  --src_geom "data/processed/tmc/ch2_tmc_nca_20260607T2319176707_g_grd_d18.csv" \
  --ref_img  "data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img" \
  --ref_geom "data/PATCH-004/OHRC/geometry/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv" \
  --out_dir data/processed/auto --prefix tmc_ohrc
```

Expected summary line (this real pair, at ~21.7 m/px):

```
GEOMETRY REGISTERED (no content): GSD=21.701669226943483 m, method=geometry
```

**IIRS ⟷ OHRC (hyperspectral IR vs visible; IIRS has no geometry CSV → content only, honest verdict):**

```bash
python scripts/run_auto.py --sensor iirs-ohrc \
  --src_img  "data/PATCH-001/data/IIRS/ch2_iir_nri_20211221T0324126144_d_img_hw1/data/raw/20211221/ch2_iir_nri_20211221T0324126144_d_img_hw1.qub" \
  --src_geom "data/PATCH-001/data/IIRS/ch2_iir_nri_20211221T0324126144_d_img_hw1/data/raw/20211221/ch2_iir_nri_20211221T0324126144_d_img_hw1.hdr" \
  --ref_img  "data/PATCH-001/data/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/data/calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_d_img_d18.img" \
  --ref_geom "data/PATCH-001/data/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/geometry/calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv" \
  --out_dir data/processed/auto --prefix iirs_ohrc
```

### Outputs

Every `run_auto` run writes a JSON report plus image artifacts to `--out_dir`:

- `<prefix>.json` — full report: verdict, method, RMSE, inliers, GSD, footprints, notes
- `<prefix>_src.png` / `<prefix>_ref.png` — ground-registered pair on the common grid
- `<prefix>_matches.png` — inlier match overlay (content path only)
- `<prefix>_checkerboard.png`, `<prefix>_overlay.png` (OHRC↔NAC path)

Verdicts a consumer can rely on:

| Verdict | Meaning |
|---|---|
| `registered` | Content-based registration with verified inliers + RMSE |
| `geometry_registered` | Content not verifiable (dark/low-sun); registered on a common ground grid from ISRO geometry |
| `no_content_correspondence` | Content matching failed; geometry fallback attempted |
| `not_registered` | Neither path trustworthy — reported honestly |

### Config-driven pipeline runs (for experiments / ablations)

```bash
# Run one experiment defined by a YAML config
python scripts/run_pipeline.py --config configs/experiment_sift_baseline.yaml

# Ablation sweep across all configs
python scripts/run_ablation.py --configs configs/
```

Results append to `results/logs/ablation_results.csv`.

### Manually mark ground-truth control points (for RMSE evaluation)

```bash
python scripts/pick_ground_truth.py \
  --source data/processed/ohrc_crop.png \
  --reference data/processed/nac_crop.png
```

---

## Datasets

| Source | Instrument | Approx. Resolution | Type |
|---|---|---|---|
| Chandrayaan-2 | OHRC | ~0.3 m/pixel (0.25–1 m across the archive) | Panchromatic optical |
| Chandrayaan-2 | TMC-2 | ~5 m/pixel nominal (measured 21.7 m/px on the processed 2026 acquisition) | Panchromatic optical, 3-view stereo |
| Chandrayaan-2 | IIRS | ~80 m/pixel | Hyperspectral (256 bands) |
| LRO | NAC / WAC | ~0.5–2 m (NAC), ~100 m (WAC) | Reference imagery |

Full download instructions and licensing notes are in [`data/README.md`](data/README.md).

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **RMSE** | Root mean square pixel error between registered and reference points, measured against manually verified ground-truth control points |
| **Inlier count** | Number of matches surviving RANSAC outlier rejection |
| **Inlier ratio** | Inliers divided by total matches — a measure of match quality |
| **Uniformity score** | How evenly match points are spatially distributed across the image |
| **Runtime** | Per-stage and end-to-end execution time |
| **Low-contrast score** | 0–1 heuristic flagging dark/low-sun frames that must use the geometry path |
| **GSD** | Ground sample distance (m/px) of the stage-on-common-grid output |

---

## Results

All results are logged under `results/logs/` (per-phase summaries, ablation CSV, real-pair verdicts).

| Pair | Sensor route | Result |
|---|---|---|
| OHRC-2021 ⟷ LRO NAC M1469248775LC | content (`ohrc-nac`) | `registered`, sub-pixel RMSE < 1 px against ground truth (see `results/logs/phase9_summary.md`) |
| OHRC-2026 ⟷ LRO NAC M1127547939RC | geometry fallback (`ohrc-nac`) | `geometry_registered`, equal-GSD ortho pair |
| TMC-2026 ⟷ OHRC-2026 | geometry fallback (`tmc-ohrc`) | `geometry_registered`, GSD 21.70 m/px, exit 0 |
| IIRS-2021 ⟷ OHRC-2021 | content-only (`iirs-ohrc`) | honest `not_registered` (no match + no geometry — never fabricated) |

The best configurable content pipeline is documented in [`results/final_config.yaml`](results/final_config.yaml).

---

## Tech Stack

**Language:** Python 3.10+ (verified on 3.13)

**Core libraries:**
- **Geospatial:** rasterio, pyproj, pygeodesy
- **Image processing:** OpenCV (contrib 4.x), scikit-image, NumPy, SciPy
- **Deep learning:** PyTorch (SuperPoint, SuperGlue, LoFTR — optional)
- **Data/experiment management:** pandas, PyYAML, tqdm
- **Visualization/demo:** Matplotlib, Streamlit

See `requirements.txt` for the exact pinned set.

---

## Team

| Name | Role |
|---|---|
| — | Data & Preprocessing Lead |
| — | Classical CV Lead |
| — | Deep Learning Lead |
| — | Evaluation & Metrics Lead |
| — | Demo & Presentation Lead |

---

## References

1. TMC-2 Payload Characteristics, *Current Science*
2. MoonMetaSync: Lunar Image Registration Analysis (2024)
3. Comparative Evaluation of Traditional and Deep Learning Feature Matching Algorithms using Chandrayaan-2 Lunar Data, ISRO Space Applications Centre (2025)

Full related-work notes are in [`docs/related_work.md`](docs/related_work.md).

---

## License

*(Specify your team's chosen license here, e.g. MIT, or "For SIH evaluation purposes only" if not open-sourcing.)*
