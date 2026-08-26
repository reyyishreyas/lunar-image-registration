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
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
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

Image registration is the process of aligning two or more images of the same scene — taken at different times, from different viewpoints, or by different sensors — into a common coordinate system. This project registers **Chandrayaan-2 optical images (OHRC, TMC-2, IIRS)** against reference lunar imagery (e.g. **LRO NAC**), overcoming three core challenges:

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

See [`docs/methodology.md`](docs/methodology.md) for the full stage-by-stage explanation and rationale.

---

## Repository Structure

```
lunar-image-registration/
├── configs/               # YAML configs, one per experiment/ablation run
├── data/                  # raw/, processed/, ground_truth/ (gitignored, see data/README.md)
├── weights/               # Pretrained model weights (SuperPoint, SuperGlue, LoFTR) — gitignored
├── src/
│   ├── pipeline.py        # Orchestrates the full pipeline from a config file
│   ├── io_utils/          # Format readers, metadata extraction
│   ├── preprocessing/     # Georeferencing, resampling, normalization, shadow correction
│   ├── detection/         # Classical, cross-modal, and learned feature detectors
│   ├── matching/          # Classical and learned feature matchers
│   ├── outlier_rejection/ # RANSAC variants, grid-based uniformity capping
│   ├── refinement/        # Sub-pixel refinement methods
│   └── evaluation/        # Metrics computation and visualization
├── scripts/               # CLI entry points (run_pipeline.py, run_ablation.py, etc.)
├── notebooks/             # Exploratory analysis only — not the source of truth
├── results/               # Logs, figures, and the final chosen configuration
├── demo/                  # Streamlit demo application
├── tests/                 # Unit tests for each module
└── docs/                  # Methodology, related work, architecture diagrams
```

---

## Getting Started

### Prerequisites

- Python 3.10+
- Git
- (Optional but recommended) A CUDA-capable GPU for deep-learning matchers, or access to Google Colab / Kaggle
- GDAL installed at the system level (required by `rasterio`)

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-org>/lunar-image-registration.git
cd lunar-image-registration

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Downloading pretrained weights

Deep-learning matchers require pretrained weights not included in this repository due to size:

```bash
# Example — adjust paths per the actual model source
bash scripts/download_weights.sh
```

Place downloaded weights in the `weights/` directory as documented in `weights/README.md`.

### Downloading data

See [`data/README.md`](data/README.md) for dataset sources and download instructions:

- **Chandrayaan-2 (OHRC, TMC-2, IIRS):** [ISSDC Chandrayaan Map Browse](https://chmapbrowse.issdc.gov.in/)
- **Reference imagery (LRO NAC/WAC):** [LROC QuickMap](https://quickmap.lroc.im-ldi.com/)

---

## Usage

### Run the full pipeline on a single image pair

```bash
python scripts/run_pipeline.py --config configs/experiment_sift_baseline.yaml
```

### Run an ablation sweep across multiple configurations

```bash
python scripts/run_ablation.py --configs configs/
```

Results are appended to `results/logs/ablation_results.csv`.

### Manually mark ground-truth control points (for RMSE evaluation)

```bash
python scripts/pick_ground_truth.py --source data/processed/ohrc_example.tif --reference data/processed/nac_example.tif
```

### Launch the interactive demo

```bash
streamlit run demo/app.py
```

---

## Datasets

| Source | Instrument | Approx. Resolution | Type |
|---|---|---|---|
| Chandrayaan-2 | OHRC | ~0.3 m/pixel | Panchromatic optical |
| Chandrayaan-2 | TMC-2 | ~5 m/pixel | Panchromatic optical, 3-view stereo |
| Chandrayaan-2 | IIRS | ~80 m/pixel | Hyperspectral (~250 bands) |
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

---

## Results

Ablation results are tracked in [`results/logs/ablation_results.csv`](results/logs/ablation_results.csv) and summarized in the presentation deck. The current best-performing configuration is documented in [`results/final_config.yaml`](results/final_config.yaml).

*(Update this section with your actual numbers once experiments are complete.)*

---

## Tech Stack

**Language:** Python 3.10+ (with optional MATLAB for RIFT2, C++ for performance-critical modules)

**Core libraries:**
- **Geospatial:** GDAL, rasterio, pyproj, pygeodesy, USGS ISIS3
- **Image processing:** OpenCV, scikit-image, NumPy, SciPy
- **Deep learning:** PyTorch (SuperPoint, SuperGlue, LoFTR)
- **Data/experiment management:** pandas, PyYAML, tqdm
- **Visualization/demo:** Matplotlib, Streamlit

Full library breakdown by pipeline stage is documented in [`docs/methodology.md`](docs/methodology.md).

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