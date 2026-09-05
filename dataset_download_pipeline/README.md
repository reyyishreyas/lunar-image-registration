# LRO NAC Overlap Pipeline

TO IMMEDIATELY RUN THE PIPELINE USE IN CMD:
```text
python get_search_results.py ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml
```

A Python pipeline for finding and downloading **LRO NAC images that overlap a Chandrayaan-2 OHRC + TMC observation region**.

## Overview

The pipeline automates the process of finding suitable LRO NAC images for a Chandrayaan-2 observation.

```text
OHRC XML
   │
   ▼
Extract OHRC footprint
   │
   ▼
TMC XML
   │
   ▼
Calculate OHRC ∩ TMC
   │
   ▼
Determine relevant target region
   │
   ▼
Evaluate LRO NAC candidates
   │
   ▼
Rank candidates
   │
   ▼
Save ranked results to CSV
   │
   ▼
Download best NAC files
```

## Why OHRC + TMC?

OHRC provides the **precise high-resolution area of interest**.

TMC provides a **larger regional observation footprint**.

Using the entire TMC footprint for NAC ranking can produce poor results because TMC observations can cover a very large area. Instead, the pipeline calculates:

```text
TMC footprint ∩ OHRC footprint
```

This keeps TMC relevant while preventing its large footprint from dominating the NAC ranking.

The goal is:

> Find the LRO NAC that best covers the terrain observed by OHRC within the TMC-supported region.

---

The actual filenames do not need to match the examples above. File paths are provided when running the pipeline.

---

Requirements:

```bash
pip install requests beautifulsoup4 shapely
```

---

# Input Files

## 1. OHRC XML

A Chandrayaan-2 OHRC PDS4 XML containing the image footprint.

Example:

```text
input/ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml
```

The pipeline extracts the four OHRC corner coordinates and calculates the bounding box automatically.

You do **not** need to manually calculate the minimum and maximum latitude/longitude.

---

## 2. TMC XML

The Chandrayaan-2 TMC observation XML corresponding to the region of interest.

TMC contains multiple product types:

```text
NCA
NCF
NCN
OTH
DTM
```

Each product has its own XML/footprint.

The specific TMC observation being used should overlap the OHRC region.

---

# Running the Pipeline

From the project root:

```powershell
python get_search_results.py <name of OHRC-XML file>
```

### Arguments

| Argument | Description |
|---|---|
| `--ohrc` | Path to the OHRC XML |
| `--tmc` | Path to the TMC XML |
| `--search-results` | Path to LRO NAC `SearchResults.txt` |
| `--download-top` | Number of highest-ranked NAC files to download |
| `--out-dir` | Output directory |


---

# Pipeline Process

## Step 1: Read OHRC Footprint

The pipeline reads the OHRC XML and extracts its four corner coordinates.

It then calculates:

```text
Minimum Latitude
Maximum Latitude
Minimum Longitude
Maximum Longitude
```

No manual bounding-box calculation is required.

---

## Step 2: Read TMC Footprint

The TMC XML is read and its geographic footprint is extracted.

Because the TMC footprint can be significantly larger than the OHRC footprint, the entire TMC bounding box is **not** treated as the target region.

---

## Step 3: Calculate OHRC ∩ TMC

The OHRC and TMC footprints are converted into geographic polygons.

The pipeline calculates:

```text
OHRC ∩ TMC
```

This produces the portion of the OHRC observation that is also covered by the selected TMC observation.

This intersection is used as the relevant region for NAC selection.

---

# Step 4: Evaluate NAC Candidates

The pipeline reads the LRO NAC `SearchResults.txt`.

Each candidate is evaluated against the target region.

Conceptually:

```text
             OHRC
        ┌─────────────┐
        │             │
        │ TARGET      │
        │   REGION    │
        │             │
        └──────┬──────┘
               │
               ∩
              TMC
               │
               ▼
       OHRC ∩ TMC REGION
               │
               ▼
          LRO NACs
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
     NAC 1   NAC 2   NAC 3
```

---

# Step 5: Rank NAC Candidates

Candidates are ranked based on how well they cover the relevant target region.

The ranking prioritizes:

1. OHRC coverage
2. OHRC/TMC intersection coverage
3. Valid overlap with the TMC observation

This avoids the problem where a huge TMC footprint causes the ranking to favor NAC frames simply because they have a large absolute intersection with the TMC strip.

The important metric is:

> How much of the relevant target region does the NAC actually cover?

---

# Step 6: Save Ranked Results

The ranked NAC candidates are saved as:

```text
output/ranked_nac.csv
```

The CSV can contain information such as:

```text
Rank
NAC ID
Overlap Area
Overlap Percentage
Target Coverage
Download URL
```

This allows the ranking to be inspected before or after downloading.

---

# Step 7: Download NAC Files

The highest-ranked NAC candidates are downloaded automatically.

For example:

```text
output/
└── downloads/
    ├── NAC_001.IMG
    └── ...
```

To download only the best candidate:

```text
--download-top 1
```

To download the top 5:

```text
--download-top 5
```

---

# Download Timeout and Retry

The LRO PDS server can sometimes take a long time to begin transferring a file.

The downloader therefore uses a longer connection timeout and retry behavior so that a slow server does not immediately cause the download to fail.

---

# Coordinate System

ISRO PDS labels may contain lunar longitude values in the range:

```text
0° → 360°
```

For example:

```text
336.486234°
```

can be represented in the conventional:

```text
-180° → +180°
```

system as:

```text
-23.513766°
```

The pipeline should normalize longitude consistently before performing geometric calculations.

Latitude remains:

```text
-90° → +90°
```

---

# Current Scope

The current pipeline focuses on:

```text
OHRC
  +
TMC
  ↓
LRO NAC
```

The broader project will eventually incorporate:

```text
OHRC
TMC
├── NCA
├── NCF
├── NCN
├── OTH
└── DTM
       +
IIRS
       ↓
LRO NAC
```

TMC product types and IIRS processing are intentionally being added incrementally.

---

# Example Workflow

Given:

```text
OHRC XML
TMC XML
SearchResults.txt
```

run:

```powershell
python main.py --ohrc "input/ohrc.xml" --tmc "input/tmc.xml" --search-results "input/SearchResults.txt" --download-top 1 --out-dir "./output"
```

The pipeline will:

```text
✓ Read OHRC footprint
✓ Calculate OHRC bounding box
✓ Read TMC footprint
✓ Calculate OHRC ∩ TMC
✓ Evaluate NAC candidates
✓ Rank NAC candidates
✓ Save ranked_nac.csv
✓ Download the highest-ranked NAC
```

---

# Goal

The original manual workflow:

```text
Find OHRC
      ↓
Manually calculate latitude/longitude
      ↓
Open LROC website
      ↓
Enter bounding box
      ↓
Download SearchResults.txt
      ↓
Inspect NAC candidates
      ↓
Calculate overlap
      ↓
Rank candidates
      ↓
Download NAC
```

is being converted into:

```text
OHRC XML + TMC XML + SearchResults.txt
                    ↓
               RUN PIPELINE
                    ↓
          Ranked NAC candidates
                    ↓
             Best NAC files
```

The intermediate calculations and ranking are retained in CSV output for verification and reproducibility.
