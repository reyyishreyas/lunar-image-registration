# LRO NAC Overlap Pipeline

To run immediately go to the cmd and in the pipeline folder and in that paste:

pip install requests beautifulsoup4 shapely

python ch2_nac_overlap_pipeline.py --xml ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml --search-results "SearchResults (1).txt" --verify-top 8 --download-top 1 --out-dir ./downloads




--------------------------------------------------------------

A testing pipeline for finding and downloading **LRO NAC images** that best overlap a given **Chandrayaan-2 OHRC image**.


## Requirements

Python 3.9+ is recommended.

Install dependencies with:

```bash
pip install -r requirements.txt
```

`requirements.txt`:

```text
requests
beautifulsoup4
shapely
```

## Required Files

The pipeline needs:

1. `ch2_nac_overlap_pipeline.py`  
   Main pipeline.

2. `nac_overlap_ranker.py`  
   Ranking logic.

3. An **OHRC PDS4 XML** file containing the OHRC image metadata and footprint.

4. A **LROC SearchResults file** containing the LRO NAC search results generated from the LROC website.

The filenames do **not** need to match the examples below. Pass the actual filenames when running the pipeline.

Example:

```text
input/
├── ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml
└── SearchResults (1).txt
```

## How It Works

```text
OHRC XML
   │
   ▼
Extract OHRC geographic footprint
   │
   ▼
Read LROC Search Results
   │
   ▼
Analyze LRO NAC candidates
   │
   ▼
Rank candidates by spatial overlap
   │
   ├──► ranked_candidates.csv
   │
   └──► Download best candidates
                │
                ▼
            downloads/
```

The pipeline:

1. Reads the OHRC XML.
2. Extracts the OHRC geographic footprint.
3. Reads the LRO NAC candidates from the LROC search-results file.
4. Calculates/analyzes their spatial overlap with the OHRC footprint.
5. Ranks the candidates.
6. Saves the ranking to a CSV file.
7. Downloads the requested top-ranked NAC files automatically.

## Running the Pipeline

Make sure the required files are present before running it.

### Current Testing Example

```bash
python ch2_nac_overlap_pipeline.py --xml ch2_ohr_ncp_20210405T1606536730_d_img_d18.xml --search-results "SearchResults (1).txt" --verify-top 8 --download-top 1 --out-dir ./downloads
```

### General Format

```bash
python ch2_nac_overlap_pipeline.py --xml "<YOUR_OHRC_XML>.xml" --search-results "<YOUR_SEARCH_RESULTS>.txt" --verify-top 8 --download-top 1 --out-dir "./downloads"
```

For example:

```bash
python ch2_nac_overlap_pipeline.py --xml "my_ohrc_image.xml" --search-results "my_search_results.txt" --verify-top 8 --download-top 1 --out-dir "./downloads"
```

## Command-Line Arguments

| Argument | Description |
|---|---|
| `--xml` | Path to the OHRC PDS4 XML file |
| `--search-results` | Path to the LROC SearchResults file |
| `--verify-top` | Number of top-ranked candidates to verify |
| `--download-top` | Number of top-ranked candidates to download |
| `--out-dir` | Directory where downloaded NAC files are stored |

## Output

The pipeline produces a ranked CSV containing the candidate results and downloads the selected NAC files.


## Current Status

This is currently a **testing pipeline** for the lunar-image-registration project.

Current workflow:

**OHRC XML → LROC Search Results → Overlap Analysis → Ranking → CSV → NAC Download**
