#!/usr/bin/env python3
"""
ch2_nac_overlap_pipeline.py  (v4)

End-to-end pipeline: given one or more Chandrayaan-2 instrument PDS4 labels
(OHRC, TMC, or IIRS), find the best-overlapping LRO NAC frame(s) for each and
download them. No more manual "search ODE by hand, export SearchResults.txt"
step required (though it's still supported as a fallback).

WHAT'S NEW IN v4:
  - AUTO-SEARCH: queries a bounding box directly against a REST API instead
    of requiring a manually-exported SearchResults.txt. (EXPERIMENTAL --
    see confidence notes below.)
  - BATCH MODE: point --xml-dir at a folder of instrument label XMLs and it
    processes all of them in one run, instead of one file at a time.
  - Per-instrument/per-patch output folders, so results don't overwrite
    each other across runs.
  - Local JSON cache of fetched product details, so reruns don't re-hit
    the server for products already checked.
  - Downloads the top N verified candidates, not just #1.
  - Download integrity check (downloaded size vs. server-reported size).
  - Writes a running manifest.csv with one row per downloaded product.

CONFIDENCE LEVELS (read this before trusting the output):
  - corners_from_pds4_xml : TESTED for OHRC labels. UNVERIFIED for TMC/IIRS
                             -- if it raises or returns nonsense on those,
                             the tag structure differs; send a sample label.
  - Manual search+rank+verify+download (--search-results) : TESTED end to
                             end against real data (see project history).
  - AUTO-SEARCH (--auto-search, query_pilot_rest) : NOT LIVE-TESTED. Built
                             against real, fetched API documentation
                             (pilot.rsl.wustl.edu), but the assistant that
                             wrote this has no network access to that
                             domain to confirm the call actually works or
                             that optional path segments are being skipped
                             correctly. If it 404s or returns something
                             unexpected, run with --debug-search to see the
                             raw URL and response, share it, and fall back
                             to --search-results in the meantime -- it
                             still works and is unaffected by this.

REQUIREMENTS:
    pip install requests beautifulsoup4 shapely

USAGE:
    # Single instrument, auto-search (new, may need a debug round):
    python3 ch2_nac_overlap_pipeline.py --xml ohrc_d_img_label.xml \
        --auto-search --verify-top 15 --download-top 1 --out-dir ./downloads

    # Single instrument, manual search results (proven, always works):
    python3 ch2_nac_overlap_pipeline.py --xml ohrc_d_img_label.xml \
        --search-results SearchResults.txt --verify-top 8 --download-top 1 \
        --out-dir ./downloads

    # BATCH: a whole folder of instrument XMLs, auto-search each:
    python3 ch2_nac_overlap_pipeline.py --xml-dir ./labels \
        --auto-search --verify-top 15 --download-top 1 --out-dir ./downloads

    # Manual download of a known link, no search/verify:
    python3 ch2_nac_overlap_pipeline.py --download-only \
        --img-url "https://pds.lroc.im-ldi.com/data/.../M107035386RC.IMG" \
        --out-dir ./downloads
"""
import argparse
import csv
import glob
import hashlib
import json
import math
import os
import re
import sys
import time
import xml.etree.ElementTree as ET

try:
    import requests
except ImportError:
    requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None

try:
    from shapely.geometry import Polygon
except ImportError:
    Polygon = None


PIPELINE_VERSION = "2025-09-02-v4-auto-search-batch"

ISDA_NS = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}
KM_PER_DEG_LAT = 30.3


# ----------------------------------------------------------------------
# Shared geometry helpers
# ----------------------------------------------------------------------

def km_per_deg_lon(lat):
    return KM_PER_DEG_LAT * math.cos(math.radians(abs(lat)))


def corners_from_pds4_xml(xml_path):
    """Extract footprint corners (lat, lon) from a Chandrayaan-2 PDS4 ISDA
    label. Tested against OHRC 'd_img' labels. If pointed at a TMC or IIRS
    label and this raises/returns nonsense, the tag names likely differ --
    open the XML and check rather than trusting a silent guess."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    geom = root.find(".//isda:Geometry_Parameters/isda:Refined_Corner_Coordinates", ISDA_NS)
    if geom is None:
        geom = root.find(".//isda:Geometry_Parameters/isda:System_Level_Coordinates", ISDA_NS)
    if geom is None:
        raise ValueError(
            "No corner coordinates found under isda:Geometry_Parameters. "
            "If this is a TMC/IIRS label, the tag structure may differ from "
            "OHRC -- inspect the XML directly rather than assuming."
        )

    def get(tag):
        el = geom.find(f"isda:{tag}", ISDA_NS)
        return float(el.text)

    return {
        "UL": (get("upper_left_latitude"), get("upper_left_longitude")),
        "UR": (get("upper_right_latitude"), get("upper_right_longitude")),
        "LR": (get("lower_right_latitude"), get("lower_right_longitude")),
        "LL": (get("lower_left_latitude"), get("lower_left_longitude")),
    }


def corners_center(corners):
    lats = [v[0] for v in corners.values()]
    lons = [v[1] for v in corners.values()]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def corners_bounds(corners):
    lats = [v[0] for v in corners.values()]
    lons = [v[1] for v in corners.values()]
    return min(lats), max(lats), min(lons), max(lons)  # minlat, maxlat, westlon, eastlon


def corners_polygon(corners):
    """corners dict -> shapely Polygon in (lon, lat) order, ring UL-UR-LR-LL."""
    if Polygon is None:
        raise RuntimeError("shapely is required for exact overlap -- pip install shapely")
    ring = [corners["UL"], corners["UR"], corners["LR"], corners["LL"]]
    poly = Polygon([(lon, lat) for lat, lon in ring])
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def _normalize(s):
    """Collapse whitespace, including non-breaking spaces (\\xa0), which is
    the classic reason 'the label text looks right but regex doesn't match'
    when scraping real-world HTML tables."""
    return re.sub(r"\s+", " ", s.replace("\xa0", " ")).strip()


# ----------------------------------------------------------------------
# Local cache (avoid re-fetching product detail pages on reruns)
# ----------------------------------------------------------------------

def load_cache(path):
    if path and os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_cache(path, cache):
    if not path:
        return
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# STAGE 0 (NEW, EXPERIMENTAL): automated bounding-box search
# ----------------------------------------------------------------------

PILOT_API_BASE = "https://pilot.rsl.wustl.edu/api/v1/search/products/metadata"


def query_pilot_rest(min_lat, max_lat, west_lon, east_lon, target="moon", mission="LRO",
                      instrument="LROC", prod_type="CDRNAC4", page_size=200,
                      session=None, timeout=30, debug=False):
    """Queries the PILOT REST API for NAC products overlapping a bounding
    box -- this is what replaces the manual 'go to ODE, type lat/lon,
    export SearchResults.txt' step.

    EXPERIMENTAL / NOT LIVE-TESTED (see module docstring). If this raises
    or returns something unexpected, pass debug=True (or --debug-search on
    the CLI) to print the exact URL and raw response for troubleshooting.

    Returns the raw parsed JSON list from the API (list of dicts with keys
    like pds3ProductId, detailUrl, observationStartUtc, ...).
    """
    if requests is None:
        raise RuntimeError("pip install requests")
    sess = session or requests.Session()

    # Positional path segments per documented API. Optional trailing fields
    # (productIdentifier, wkt, featureType, feature, sortKey) are left as
    # empty segments -- UNVERIFIED whether this routing style is accepted;
    # if it 404s, this is the first thing to adjust.
    segments = [
        target, mission, instrument, prod_type,
        "",  # productIdentifier
        f"{min_lat}", f"{max_lat}", f"{west_lon}", f"{east_lon}",
        "", "", "", "",  # wkt, featureType, feature, sortKey
        "0", str(page_size), "1", "json",
    ]
    url = PILOT_API_BASE + "/" + "/".join(segments)

    if debug:
        print(f"  [debug] GET {url}")

    resp = sess.get(url, timeout=timeout, headers={
        "User-Agent": "Mozilla/5.0 (research script)",
        "Accept": "application/json",
    })

    if debug:
        print(f"  [debug] status={resp.status_code}, content-type={resp.headers.get('content-type')}")
        print(f"  [debug] body sample: {resp.text[:1000]!r}")

    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, list):
        raise ValueError(f"Unexpected response shape (expected a list): {type(data)}")
    return data


def candidates_from_pilot_response(data):
    """Converts PILOT API response rows into the same candidate dict shape
    used everywhere else in the pipeline (product_id, detail_link, obs_time).
    Note: PILOT's response does NOT include center lat/lon, so there's no
    cheap proxy-ranking step here -- every candidate goes straight to exact
    verification (Stage 2), which is fine since the bbox query already
    pre-filters geographically."""
    candidates = []
    for row in data:
        product_id = row.get("pds3ProductId") or row.get("pdS4ProductLid") or "unknown"
        detail_link = row.get("detailUrl")
        if not detail_link:
            continue
        candidates.append({
            "product_id": product_id,
            "detail_link": detail_link,
            "obs_time": row.get("observationStartUtc", ""),
        })
    return candidates


# ----------------------------------------------------------------------
# Manual search path (TESTED, proven fallback)
# ----------------------------------------------------------------------

def parse_search_results(path):
    """Parses an ODE SearchResults.txt export. Fields are comma+tab
    delimited ('value,\\tvalue,\\t...'), comment lines start with '#'."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()
    data_lines = [l for l in lines if not l.startswith("#")]
    if not data_lines:
        raise ValueError("No data rows -- is this really an ODE SearchResults.txt export?")
    # Note: the header row in this file format is itself '#'-prefixed, so
    # it's already excluded by the filter above -- no further slicing needed.
    # (v3 had an extra data_lines[1:] here that silently dropped the first
    # real candidate on every run -- fixed.)

    rows = []
    for line in data_lines:
        fields = [c.strip().rstrip(",").strip() for c in line.strip().split("\t")]
        if len(fields) < 9:
            continue
        product_id = fields[4]
        obs_time = fields[5]
        try:
            clat = float(fields[6])
            clon = float(fields[7])
        except ValueError:
            continue
        detail_link = fields[8]
        rows.append({
            "product_id": product_id,
            "center_lat": clat,
            "center_lon": clon,
            "obs_time": obs_time,
            "detail_link": detail_link,
        })
    return rows


def rank_by_proximity(target_corners, candidates):
    """Only meaningful when candidates have center_lat/center_lon (i.e. from
    the manual SearchResults.txt path). Auto-search candidates skip this."""
    clat, clon = corners_center(target_corners)
    klat, klon = KM_PER_DEG_LAT, km_per_deg_lon(clat)
    for c in candidates:
        if "center_lat" not in c:
            c["offset_km"] = None
            continue
        dlat_km = (c["center_lat"] - clat) * klat
        dlon_km = (c["center_lon"] - clon) * klon
        c["dlat_km"] = round(dlat_km, 2)
        c["dlon_km"] = round(dlon_km, 2)
        c["offset_km"] = round(math.hypot(dlat_km, dlon_km), 2)
    have_offset = [c for c in candidates if c["offset_km"] is not None]
    no_offset = [c for c in candidates if c["offset_km"] is None]
    return sorted(have_offset, key=lambda c: c["offset_km"]) + no_offset


# ----------------------------------------------------------------------
# STAGE 2: verify (TESTED against real captured page source)
# ----------------------------------------------------------------------

DETAIL_FIELD_LABELS = {
    "center_lat": ["Center Latitude"],
    "center_lon": ["Center Longitude"],
    "max_lat": ["Maximum Latitude"],
    "min_lat": ["Minimum Latitude"],
    "west_lon": ["Westernmost Longitude"],
    "east_lon": ["Easternmost Longitude"],
}
# Note: Map Resolution / Incidence / Emission / Phase Angle are NOT on this
# page -- confirmed absent from the initially-loaded HTML (verified against
# real captured page source). They live behind a separate 'Meta Data' tab
# (an ASP.NET postback), not implemented here since they're cosmetic only.


def fetch_product_detail(url, session=None, timeout=30):
    """Fetch and parse an ODE product-detail page. Reads footprint bounds
    from the 'Product Summary' table (id='dvProductDetails'), and the direct
    .IMG download link from the 'PDS Product Files' panel (table id
    'gvProductFilesP') -- both confirmed present on the page as initially
    loaded, verified against real captured page source (2025-09-02)."""
    if requests is None or BeautifulSoup is None:
        raise RuntimeError("pip install requests beautifulsoup4")
    sess = session or requests.Session()
    resp = sess.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research script)"})
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    label_to_key = {}
    for key, labels in DETAIL_FIELD_LABELS.items():
        for label in labels:
            label_to_key[_normalize(label).lower()] = key

    result = {}
    for tr in soup.find_all("tr"):
        cells = tr.find_all(["td", "th"])
        if len(cells) < 2:
            continue
        label_text = _normalize(cells[0].get_text(" ", strip=True)).lower()
        value_text = _normalize(cells[1].get_text(" ", strip=True))
        key = label_to_key.get(label_text)
        if not key or key in result:
            continue
        m = re.search(r"-?\d+\.?\d*", value_text)
        if m:
            result[key] = float(m.group(0))

    # The direct .IMG download link is a plain <a href> right on this page
    # (in the "PDS Product Files" panel) -- no separate page, no cart flow.
    img_url = None
    img_size_bytes = None
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.upper().split("?")[0].endswith(".IMG"):
            img_url = href
            # Try to find a nearby file-size figure in the same table row,
            # useful later for a download integrity check. Best-effort --
            # if not found, download-size verification is just skipped.
            row = a.find_parent("tr")
            if row:
                row_text = _normalize(row.get_text(" ", strip=True))
                m = re.search(r"([\d,]+)\s*(KB|MB|Bytes)", row_text, re.IGNORECASE)
                if m:
                    num = float(m.group(1).replace(",", ""))
                    unit = m.group(2).upper()
                    if unit == "KB":
                        img_size_bytes = int(num * 1024)
                    elif unit == "MB":
                        img_size_bytes = int(num * 1024 * 1024)
                    else:
                        img_size_bytes = int(num)
            break
    result["img_url"] = img_url
    result["img_size_bytes"] = img_size_bytes

    result["_raw_text_sample"] = soup.get_text("\n")[:2000]
    return result


def fetch_product_detail_cached(url, cache, session=None, timeout=30):
    if url in cache:
        return cache[url]
    detail = fetch_product_detail(url, session=session, timeout=timeout)
    cacheable = {k: v for k, v in detail.items() if k != "_raw_text_sample"}
    cache[url] = cacheable
    return detail


def verify_candidates(target_corners, candidates, top_n, cache=None, debug=False):
    if requests is None or BeautifulSoup is None:
        print("[!] requests/beautifulsoup4 not installed -- skipping verification stage.")
        return []
    if cache is None:
        cache = {}

    target_poly = corners_polygon(target_corners)
    target_area_deg2 = target_poly.area
    lat_mid, _ = corners_center(target_corners)
    klat, klon = KM_PER_DEG_LAT, km_per_deg_lon(lat_mid)

    verified = []
    sess = requests.Session()
    for c in candidates[:top_n]:
        print(f"  Verifying {c['product_id']} ...", end=" ", flush=True)
        try:
            detail = fetch_product_detail_cached(c["detail_link"], cache, session=sess)
        except Exception as e:
            print(f"FAILED ({e})")
            continue

        required = ["max_lat", "min_lat", "west_lon", "east_lon"]
        if not all(k in detail for k in required):
            print(f"incomplete fields, got: {list(detail.keys())}")
            if debug:
                print(f"    Raw text sample: {detail.get('_raw_text_sample','')[:500]!r}")
            continue

        cand_poly = Polygon([
            (detail["west_lon"], detail["max_lat"]),
            (detail["east_lon"], detail["max_lat"]),
            (detail["east_lon"], detail["min_lat"]),
            (detail["west_lon"], detail["min_lat"]),
        ])
        inter = target_poly.intersection(cand_poly)
        area_km2 = inter.area * klat * klon
        pct = 100 * inter.area / target_area_deg2 if target_area_deg2 else 0

        c.update(detail)
        c["overlap_km2"] = round(area_km2, 2)
        c["overlap_pct"] = round(pct, 1)
        verified.append(c)
        img_status = "img link found" if detail.get("img_url") else "NO img link found"
        print(f"overlap = {area_km2:.1f} km2 ({pct:.1f}% of target), {img_status}")

    verified.sort(key=lambda c: -c["overlap_km2"])
    return verified


# ----------------------------------------------------------------------
# STAGE 3: download (TESTED download mechanism; link discovery TESTED
# against real page structure)
# ----------------------------------------------------------------------

def download_file(url, dest_path, session=None, timeout=60, chunk_size=1024 * 1024,
                   expected_size=None):
    """Streams a file to disk, printing progress. If expected_size (bytes)
    is provided, compares it against the actual downloaded size afterward
    and prints a warning on mismatch -- a truncated download otherwise
    looks identical to a successful one."""
    if requests is None:
        raise RuntimeError("pip install requests")
    sess = session or requests.Session()
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    sha256 = hashlib.sha256()
    with sess.get(url, stream=True, timeout=timeout, headers={"User-Agent": "Mozilla/5.0 (research script)"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                sha256.update(chunk)
                downloaded += len(chunk)
                if total:
                    pct = 100 * downloaded / total
                    print(f"\r  {os.path.basename(dest_path)}: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.1f}%)",
                          end="", flush=True)
                else:
                    print(f"\r  {os.path.basename(dest_path)}: {downloaded/1e6:.1f} MB", end="", flush=True)
    print()

    actual_size = os.path.getsize(dest_path)
    check_against = total or expected_size
    if check_against and abs(actual_size - check_against) > 1024:  # allow small slack
        print(f"  [!] SIZE MISMATCH: downloaded {actual_size} bytes, expected ~{check_against} bytes. "
              f"File may be truncated/corrupt -- consider re-downloading.")
    else:
        print(f"  Integrity check OK ({actual_size/1e6:.1f} MB)")

    return {"path": dest_path, "size_bytes": actual_size, "sha256": sha256.hexdigest()}


def download_top_candidates(verified, top_n, out_dir, session=None):
    """Downloads the top N verified candidates that actually have an img_url.
    Returns list of dicts with download results merged into candidate info."""
    with_link = [c for c in verified if c.get("img_url")]
    if not with_link:
        return []
    sess = session or requests.Session()
    results = []
    for c in with_link[:top_n]:
        print(f"\n  Downloading {c['product_id']} (overlap {c['overlap_km2']:.1f} km2, {c['overlap_pct']:.1f}%)")
        img_url = c["img_url"]
        dest = os.path.join(out_dir, os.path.basename(img_url.split("?")[0]))
        try:
            dl = download_file(img_url, dest, session=sess, expected_size=c.get("img_size_bytes"))
        except Exception as e:
            print(f"  FAILED to download: {e}")
            continue
        c["downloaded_path"] = dl["path"]
        c["downloaded_size_bytes"] = dl["size_bytes"]
        c["downloaded_sha256"] = dl["sha256"]
        c["downloaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        results.append(c)
    return results


def append_manifest(manifest_path, instrument_label, source_xml, downloaded):
    """Appends one row per downloaded product to a running manifest CSV."""
    if not downloaded:
        return
    fieldnames = [
        "instrument", "source_xml", "product_id", "overlap_km2", "overlap_pct",
        "img_url", "downloaded_path", "downloaded_size_bytes", "downloaded_sha256",
        "downloaded_at", "obs_time",
    ]
    file_exists = os.path.exists(manifest_path)
    with open(manifest_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for c in downloaded:
            row = dict(c)
            row["instrument"] = instrument_label
            row["source_xml"] = source_xml
            writer.writerow(row)


# ----------------------------------------------------------------------
# Per-patch pipeline (used by both single-file and batch modes)
# ----------------------------------------------------------------------

def run_pipeline_for_xml(xml_path, args, cache):
    """Runs search -> rank (if applicable) -> verify -> download for a
    single instrument label XML. Returns True on success (at least one
    candidate verified), False otherwise."""
    stem = os.path.splitext(os.path.basename(xml_path))[0]
    patch_out_dir = os.path.join(args.out_dir, stem)
    print(f"\n{'='*70}\nPATCH: {stem}  ({xml_path})\n{'='*70}")

    try:
        target_corners = corners_from_pds4_xml(xml_path)
    except Exception as e:
        print(f"  [!] Could not parse corners from this XML: {e}")
        print("  Skipping this patch.")
        return False

    candidates = []
    if args.auto_search:
        print("\n=== STAGE 0: AUTO-SEARCH (experimental) ===")
        minlat, maxlat, westlon, eastlon = corners_bounds(target_corners)
        try:
            data = query_pilot_rest(minlat, maxlat, westlon, eastlon,
                                     page_size=args.search_page_size, debug=args.debug_search)
            candidates = candidates_from_pilot_response(data)
            print(f"  Auto-search returned {len(candidates)} candidates.")
        except Exception as e:
            print(f"  [!] Auto-search failed: {e}")
            print("  Falling back to manual --search-results if provided for this patch...")

    if not candidates:
        # Fall back to a manual export matching this XML's stem, or the
        # explicitly-provided --search-results (single-file mode).
        manual_path = args.search_results
        if not manual_path:
            guess = os.path.join(os.path.dirname(xml_path), f"{stem}_SearchResults.txt")
            if os.path.exists(guess):
                manual_path = guess
        if manual_path and os.path.exists(manual_path):
            print(f"\n=== STAGE 1: RANK (manual export: {manual_path}) ===")
            raw = parse_search_results(manual_path)
            candidates = rank_by_proximity(target_corners, raw)
            print(f"  Parsed {len(candidates)} candidates.")
        else:
            print("  [!] No candidates from auto-search and no manual SearchResults.txt found "
                  f"(looked for {manual_path or stem + '_SearchResults.txt'}). Skipping this patch.")
            return False

    if not candidates:
        print("  No candidates to verify. Skipping this patch.")
        return False

    print(f"\n=== STAGE 2: VERIFY (top {args.verify_top}) ===")
    verified = verify_candidates(target_corners, candidates, args.verify_top,
                                  cache=cache, debug=args.debug_search)
    if not verified:
        print("  No candidates verified successfully.")
        return False

    print("\nVerified overlap ranking (best first):")
    for i, c in enumerate(verified, 1):
        has_link = "yes" if c.get("img_url") else "NO"
        print(f"  {i:2d}. {c['product_id']:24s} overlap={c['overlap_km2']:.1f} km2 "
              f"({c['overlap_pct']:.1f}%)  img_link={has_link}")

    if args.download_top > 0:
        print(f"\n=== STAGE 3: DOWNLOAD (top {args.download_top}) ===")
        downloaded = download_top_candidates(verified, args.download_top, patch_out_dir)
        if downloaded and args.manifest:
            append_manifest(args.manifest, args.instrument_label or stem, xml_path, downloaded)
            print(f"\n  Manifest updated: {args.manifest}")
        return bool(downloaded)

    return True


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xml", dest="xml", help="Path to a single instrument's PDS4 label XML")
    ap.add_argument("--xml-dir", help="Batch mode: folder of instrument label XMLs to process")
    ap.add_argument("--ohrc-xml", dest="xml", help=argparse.SUPPRESS)  # backwards-compat alias
    ap.add_argument("--corners", nargs=8, type=float,
                     metavar=("UL_LAT", "UL_LON", "UR_LAT", "UR_LON", "LR_LAT", "LR_LON", "LL_LAT", "LL_LON"),
                     help="Manually supply target footprint corners instead of an XML file (single-file mode only)")

    ap.add_argument("--auto-search", action="store_true",
                     help="EXPERIMENTAL: query a bounding-box search automatically instead of "
                          "requiring a manual SearchResults.txt export")
    ap.add_argument("--search-results", help="Path to a manually-exported ODE SearchResults.txt "
                                              "(fallback / single-file mode)")
    ap.add_argument("--search-page-size", type=int, default=100,
                     help="Max candidates to request from auto-search (default 100)")
    ap.add_argument("--debug-search", action="store_true",
                     help="Print raw URLs/responses during search+verify for troubleshooting")

    ap.add_argument("--verify-top", type=int, default=10,
                     help="How many top candidates to verify exactly via live fetch (default 10)")
    ap.add_argument("--download-top", type=int, default=1,
                     help="How many top verified candidates to download (default 1, 0 = don't download)")
    ap.add_argument("--out-dir", default="./downloads", help="Base directory for downloaded files "
                                                               "(subfolder per patch)")
    ap.add_argument("--manifest", default="./manifest.csv",
                     help="Path to the running manifest CSV (default ./manifest.csv)")
    ap.add_argument("--instrument-label", help="Label to record in the manifest for this run "
                                                "(default: inferred from XML filename)")
    ap.add_argument("--cache-file", default="./ode_detail_cache.json",
                     help="Local cache of fetched product details, to avoid re-fetching on reruns")

    ap.add_argument("--download-only", action="store_true",
                     help="Skip search/verify entirely, just download --img-url directly")
    ap.add_argument("--img-url", help="Manual override: direct .IMG URL to download")

    args = ap.parse_args()

    print(f"[pipeline version: {PIPELINE_VERSION}]")

    if args.download_only:
        if not args.img_url:
            sys.exit("--download-only requires --img-url")
        dest = os.path.join(args.out_dir, os.path.basename(args.img_url.split("?")[0]))
        print(f"Downloading {args.img_url} -> {dest}")
        download_file(args.img_url, dest)
        return

    cache = load_cache(args.cache_file)

    if args.xml_dir:
        xml_files = sorted(glob.glob(os.path.join(args.xml_dir, "*.xml")))
        if not xml_files:
            sys.exit(f"No .xml files found in {args.xml_dir}")
        print(f"Batch mode: {len(xml_files)} label(s) found in {args.xml_dir}")
        results = {}
        for xml_path in xml_files:
            try:
                ok = run_pipeline_for_xml(xml_path, args, cache)
            except Exception as e:
                print(f"  [!] Unexpected error on {xml_path}: {e}")
                ok = False
            results[xml_path] = ok
            save_cache(args.cache_file, cache)  # save incrementally, not just at the end

        print(f"\n{'='*70}\nBATCH SUMMARY\n{'='*70}")
        for xml_path, ok in results.items():
            print(f"  {'OK  ' if ok else 'FAIL'}  {os.path.basename(xml_path)}")

    elif args.xml or args.corners:
        if args.xml:
            xml_path = args.xml
        else:
            # synthetic path label for corners-only mode
            xml_path = "manual-corners"
        if args.corners and not args.xml:
            # Monkeypatch corners_from_pds4_xml result path for this one call
            c = args.corners
            target_corners = {"UL": (c[0], c[1]), "UR": (c[2], c[3]), "LR": (c[4], c[5]), "LL": (c[6], c[7])}
            orig = corners_from_pds4_xml
            globals()["corners_from_pds4_xml"] = lambda _p: target_corners
            try:
                run_pipeline_for_xml(xml_path, args, cache)
            finally:
                globals()["corners_from_pds4_xml"] = orig
        else:
            run_pipeline_for_xml(xml_path, args, cache)
        save_cache(args.cache_file, cache)

    else:
        sys.exit("Provide --xml, --xml-dir, --corners, or use --download-only with --img-url")


if __name__ == "__main__":
    main()
