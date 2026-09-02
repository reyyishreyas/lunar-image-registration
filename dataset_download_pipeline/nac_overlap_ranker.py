#!/usr/bin/env python3
"""
nac_overlap_ranker.py

Ranks LRO ODE search-result candidates (NAC, or any instrument) by how well
their footprint likely overlaps a target Chandrayaan-2 instrument's footprint
(OHRC, TMC, or IIRS), using center-point proximity as a fast proxy.

Why a proxy instead of exact polygon intersection: ODE's basic search-result
export only gives you CENTER lat/lon per product, not full corner coordinates.
Fetching full footprint corners for every candidate (100+ products) means
hitting each product's detail page individually, which is slow. Center-point
distance turns out to be a strong predictor of overlap when the target
footprint is narrow in one axis (true for OHRC's ~3km-wide strips) --
validated against 5 manually-checked NAC/OHRC pairs where this ranking
correctly predicted from 93% overlap down to 1% overlap, in order.

USAGE:
    python3 nac_overlap_ranker.py --ohrc-xml path/to/ohrc_d_img_label.xml \
        --search-results path/to/SearchResults.txt \
        --top 15 \
        --out ranked_candidates.csv

    Or pass corners manually if you don't have a PDS4 XML label handy:
    python3 nac_overlap_ranker.py --corners UL_LAT UL_LON UR_LAT UR_LON \
        LR_LAT LR_LON LL_LAT LL_LON --search-results path/to/SearchResults.txt

NOTE: this is a SCREENING tool, not a final answer. Always pull full metadata
(footprint corners, incidence angle, resolution) for your top few candidates
from ODE's product detail page before committing to a multi-hundred-MB download.
"""
import argparse
import csv
import math
import xml.etree.ElementTree as ET


PDS4_NS = {"pds": "http://pds.nasa.gov/pds4/pds/v1"}
ISDA_NS = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}


def corners_from_pds4_xml(xml_path):
    """Extract Refined_Corner_Coordinates (lat, lon) from a Chandrayaan-2
    PDS4 ISDA-schema label (works for OHRC 'd_img' labels; TMC/IIRS labels
    may use a different tag structure -- check the XML if this fails, since
    field names should never be guessed for a new product type)."""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    geom = root.find(".//isda:Geometry_Parameters/isda:Refined_Corner_Coordinates", ISDA_NS)
    if geom is None:
        geom = root.find(".//isda:Geometry_Parameters/isda:System_Level_Coordinates", ISDA_NS)
    if geom is None:
        raise ValueError(
            "Could not find corner coordinates in this XML. "
            "This script was built against OHRC 'd_img' labels -- if this is "
            "a TMC or IIRS label, check the actual tag names and adjust, "
            "don't assume they match."
        )

    def get(tag):
        el = geom.find(f"isda:{tag}", ISDA_NS)
        return float(el.text)

    ul = (get("upper_left_latitude"), get("upper_left_longitude"))
    ur = (get("upper_right_latitude"), get("upper_right_longitude"))
    lr = (get("lower_right_latitude"), get("lower_right_longitude"))
    ll = (get("lower_left_latitude"), get("lower_left_longitude"))
    return ul, ur, lr, ll


def center_and_scale(corners):
    """corners: list of (lat, lon) tuples. Returns (center_lat, center_lon, km_per_deg_lat, km_per_deg_lon)."""
    lats = [c[0] for c in corners]
    lons = [c[1] for c in corners]
    center_lat = sum(lats) / len(lats)
    center_lon = sum(lons) / len(lons)
    km_per_deg_lat = 30.3  # ~constant everywhere on the Moon
    km_per_deg_lon = 30.3 * math.cos(math.radians(abs(center_lat)))
    return center_lat, center_lon, km_per_deg_lat, km_per_deg_lon


def parse_ode_search_results(path):
    """Parses an ODE 'SearchResults.txt' export (comment lines start with #,
    fields are comma+tab delimited: 'value,\\tvalue,\\t...')."""
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    data_lines = [l for l in lines if not l.startswith("#")]
    if not data_lines:
        raise ValueError("No data rows found -- is this an ODE SearchResults.txt export?")
    data_lines = data_lines[1:]  # drop header row

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
            continue  # row without valid center coords, skip
        detail_link = fields[8]
        rows.append((product_id, clat, clon, obs_time, detail_link))
    return rows


def rank_candidates(target_corners, candidates):
    center_lat, center_lon, km_lat, km_lon = center_and_scale(target_corners)
    ranked = []
    for product_id, clat, clon, obs_time, link in candidates:
        dlat_km = (clat - center_lat) * km_lat
        dlon_km = (clon - center_lon) * km_lon
        dist_km = math.hypot(dlat_km, dlon_km)
        ranked.append(
            {
                "product_id": product_id,
                "center_lat": clat,
                "center_lon": clon,
                "dlat_km": round(dlat_km, 2),
                "dlon_km": round(dlon_km, 2),
                "offset_km": round(dist_km, 2),
                "obs_time": obs_time,
                "detail_link": link,
            }
        )
    ranked.sort(key=lambda r: r["offset_km"])
    return ranked


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ohrc-xml", help="Path to a Chandrayaan-2 PDS4 label XML with corner coordinates")
    ap.add_argument(
        "--corners",
        nargs=8,
        type=float,
        metavar=("UL_LAT", "UL_LON", "UR_LAT", "UR_LON", "LR_LAT", "LR_LON", "LL_LAT", "LL_LON"),
        help="Manually supply the 4 corners instead of an XML file",
    )
    ap.add_argument("--search-results", required=True, help="Path to ODE SearchResults.txt export")
    ap.add_argument("--top", type=int, default=15, help="How many top candidates to print (default 15)")
    ap.add_argument("--out", default="ranked_candidates.csv", help="Output CSV path (full ranked list)")
    args = ap.parse_args()

    if args.ohrc_xml:
        corners = corners_from_pds4_xml(args.ohrc_xml)
    elif args.corners:
        c = args.corners
        corners = [(c[0], c[1]), (c[2], c[3]), (c[4], c[5]), (c[6], c[7])]
    else:
        raise SystemExit("Provide either --ohrc-xml or --corners")

    candidates = parse_ode_search_results(args.search_results)
    ranked = rank_candidates(corners, candidates)

    print(f"Parsed {len(candidates)} candidates from {args.search_results}\n")
    print(f"Top {min(args.top, len(ranked))} by center-proximity to target footprint:\n")
    header = f"{'#':>3s} {'Product ID':24s} {'CenterLat':>10s} {'CenterLon':>10s} {'dLat(km)':>9s} {'dLon(km)':>9s} {'Offset(km)':>11s}  ObsTime"
    print(header)
    for i, r in enumerate(ranked[: args.top], 1):
        print(
            f"{i:3d} {r['product_id']:24s} {r['center_lat']:10.3f} {r['center_lon']:10.3f} "
            f"{r['dlat_km']:9.2f} {r['dlon_km']:9.2f} {r['offset_km']:11.2f}  {r['obs_time']}"
        )

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ranked[0].keys())
        writer.writeheader()
        writer.writerows(ranked)
    print(f"\nFull ranked list ({len(ranked)} rows) written to {args.out}")
    print(
        "\nReminder: this is a proxy ranking. Pull full metadata (footprint corners, "
        "incidence angle, resolution) for your top few picks from ODE before downloading."
    )


if __name__ == "__main__":
    main()
