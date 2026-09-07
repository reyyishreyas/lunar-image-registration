#!/usr/bin/env python3
"""
get_search_results.py

Give it any Chandrayaan-2 instrument PDS4 label XML (OHRC, TMC, or IIRS).
It will:
  1. Read the bounding box from the XML automatically.
  2. Query NASA ODE REST API directly — no manual form, no SearchResults.txt.
  3. Rank all candidates by real polygon overlap with your instrument footprint.
  4. Among the 100% overlap candidates, pick the one with lowest incidence angle
     (least shadow, best for feature matching).
  5. Download the winning NAC .IMG file straight to disk.

USAGE:
    python get_search_results.py your_label_d_img.xml
    python get_search_results.py your_label_d_img.xml --out-dir D:/SIH/data
    python get_search_results.py your_label_d_img.xml --top 3
"""
import argparse, json, math, os, re, sys, time, xml.etree.ElementTree as ET
try:
    import requests
except ImportError:
    sys.exit("Run: pip install requests")
try:
    from shapely import wkt as shapely_wkt
    from shapely.geometry import Polygon
except ImportError:
    sys.exit("Run: pip install shapely")

ISDA_NS  = {"isda": "https://isda.issdc.gov.in/pds4/isda/v1"}
ODE_REST = "https://oderest.rsl.wustl.edu/live2/"
KM_LAT   = 30.3


# ── 1. Read corners from XML ─────────────────────────────────────────────────

def get_corners(xml_path):
    root = ET.parse(xml_path).getroot()
    geom = root.find(".//isda:Geometry_Parameters/isda:Refined_Corner_Coordinates", ISDA_NS)
    if geom is None:
        geom = root.find(".//isda:Geometry_Parameters/isda:System_Level_Coordinates", ISDA_NS)
    if geom is None:
        raise ValueError(
            "No corner coordinates found. Tested on OHRC '_d_img_' labels.\n"
            "If this is TMC or IIRS, send the XML to Claude so correct tags can be confirmed."
        )
    def g(tag):
        el = geom.find(f"isda:{tag}", ISDA_NS)
        if el is None: raise ValueError(f"Missing tag: isda:{tag}")
        return float(el.text)
    lats = [g("upper_left_latitude"), g("upper_right_latitude"),
            g("lower_right_latitude"), g("lower_left_latitude")]
    lons = [g("upper_left_longitude"), g("upper_right_longitude"),
            g("lower_right_longitude"), g("lower_left_longitude")]
    corners = {
        "UL": (g("upper_left_latitude"),  g("upper_left_longitude")),
        "UR": (g("upper_right_latitude"), g("upper_right_longitude")),
        "LR": (g("lower_right_latitude"), g("lower_right_longitude")),
        "LL": (g("lower_left_latitude"),  g("lower_left_longitude")),
    }
    return corners, min(lats), max(lats), min(lons), max(lons)


def corners_to_polygon(corners):
    ring = [corners["UL"], corners["UR"], corners["LR"], corners["LL"]]
    poly = Polygon([(lon, lat) for lat, lon in ring])
    return poly if poly.is_valid else poly.buffer(0)


# ── 2. Query ODE REST API ────────────────────────────────────────────────────

def query_ode(minlat, maxlat, westlon, eastlon, page_size=500):
    params = dict(query="product", target="moon", results="fmpc",
                  ihid="LRO", iid="LROC", pt="CDRNAC4", output="JSON",
                  minlat=minlat, maxlat=maxlat, westlon=westlon, eastlon=eastlon,
                  limit=page_size)
    print(f"  Querying ODE REST API...")
    resp = requests.get(ODE_REST, params=params, timeout=60,
                        headers={"User-Agent": "Mozilla/5.0 (research script)"})
    resp.raise_for_status()
    data = resp.json()
    products = data.get("ODEResults", {}).get("Products", {}).get("Product", [])
    if isinstance(products, dict):
        products = [products]
    print(f"  Got {len(products)} candidates from ODE.")
    return products


# ── 3. Rank by overlap ───────────────────────────────────────────────────────

def rank_products(products, target_poly):
    lats = [pt[0] for pt in target_poly.exterior.coords]
    lat_mid = sum(lats) / len(lats)
    km_lon = KM_LAT * math.cos(math.radians(abs(lat_mid)))
    target_area = target_poly.area

    ranked = []
    for p in products:
        pid = p.get("pdsid", "unknown")
        wkt_str = p.get("Footprint_geometry", "")
        try:
            cand_poly = shapely_wkt.loads(wkt_str)
            if not cand_poly.is_valid:
                cand_poly = cand_poly.buffer(0)
            inter = target_poly.intersection(cand_poly)
            overlap_km2 = inter.area * KM_LAT * km_lon
            overlap_pct = 100 * inter.area / target_area if target_area else 0
        except Exception:
            overlap_km2, overlap_pct = 0.0, 0.0

        try:
            incidence = float(p.get("Incidence_angle", 999))
        except (ValueError, TypeError):
            incidence = 999.0

        # Direct .IMG download URL from Product_files
        pf = p.get("Product_files", {}).get("Product_file", [])
        if isinstance(pf, dict):
            pf = [pf]
        img_url = next(
            (f["URL"] for f in pf if f.get("FileName", "").upper().endswith(".IMG")),
            None
        )

        ranked.append({
            "product_id":  pid,
            "overlap_km2": round(overlap_km2, 2),
            "overlap_pct": round(overlap_pct, 1),
            "incidence":   incidence,
            "img_url":     img_url,
            "obs_time":    p.get("Observation_time", ""),
        })

    # Sort: highest overlap first, then lowest incidence as tiebreaker
    ranked.sort(key=lambda r: (-r["overlap_pct"], r["incidence"]))
    return ranked


# ── 4. Download ──────────────────────────────────────────────────────────────

def download(url, dest_path, chunk=1024 * 1024):
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=120,
                      headers={"User-Agent": "Mozilla/5.0 (research script)"}) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done  = 0
        with open(dest_path, "wb") as f:
            for ch in r.iter_content(chunk_size=chunk):
                if ch:
                    f.write(ch)
                    done += len(ch)
                    bar = f"{done/1e6:.1f}/{total/1e6:.1f} MB" if total else f"{done/1e6:.1f} MB"
                    pct = f" ({100*done/total:.0f}%)" if total else ""
                    print(f"\r  {os.path.basename(dest_path)}: {bar}{pct}", end="", flush=True)
    print()
    actual = os.path.getsize(dest_path)
    if total and abs(actual - total) > 1024:
        print(f"  [!] Size mismatch: got {actual} bytes, expected {total}. May be truncated.")
    else:
        print(f"  OK — {actual/1e6:.1f} MB written.")
    return dest_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("xml",       help="Instrument PDS4 label XML (e.g. ohrc_d_img.xml)")
    ap.add_argument("--out-dir", default="./downloads",
                    help="Where to save downloaded .IMG files (default: ./downloads)")
    ap.add_argument("--top",     type=int, default=1,
                    help="How many top candidates to download (default 1)")
    ap.add_argument("--no-download", action="store_true",
                    help="Just rank and print, don't download anything")
    args = ap.parse_args()

    if not os.path.exists(args.xml):
        sys.exit(f"File not found: {args.xml}")

    # 1. Parse XML
    print(f"\nReading: {args.xml}")
    corners, minlat, maxlat, westlon, eastlon = get_corners(args.xml)
    target_poly = corners_to_polygon(corners)
    print(f"  Bounding box: lat [{minlat:.4f}, {maxlat:.4f}]  lon [{westlon:.4f}, {eastlon:.4f}]")

    # 2. Query
    products = query_ode(minlat, maxlat, westlon, eastlon)
    if not products:
        sys.exit("No candidates returned. Try widening the bounding box.")

    # 3. Rank
    ranked = rank_products(products, target_poly)

    print(f"\n{'#':>3}  {'Product ID':25}  {'Overlap':>10}  {'% cover':>8}  {'Incidence':>10}  ObsTime")
    print("─" * 85)
    for i, r in enumerate(ranked[:20], 1):
        flag = " ← best" if i == 1 else ""
        print(f"{i:3d}  {r['product_id']:25}  {r['overlap_km2']:>8.1f}km²"
              f"  {r['overlap_pct']:>7.1f}%  {r['incidence']:>9.2f}°  {r['obs_time'][:10]}{flag}")

    # 4. Download top N
    if args.no_download:
        print("\n(--no-download set, stopping here)")
        return

    to_dl = [r for r in ranked if r["img_url"]][:args.top]
    if not to_dl:
        print("\n[!] No candidates with a direct download URL found.")
        return

    stem = os.path.splitext(os.path.basename(args.xml))[0]
    patch_dir = os.path.join(args.out_dir, stem)

    print(f"\nDownloading top {len(to_dl)} candidate(s) to {patch_dir}/")
    for r in to_dl:
        fname = os.path.basename(r["img_url"].split("?")[0])
        dest  = os.path.join(patch_dir, fname)
        print(f"\n  {r['product_id']}  (overlap {r['overlap_pct']:.1f}%, incidence {r['incidence']:.1f}°)")
        print(f"  → {dest}")
        download(r["img_url"], dest)

    print("\nDone.")


if __name__ == "__main__":
    main()
