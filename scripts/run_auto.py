#!/usr/bin/env python3
"""Fully-automatic registration CLI (Phase 8).

Drop in the three inputs and it does everything: georeference + equal-GSD
staging, detection/matching/robust-fit (champion config), automatic content-vs-
geometry decision, RMSE vs a reference (optional), and a JSON report + figures.

Usage:
    venv/bin/python scripts/run_auto.py \
        --ohrc_img data/PATCH-001/data/OHRC/.../ch2_ohr_ncp_2021..._d_img_d18.img \
        --ohrc_geom data/PATCH-001/data/OHRC/.../ch2_ohr_ncp_2021..._g_grd_d18.csv \
        --nac_img   data/PATCH-001/data/LRO_NAC/LC/M1469248775LC.IMG

    # optional:
        --nac_geom     <NAC SPICE geometry CSV, for equal-GSD seed / geometry fallback>
        --ground_truth <csv of x1,y1,x2,y2 sub-pixel controls>
        --out_dir data/processed/auto --prefix auto
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from src.auto_pipeline import run_auto, write_json_report, summarize

    ap = argparse.ArgumentParser(
        description="fully-automatic OHRC<->NAC registration")
    ap.add_argument("--ohrc_img", required=True, help="OHRC raw .img")
    ap.add_argument("--ohrc_geom", required=True,
                    help="OHRC ISRO geometry CSV (pixel,scan -> lon,lat)")
    ap.add_argument("--nac_img", required=True, help="LRO NAC .IMG")
    ap.add_argument("--nac_geom", default="",
                    help="NAC SPICE geometry CSV (equal-GSD seed / geometry fallback)")
    ap.add_argument("--ground_truth", default="data/ground_truth/pair1_gt_v2.csv",
                    help="csv x1,y1,x2,y2 reference (only pair-1 has one)")
    ap.add_argument("--no-ground-truth", action="store_true",
                    help="skip the reference check")
    ap.add_argument("--out_dir", default="data/processed/auto")
    ap.add_argument("--prefix", default="auto")
    ap.add_argument("--normalize", default="clahe",
                    choices=["clahe", "none", "edges", "histogram_match"])
    ap.add_argument("--equal-gsd", action="store_true",
                    help="stage both sensors to a common ground grid (SPICE seed)")
    args = ap.parse_args()

    gt = "" if args.no_ground_truth else args.ground_truth
    report = run_auto(
        args.ohrc_img, args.ohrc_geom, args.nac_img,
        out_dir=args.out_dir, prefix=args.prefix,
        nac_geom_csv=args.nac_geom, ground_truth=gt,
        normalize=args.normalize, equal_gsd=args.equal_gsd,
    )
    print("\n" + summarize(report))
    json_path = write_json_report(report, os.path.join(args.out_dir, "report.json"))
    print(f"\nreport -> {json_path}")
    for k, v in report.get("artifacts", {}).items():
        print(f"  {k}: {v}")
    return 0 if report.get("verdict") in ("registered", "geometry_registered") else 1


if __name__ == "__main__":
    sys.exit(main())
