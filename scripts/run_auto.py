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

    Phase 9 multi-sensor (PS 26166):
        --sensor tmc-ohrc  --src_img <TMC .img> --src_geom <TMC geometry CSV> \
            --ref_img <OHRC .img> --ref_geom <OHRC geometry CSV>
        --sensor iirs-ohrc --src_img <IIRS .qub> --src_geom <IIRS .hdr> \
            --ref_img <OHRC .img> --ref_geom <OHRC geometry CSV>
        --sensor tmc-nac   --src_img <TMC .img> --src_geom <TMC geometry CSV> \
            --ref_img <LRO NAC .IMG> --ref_geom <NAC SPICE CSV>
        --sensor iirs-nac  --src_img <IIRS .qub> --src_geom <IIRS .hdr> \
            --ref_img <LRO NAC .IMG>
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def main():
    from src.auto_pipeline import run_auto, run_sensor_auto, write_json_report, summarize

    ap = argparse.ArgumentParser(
        description="fully-automatic lunar registration (CLI)")
    ap.add_argument("--sensor", default="ohrc-nac",
                    choices=["any", "ohrc-nac", "tmc-ohrc", "iirs-ohrc",
                             "tmc-nac", "iirs-nac", "ohrc-ohrc", "tmc-tmc",
                             "iirs-iirs", "ohrc-tmc"],
                    help="sensor pair to register; 'any' auto-detects each "
                         "file's sensor from its name (default ohrc-nac)")
    ap.add_argument("--src_img", required=True,
                    help="source: OHRC/TMC raw .img, or IIRS .qub")
    ap.add_argument("--src_geom", required=True,
                    help="source geometry: OHRC/TMC ISRO CSV, or IIRS .hdr")
    ap.add_argument("--ref_img", required=True,
                    help="reference: OHRC raw .img, LRO NAC .IMG, or IIRS .qub")
    ap.add_argument("--ref_geom", default="",
                    help="reference geometry: OHRC ISRO CSV, or NAC SPICE CSV")
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

    if args.sensor == "any":
        from src.auto_pipeline import detect_pair
        sensor_pair = detect_pair(args.src_img, args.ref_img)
        print(f"[run_auto] auto-detected sensor pair: {sensor_pair}")
    else:
        sensor_pair = args.sensor

    if sensor_pair == "ohrc-nac":
        gt = "" if args.no_ground_truth else args.ground_truth
        report = run_auto(
            args.src_img, args.src_geom, args.ref_img,
            out_dir=args.out_dir, prefix=args.prefix,
            nac_geom_csv=args.ref_geom, ground_truth=gt,
            normalize=args.normalize, equal_gsd=args.equal_gsd,
        )
    else:
        report = run_sensor_auto(
            sensor_pair, args.src_img, args.src_geom, args.ref_img, args.ref_geom,
            out_dir=args.out_dir, prefix=args.prefix,
        )
    print("\n" + summarize(report))
    json_path = write_json_report(report, os.path.join(args.out_dir, "report.json"))
    print(f"\nreport -> {json_path}")
    for k, v in report.get("artifacts", {}).items():
        print(f"  {k}: {v}")
    return 0 if report.get("verdict") in ("registered", "geometry_registered") else 1


if __name__ == "__main__":
    sys.exit(main())
