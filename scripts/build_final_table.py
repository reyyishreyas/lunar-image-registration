#!/usr/bin/env python3
"""Build the presentable ablation table (AI_EXECUTION_PLAN.md Step 6.2).

Reads results/logs/ablation.csv (raw pipeline rows) and emits
results/final_ablation_table.csv with exactly:
    Config ID, Preprocessing, Detector, Matcher, Outlier rejection, Refinement,
    RMSE(px), Inliers, Inlier ratio, Time(s)

Handles the Phase-5 C9 polar row (an extra trailing note column) and dedupes
repeated FINAL regression rows.
"""

from __future__ import annotations

import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "results", "logs", "ablation.csv")
DST = os.path.join(ROOT, "results", "final_ablation_table.csv")

RAW_COLS = ["config_id", "preproc", "detector", "matcher", "outlier",
            "refinement", "rmse_px", "inliers", "inlier_ratio", "n_matches",
            "time_s", "georef_s"]
OUT_COLS = ["Config ID", "Preprocessing", "Detector", "Matcher",
            "Outlier rejection", "Refinement", "RMSE(px)", "Inliers",
            "Inlier ratio", "Time(s)"]


def main():
    rows = []
    with open(SRC, newline="") as fh:
        rd = csv.reader(fh)
        header = next(rd)
        if header[0].lower() == "config_id":
            pass
        else:
            fh.seek(0)
            rd = csv.reader(fh)
        for r in rd:
            if not r or not r[0].strip():
                continue
            if len(r) < len(RAW_COLS):
                continue
            rows.append(dict(zip(RAW_COLS, r[:len(RAW_COLS)])))

    seen = set()
    keep = []
    for r in rows:  # dedupe identical regression rows, keep the last occurrence
        key = (r["config_id"], r["preproc"], r["detector"], r["matcher"],
               r["outlier"], r["refinement"], r["rmse_px"], r["inliers"],
               r["inlier_ratio"], r["n_matches"])
        seen.add(key)
    rows.reverse()
    for r in reversed(rows):
        key = (r["config_id"], r["preproc"], r["detector"], r["matcher"],
               r["outlier"], r["refinement"], r["rmse_px"], r["inliers"],
               r["inlier_ratio"], r["n_matches"])
        if key in seen:
            keep.append(r)
            seen.remove(key)
    keep.reverse()

    def _sort_key(r):
        _id = r["config_id"] + ("-polar" if "sun_el" in r["preproc"]
                                else "-grid" if "clahe:4.0" in r["preproc"]
                                and r["config_id"] == "C9" and r["rmse_px"] != "FAIL"
                                else "")
        r["config_id"] = _id
        m = re.fullmatch(r"C(\d+)(-\w+)?", _id)
        return (0, int(m.group(1))) if m else (1, _id)

    keep.sort(key=_sort_key)

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(DST, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(OUT_COLS)
        for r in keep:
            w.writerow([r[c] for c in RAW_COLS[:len(OUT_COLS)]])
    print(f"{len(keep)} configurations -> {DST}")


if __name__ == "__main__":
    sys.exit(main())