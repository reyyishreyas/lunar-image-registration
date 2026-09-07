"""Phase 4 Step 4.1 grid-sweep runner.

Runs the Phase 3 best config (C3) with grid-based match capping at 4x4, 8x8,
16x16, trims outlier rejection noise via RMSE, logs to results/logs/grid_sweep.csv.
"""

from __future__ import annotations

import csv
import os

import yaml

import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.pipeline import run_experiment  # noqa: E402

import argparse


def main():
    parser = argparse.ArgumentParser(description="run grid-uniform cap sweep")
    parser.add_argument("--base", default="configs/experiment_C3.yaml")
    parser.add_argument("--out", default="results/logs/grid_sweep.csv")
    parser.add_argument("--max_total", type=int, default=200)
    args = parser.parse_args()

    base_cfg = yaml.safe_load(open(args.base, encoding="utf-8"))
    tmp_cfg = os.path.splitext(args.base)[0] + "_gridtmp.yaml"

    rows = []
    grids = [4, 8, 16]
    for g in grids:
        import math
        per_cell = math.ceil(args.max_total / (g * g))
        cfg = dict(base_cfg)
        cfg["outlier_rejection"] = {
            "name": "ransac",
            "method": "ransac",
            "ransac_thresh": 5.0,
            "grid_uniform": {
                "rows": g,
                "cols": g,
                "max_per_cell": per_cell,
                "max_total": args.max_total,
            },
        }
        cfg["experiment"]["id"] = f"GRID{g}x{g}"
        cfg["experiment"]["description"] = f"grid cap {g}x{g} per_cell={per_cell}"
        with open(tmp_cfg, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh)
        row = run_experiment(tmp_cfg, verbose=False)
        row["grid"] = f"{g}x{g}"
        row["max_per_cell"] = per_cell
        rows.append(row)
        print(f"grid {g}x{g}: rmse={row['rmse_px']} inliers={row['inliers']} "
              f"n={row['n_matches']} t={row['time_s']}s")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    cols = ["config_id", "grid", "max_per_cell", "rmse_px", "inliers",
            "inlier_ratio", "n_matches", "time_s"]
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})

    os.remove(tmp_cfg)
    best = min(rows, key=lambda r: float(r["rmse_px"]))
    print(f"\nbest grid: {best['grid']} rmse={best['rmse_px']}")


if __name__ == "__main__":
    main()