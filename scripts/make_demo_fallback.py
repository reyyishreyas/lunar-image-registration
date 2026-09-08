#!/usr/bin/env python3
"""Build demo/fallback/ artifacts (AI_EXECUTION_PLAN.md Phase 6, Section 9).

Run once before presenting:
    venv/bin/python scripts/make_demo_fallback.py

Artifacts:
  demo/fallback/cached_final_output.json  — FINAL_CONFIG row + figure/checker paths
  demo/fallback/pair1_matches_FINAL.png   — pipeline match overlay (copy)
  demo/fallback/pair1_checkerboard.png    — checkerboard of the staged crops
  demo/fallback/demo_recording.mp4        — short pre-recorded run (cv2, no ffmpeg)

Classical-only pipeline: safe to run live on a CPU machine; the cache/video are
the agreed fallback if the venue network or session is unavailable.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

FINAL_CONFIG = "results/final_config.yaml"
FALLBACK = os.path.join(ROOT, "demo", "fallback")
FALLBACK_OUT = os.path.join(ROOT, "demo", "fallback", "cached_final_output.json")


def _checkerboard(src, ref, tiles=8):
    h, w = src.shape[:2]
    out = np.zeros((h, w), np.uint8)
    th, tw = h // tiles, w // tiles
    for i in range(tiles):
        for j in range(tiles):
            out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = (
                ref[i * th:(i + 1) * th, j * tw:(j + 1) * tw]
                if (i + j) % 2 == 0
                else src[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
    return out


def _put_metric(frame, text, y):
    cv2.putText(frame, text, (60, y), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2, cv2.LINE_AA)


def _make_video(fig_path, checker_path, row):
    img = cv2.imread(fig_path)
    chk = cv2.imread(checker_path)
    h, w = img.shape[:2]

    titles = [
        "Lunar Image Registration (OHRC <-> LRO NAC)",
        "Pair-1: OHRC-2021  +  NAC M1469248775LC  |  FINAL_CONFIG",
        f"RMSE {row['rmse_px']} px   inliers {row['inliers']}   ratio {row['inlier_ratio']:.3f}",
    ]
    frames = []
    for t in titles:
        f = np.full((h, w, 3), 18, np.uint8)
        cv2.putText(f, t, (60, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 200, 255), 2, cv2.LINE_AA)
        frames.extend([f] * 12)

    frames.extend([img] * 30)
    frames.extend([chk] * 30)

    w1 = cv2.hconcat([cv2.resize(img, None, fx=0.5, fy=0.5),
                      cv2.resize(chk, None, fx=0.5, fy=0.5)])
    frames.extend([w1] * 24)

    path = os.path.join(FALLBACK, "demo_recording.mp4")
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 5.0,
                         (w, h))
    for f in frames:
        wr.write(f)
    wr.release()
    return path


def main():
    from src.pipeline import run_experiment

    os.makedirs(FALLBACK, exist_ok=True)
    row = run_experiment(os.path.join(ROOT, FINAL_CONFIG), verbose=False)

    src = cv2.imread(os.path.join(ROOT, "data/processed/pair1_src.png"),
                     cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(os.path.join(ROOT, "data/processed/pair1_ref.png"),
                     cv2.IMREAD_GRAYSCALE)
    checker_p = os.path.join(FALLBACK, "pair1_checkerboard.png")
    cv2.imwrite(checker_p, _checkerboard(src, ref))

    fig_p = os.path.join(FALLBACK, "pair1_matches_FINAL.png")
    srcfig = os.path.join(ROOT, "results/figures/pair1_matches_FINAL.png")
    os.makedirs(os.path.dirname(srcfig), exist_ok=True)
    if os.path.exists(srcfig):
        cv2.imwrite(fig_p, cv2.imread(srcfig))

    video_p = _make_video(fig_p, checker_p, row)

    cache = {"config": "results/final_config.yaml",
             "pair": "phase1",
             "rmse": row["rmse_px"], "inliers": row["inliers"],
             "inlier_ratio": row["inlier_ratio"], "n_matches": row["n_matches"],
             "time_s": row["time_s"],
             "fig": fig_p.replace(ROOT + "/", ""),
             "checker": checker_p.replace(ROOT + "/", ""),
             "video": video_p.replace(ROOT + "/", "")}
    with open(FALLBACK_OUT, "w") as fh:
        json.dump(cache, fh, indent=2)

    print(f"fallback cached output : {FALLBACK_OUT}")
    print(f"match overlay          : {fig_p}")
    print(f"checkerboard           : {checker_p}")
    print(f"pre-recorded video     : {video_p}")
    print({k: cache[k] for k in ("rmse", "inliers", "inlier_ratio", "n_matches")})


if __name__ == "__main__":
    main()