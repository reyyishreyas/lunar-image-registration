"""Phase 1 Step 1.6 - manual ground-truth control-point picking.

This step is manual and MUST be done by a human (AI_EXECUTION_PLAN.md Step 1.6).
Do not fabricate ground-truth points.

Usage (runs an interactive matplotlib window):
    python scripts/pick_ground_truth.py \
        --src data/processed/pair1_src.png \
        --ref data/processed/pair1_ref.png \
        --out data/ground_truth/pair1_gt.csv

Controls:
    Click alternately on the source image then on the corresponding feature in
    the reference image; each completed pair is appended and misclicks are
    marked. Keyboard:
        u   undo last point
        s   save what we have and quit
        q   quit without saving

Output CSV columns: x1,y1,x2,y2  (all in pixels).
"""

from __future__ import annotations

import argparse
import csv
import os

import cv2
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/processed/pair1_src.png")
    ap.add_argument("--ref", default="data/processed/pair1_ref.png")
    ap.add_argument("--out", default="data/ground_truth/pair1_gt.csv")
    ap.add_argument("--target", type=int, default=20)
    args = ap.parse_args()

    src = cv2.imread(args.src, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(args.ref, cv2.IMREAD_GRAYSCALE)
    if src is None or ref is None:
        raise SystemExit(f"cannot read {args.src} / {args.ref}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 12))
    ax1.set_title("SOURCE (click here, then on the SAME feature in the reference)")
    ax2.set_title("REFERENCE")
    im1 = ax1.imshow(src, cmap="gray")
    im2 = ax2.imshow(ref, cmap="gray")
    fig.tight_layout()

    pending = None
    pairs = []

    def redraw():
        for ax in (ax1, ax2):
            ax.cla()
        ax1.imshow(src, cmap="gray")
        ax2.imshow(ref, cmap="gray")
        for i, (a, b) in enumerate(pairs):
            ax1.plot(a[0], a[1], "ro", markersize=4)
            ax1.text(a[0], a[1], str(i), color="red", fontsize=8)
            ax2.plot(b[0], b[1], "ro", markersize=4)
            ax2.text(b[0], b[1], str(i), color="red", fontsize=8)
        if pending is not None:
            ax1.plot(pending[0], pending[1], "go", markersize=4)
        ax1.set_title(f"SOURCE  -  collected {len(pairs)}/{args.target} pairs")
        ax2.set_title("REFERENCE")
        fig.canvas.draw_idle()

    def onclick(event):
        nonlocal pending
        if event.inaxes is ax1 and pending is None:
            pending = (event.xdata, event.ydata)
        elif event.inaxes is ax2 and pending is not None:
            pairs.append((pending, (event.xdata, event.ydata)))
            pending = None
        redraw()

    def onkey(event):
        nonlocal pending
        if event.key == "u":
            if pending is not None:
                pending = None
            elif pairs:
                pairs.pop()
        elif event.key == "s":
            save()
        elif event.key == "q":
            plt.close(fig)
        redraw()

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x1", "y1", "x2", "y2"])
            for (a, b) in pairs:
                w.writerow([round(a[0], 2), round(a[1], 2), round(b[0], 2), round(b[1], 2)])
        print(f"saved {len(pairs)} ground-truth points -> {args.out}")
        plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", onclick)
    fig.canvas.mpl_connect("key_press_event", onkey)
    plt.show()


if __name__ == "__main__":
    main()