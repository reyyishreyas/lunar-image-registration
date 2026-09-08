#!/usr/bin/env python3
"""Per-line NAC ground geometry via SPICE kernels.

For a given LROC NAC image, compute the ground footprint point for a grid
of (line, sample) positions using the LRO SPICE archive kernels:
    required kernels live in data/spice/ and must cover the observation epoch.

Usage:
    venv/bin/python scripts/nac_geometry.py --nac data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG
        [--line-step 100] [--sample-step 200] [--out out.csv]

Output CSV columns (space-separated, like OHRC geometry CSV):
    Longitude  Latitude  Radius_km  Scan  Pixel

Kernels expected under data/spice/:
    naif0012.tls pck00010.tpc moon_pa_de421_1900_2050.bpc moon_080317.tf
    moon_assoc_me.tf lro_frames_2014049_v01.tf lro_clkcor_<epoch>.tsc
    lrorg_<epoch>.bsp de421.bsp lro_lroc_v20.ti lrolc_<epoch>.bc
"""
import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

try:
    import spiceypy as sp
except ImportError:
    sys.exit("spiceypy required: venv/bin/python -m pip install spiceypy")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
SPICE = DATA / "spice"

CORNERS_ODE = None  # optional (lon, lat) ground truth corners for validation


def fa(a, b, t):
    return a + (b - a) * t


def read_nac_label(path):
    raw = Path(path).read_bytes()
    head = raw[:4096].decode("latin-1", "replace")
    txt = raw[:20000].decode("latin-1", "replace")

    def g(name):
        m = re.search(name + r"\s*=\s*([^;\n]+)", txt)
        return m.group(1).strip().strip('"') if m else None

    return {
        "start": g("START_TIME"),
        "stop": g("STOP_TIME"),
        "sclk_start": g("SPACECRAFT_CLOCK_START_COUNT"),
        "sclk_stop": g("SPACECRAFT_CLOCK_STOP_COUNT"),
        "lines": int(g("LINES")),
        "samples": int(g("LINE_SAMPLES")),
        "frame": -85610 if "R" in str(Path(path).stem) else -85600,
    }


def load_kernels():
    files = sorted(SPICE.glob("*.tls")) + sorted(SPICE.glob("*.tpc")) + \
        sorted(SPICE.glob("*.bpc")) + sorted(SPICE.glob("*.tf")) + \
        sorted(SPICE.glob("*.tsc")) + sorted(SPICE.glob("lrorg_*.bsp")) + \
        sorted(SPICE.glob("de421.bsp")) + sorted(SPICE.glob("lro_lroc_*.ti")) + \
        sorted(SPICE.glob("lrolc_*.bc")) + sorted(SPICE.glob("lrosc_*.bc"))
    for f in files:
        if f.name.startswith("._"):
            continue
        if f.suffix in (".bsp", ".bpc") or f.suffix in (".tls", ".tpc", ".tf", ".tsc", ".ti", ".bc"):
            sp.furnsh(str(f))
            print(f"  furnsh {f.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nac", required=True)
    ap.add_argument("--line-step", type=int, default=100)
    ap.add_argument("--sample-step", type=int, default=200)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    lab = read_nac_label(args.nac)
    print(f"label: start={lab['start']} stop={lab['stop']} "
          f"lines={lab['lines']} samples={lab['samples']} frame={lab['frame']}")
    load_kernels()

    frame = lab["frame"]
    s0 = sp.scencd(-85, lab["sclk_start"])
    s1 = sp.scencd(-85, lab["sclk_stop"])
    NL = lab["lines"]
    tick_per_line = (s1 - s0) / (NL - 1.0)

    shp, fname, bsight, n, bounds = sp.getfov(frame, 512)
    by = np.array([b[1] for b in bounds])
    bmin, bmax = by.min(), by.max()   # across-track tangent extents at samples 0 and s-1
    NS = lab["samples"]

    def ray(sample):
        t = sample / (NS - 1.0)
        return np.array([0.0, fa(bmin, bmax, t), 1.0])

    radii = np.array(sp.bodvrd("MOON", "RADII", 3)[1])
    print(f"moon radii km: {radii}")

    def ground(et, tick, s):
        cm, _ = sp.ckgp(frame, tick, 0.0, "MOON_ME")
        pos, _ = sp.spkpos("LRO", et, "MOON_ME", "NONE", "MOON")
        d = cm.T @ ray(s)
        a, b, c = radii
        # ellipsoid x^2/a^2+y^2/b^2+z^2/c^2 = 1 ; point = pos + t*d
        q = np.array([1 / a**2, 1 / b**2, 1 / c**2])
        A = np.sum(q * d * d)
        B = 2 * np.sum(q * pos * d)
        C = np.sum(q * pos * pos) - 1.0
        disc = B * B - 4 * A * C
        t = (-B - np.sqrt(disc)) / (2 * A)
        pt = pos + t * d
        rec, lon, lat = sp.reclat(pt)
        return np.degrees(lon), np.degrees(lat), np.linalg.norm(pt)

    lines = list(range(0, NL, args.line_step))
    samples = list(range(0, NS, args.sample_step))
    rows = []
    t0 = time.time()
    for ln in lines:
        tick = s0 + ln * tick_per_line
        et = sp.sct2e(-85, tick)
        for s in samples:
            lo, la, r = ground(et, tick, s)
            rows.append(f"{lo:.7f} {la:.7f} {r:.3f} {ln} {s}")
        if (ln // args.line_step) % 25 == 0:
            el = (time.time() - t0) / 60
            print(f"  line {ln}/{NL}  ({el:.1f} min)")

    print(f"done: {len(rows)} points in {(time.time()-t0)/60:.1f} min")
    if args.out is None:
        args.out = str(DATA / "processed" / f"nac_geometry_{Path(args.nac).stem}.csv")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(["Longitude Latitude Radius Scan Pixel"] + rows) + "\n")
    print(f"wrote {out}")

    # corner summary
    a = np.array([r.split() for r in rows], dtype=float)
    print(f"corner extents: lon {a[:,0].min():.4f}..{a[:,0].max():.4f}  "
          f"lat {a[:,1].min():.4f}..{a[:,1].max():.4f}")


if __name__ == "__main__":
    main()