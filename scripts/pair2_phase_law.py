"""Pair-2 sun-angle-invariant correspondence attempt via PHASE CONGRUENCY.

Phase congruency (Kovesi 1999) marks points where Fourier components are in
phase — corners and edges independent of absolute brightness, contrast, and
illumination ramp. It is the canonical illumination-invariant edge detector,
i.e. exactly what "sun angle invariant correspondence" (PS-26166) requires.

Pipeline:
  1. PC maps for OHRC (raw) and NAC (normalised) on the co-located 60 m/px grid.
  2. Differential lock test: Pearson corr of PC response at true(0,0) vs a
     shift grid, vs row-reversed (envelope-killing) null; leverage = how much
     the true alignment beats its own median / the null.
  3. Sparse chamfer: bright-PC arcs of OHRC vs NAC PC ridge map over shifts.
  4. POSITIVE CONTROL: NAC relit (gamma + offset) + 3-px shift, to prove the
     PC pipeline itself locks when common structure exists.
Writes results/logs/pair2_phase_law.md + .json
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


def phase_congruency(img, n_scale=4, n_orient=6, min_wavelength=3.0,
                     sigma_frac=0.55, ang_sigma=0.45, eps=1e-9):
    """2D phase congruency (Kovesi-style log-Gabor) -> 8-bit response map.

    Illumination-invariant: responses are normalised by local amplitude, so
    absolute brightness / contrast / illumination ramps cancel.
    """
    im = np.asarray(img, np.float64)
    im = (im - im.mean()) / (im.std() + eps)
    r, c = im.shape
    R, C = 2 ** int(np.ceil(np.log2(r))), 2 ** int(np.ceil(np.log2(c)))
    pad = np.zeros((R, C))
    pad[:r, :c] = im
    rf = np.fft.fft2(pad)
    fy, fx = np.fft.fftfreq(R)[:, None] * R, np.fft.fftfreq(C)[None, :] * C
    radius = np.sqrt(fx ** 2 + fy ** 2)
    radius[0, 0] = 1.0
    theta = np.arctan2(fy, fx)
    pcsum = np.zeros((R, C))
    for alpha in np.linspace(0, np.pi, n_orient, endpoint=False):
        dtheta = np.arctan2(np.sin(theta - alpha), np.cos(theta - alpha))
        ang = np.exp(-(dtheta ** 2) / (2 * ang_sigma ** 2))
        A_sum = np.zeros((R, C)); E_sum = np.zeros((R, C))
        for s in range(n_scale):
            wl = min_wavelength * 2.0 ** s
            radial = np.exp(-(np.log(radius / wl) ** 2) / (2 * sigma_frac ** 2))
            radial[0, 0] = 0.0
            resp = np.fft.ifft2(rf * radial * ang).real
            A_sum += np.abs(resp)
            E_sum += resp
        E = np.sqrt(E_sum ** 2 + eps)
        pcsum += E / (A_sum + eps)
    pc = pcsum / n_orient
    pc = pc[:r, :c]
    return np.clip(pc / (np.percentile(pc, 97.5) + eps) * 255, 0, 255).astype(np.uint8)


def as_u8(a, lo=1.0, hi=99.5):
    f = np.asarray(a, np.float32)
    lo, hi = np.nanpercentile(f, (lo, hi))
    return np.clip((f - lo) * (255.0 / max(1e-9, hi - lo)), 0, 255).astype(np.uint8)


def scan_corr(A, B, box=(-30, 30, -10, 10)):
    A = A.astype(np.float32); B = B.astype(np.float32)
    out = {}
    for dy in range(box[2], box[3] + 1):
        for dx in range(box[0], box[1] + 1):
            sy0, sy1 = max(-dy, 0), min(A.shape[0], A.shape[0] - dy)
            sx0, sx1 = max(-dx, 0), min(A.shape[1], A.shape[1] - dx)
            x = A[sy0:sy1, sx0:sx1].ravel(); y = B[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx].ravel()
            if x.size < 2000:
                continue
            x = x - x.mean(); y = y - y.mean()
            den = float(np.sqrt((x * x).sum() * (y * y).sum()))
            out[(dx, dy)] = float((x * y).sum()) / den if den > 0 else 0.0
    return out


def lock_report(name, A, B, shift_t=(0, 0)):
    m = scan_corr(A, B)
    mN = scan_corr(A, B[::-1, :])
    best = max(m, key=m.get)
    bestN = max(mN, key=mN.get)
    c00 = m.get(shift_t, float("nan"))
    lev = (m[best] - np.median(list(m.values()))) / max(
        1e-9, (mN[bestN] - np.median(list(mN.values()))))
    # does the TRUE shift win relative to random?
    rank = sum(1 for k in m if m[k] > c00) / len(m)
    row = {"name": name, "corr_at_true": round(float(c00), 4),
           "best": (list(best), round(m[best], 4)),
           "median": round(float(np.median(list(m.values()))), 4),
           "null_best": (list(bestN), round(mN[bestN], 4)),
           "leverage": round(float(lev), 3),
           "frac_shifts_beating_true": round(rank, 3)}
    print(f"{name:22s} c_true={row['corr_at_true']:.4f} best={row['best']} "
          f"med={row['median']:.4f} null_best={row['null_best']} "
          f"leverage={row['leverage']} frac_beat={row['frac_shifts_beating_true']}")
    return row


def main() -> int:
    d = np.load(os.path.join(PROJECT_ROOT,
                             "data/processed/spice_georef/pair2_fullstrip.npz"),
                allow_pickle=True)
    src, ref = d["src"], d["ref"]          # OHRC-2026 (uint8), NAC (float32)
    s8, r8 = src.astype(np.float64), as_u8(ref).astype(np.float64)

    print("computing PC maps (2 images, 6 ori x 4 scales of 1536x194)...")
    pc_s = phase_congruency(s8)
    pc_r = phase_congruency(r8)

    rows = []
    print("\n--- pair-2 true vs row-reversed null ---")
    rows.append(lock_report("pair2 PC", pc_s, pc_r))
    # sparse chamfer: OHRC top-1% PC arcs vs NAC PC ridge distance
    thr = np.percentile(pc_s, 99)
    mask_s = (pc_s > thr).astype(np.uint8)
    dmap = cv2.distanceTransform((pc_r > np.percentile(pc_r, 95)).astype(np.uint8), cv2.DIST_L2, 5)
    cham = {}
    for dy in (-6, -3, 0, 3, 6):
        for dx in (-30, -15, 0, 15, 30):
            M = np.roll(mask_s, (dy, dx), axis=(0, 1))
            cham[(dx, dy)] = float((dmap * M).sum() / max(1, M.sum()))
    rows.append({"name": "pair2 chamfer bright-PC arcs",
                 "best": (list(min(cham, key=cham.get)), round(cham[min(cham, key=cham.get)], 2)),
                 "at_true": round(cham[(0, 0)], 2),
                 "all": {f"{k[0]},{k[1]}": round(v, 2) for k, v in cham.items()}})
    print("chamfer arcs:", rows[-1]["best"], "| at_true:", rows[-1]["at_true"])

    # positive control: relight NAC (gamma + offset) and translate +2px in PC domain
    r2 = np.clip(r8 ** 1.6 + 40 * (r8 > 128), 0, 255)
    pc_r2 = phase_congruency(r2)
    rows.append(lock_report("positive-control PC (relit+2px)", np.roll(pc_r2, 2, axis=0), pc_r,
                            shift_t=(0, 2)))

    out = {"status": "control", "rows": rows}
    os.makedirs(os.path.join(PROJECT_ROOT, "results/logs"), exist_ok=True)
    cv2.imwrite(os.path.join(PROJECT_ROOT, "results/logs/pair2_phase_src.png"), pc_s)
    cv2.imwrite(os.path.join(PROJECT_ROOT, "results/logs/pair2_phase_ref.png"), pc_r)
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_phase_law.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_phase_law.md"), "w") as fh:
        fh.write("# Pair-2 phase-congruency (sun-angle-invariant) attempt\n\n")
        fh.write("Phase congruency marks edges in a brightness/contrast-invariant "
                 "way (Kovesi log-Gabor). Table: correlation of PC response at the "
                 "true alignment versus other shifts and versus a row-reversed "
                 "null.\n\n")
        fh.write("| test | corr@true | best shift | median | null best | leverage | "
                 "frac shifts > true |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            if "leverage" in r:
                fh.write(f"| {r['name']} | {r['corr_at_true']} | {r['best']} | "
                         f"{r['median']} | {r['null_best']} | {r['leverage']} | "
                         f"{r['frac_shifts_beating_true']} |\n")
            else:
                fh.write(f"| {r['name']} | sh {r['at_true']} (best {r['best']}) |\n")
        fh.write("\nPositive control (NAC relit with different gamma + 2-px shift) "
                 "must lock sharply for the pipeline itself to be trusted.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())