"""Pair-2 RESOLUTION test at finer scale.

The Phase-5 staged orthos collapse both sensors to 60 m/px, 6x coarser than
the ~10 m/px native OHRC pitch. A near-black OHRC strip (raw max 111) may
only retain faint terrain signal at/near its native sampling. This script
re-stages OHRC at ~15 m/px (4x finer) from the ISRO geometry CSV
(map_coordinates, bilinear), upsamples the co-located NAC ortho to the same
grid, and re-runs the full resolution battery:

  * 5 cross-modal fronts x SIFT + BF(0.75) + USAC_MAGSAC(5)
    - inliers, ratio, RMSE vs IDENTITY (true transform on the same grid),
      fraction of inliers within 3 px of identity
    - row-reversed null control (inliers > 3x null => not an envelope artifact)
  * geometry-dominant correlation scan (gradient & LoG-ridge) over a local
    shift grid, with row-reversed-null leverage
  * loose SIFT (ratio 0.9) displacement clustering near identity

Message written to:  results/logs/pair2_scale15_resolve.md  (+ .json)
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scipy.ndimage import map_coordinates                      # noqa: E402

from src.preprocessing.ch2_staging import GroundGridInverse, read_ground_grid  # noqa: E402
from src.preprocessing.tmc import read_ch2_raw                 # noqa: E402
from src.detection.classical import detect_sift                # noqa: E402
from src.detection.cross_modal import _apply_front             # noqa: E402
from src.matching.classical_match import match_bf_ratio        # noqa: E402
from src.outlier_rejection.ransac import find_homography_ransac  # noqa: E402

OHRC_IMG = "data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img"
OHRC_GRD = "data/PATCH-004/OHRC/geometry/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv"
NPZ = "data/processed/spice_georef/pair2_fullstrip.npz"
FRONTS = ("none", "clahe", "histogram_match", "log_ratio", "gradient_structure")


def as_u8(a, lo=1.0, hi=99.5):
    f = np.asarray(a, np.float32)
    lo, hi = np.nanpercentile(f, (lo, hi))
    return np.clip((f - lo) * (255.0 / max(1e-9, hi - lo)), 0, 255).astype(np.uint8)


def sig_scan(a, b, box=(-20, 20, -8, 8), ds=2, use_mag=True):
    """Correlation-scan lock + row-reversed-null leverage."""
    if use_mag:
        def feat(x):
            gx = cv2.Sobel(x, cv2.CV_32F, 1, 0, 3)
            gy = cv2.Sobel(x, cv2.CV_32F, 0, 1, 3)
            m = np.sqrt(gx * gx + gy * gy)
            return m
    else:
        def feat(x):  # LoG ridge map
            g = cv2.GaussianBlur(x.astype(np.float32), (5, 5), 0)
            lg = cv2.Laplacian(g, cv2.CV_32F)
            return np.where(lg > 0, np.abs(lg), 0)
    A = feat(a)[::ds, ::ds].astype(np.float32)
    B = feat(b)[::ds, ::ds].astype(np.float32)
    BN = B[::-1, :]

    def scan(C):
        out = {}
        for dy in range(box[2], box[3] + 1):
            for dx in range(box[0], box[1] + 1):
                sy0, sy1 = max(-dy, 0), min(A.shape[0], A.shape[0] - dy)
                sx0, sx1 = max(-dx, 0), min(A.shape[1], A.shape[1] - dx)
                x = A[sy0:sy1, sx0:sx1].ravel()
                y = C[sy0 + dy:sy1 + dy, sx0 + dx:sx1 + dx].ravel()
                if x.size < 2000:
                    continue
                x = x - x.mean(); y = y - y.mean()
                den = float(np.sqrt((x * x).sum() * (y * y).sum()))
                out[(dx * ds, dy * ds)] = float((x * y).sum()) / den if den > 0 else 0.0
        return out

    m, mN = scan(B), scan(BN)
    best = max(m, key=m.get); bestN = max(mN, key=mN.get)
    lev = (m[best] - np.median(list(m.values()))) / max(1e-9, mN[bestN] - np.median(list(mN.values())))
    return {"true_corr00": m.get((0, 0)), "true_best": (list(best), m[best]),
            "true_med": float(np.median(list(m.values()))),
            "null_best": (list(bestN), mN[bestN]), "leverage": float(lev)}


def front_test(s, r, front):
    rN = r[::-1, :].copy()
    kp1, d1 = detect_sift(s, nfeatures=20000, contrast_threshold=0.03)
    kp2, d2 = detect_sift(r, nfeatures=20000, contrast_threshold=0.03)
    m = match_bf_ratio(d1, d2, ratio=0.75, verbose=False)
    ps1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
    ps2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
    H, inl = (find_homography_ransac(ps1, ps2, ransac_thresh=5.0, method="usac_magsac")
              if len(m) >= 8 else (None, None))
    n_in = int(inl.sum()) if inl is not None else 0
    ratio = round(float(n_in / max(len(m), 1)), 4)
    rmse_id = frac_near = None
    if H is not None and n_in >= 4:
        p1i, p2i = ps1[inl > 0], ps2[inl > 0]
        disp = np.sqrt(np.sum((p2i - p1i) ** 2, axis=1))
        frac_near = float((disp <= 3.0).mean())
        rmse_id = float(np.sqrt(np.mean(disp ** 2)))

    kpN, dN = detect_sift(rN, nfeatures=20000, contrast_threshold=0.03)
    mN = match_bf_ratio(d1, dN, ratio=0.75, verbose=False)
    ps1N = np.float32([kp1[x.queryIdx].pt for x in mN]).reshape(-1, 2) if mN else np.empty((0, 2))
    ps2N = np.float32([kpN[x.trainIdx].pt for x in mN]).reshape(-1, 2) if mN else np.empty((0, 2))
    _, inlN = (find_homography_ransac(ps1N, ps2N, ransac_thresh=5.0, method="usac_magsac")
               if len(mN) >= 8 else (None, np.zeros(0, bool)))
    n_null = int(inlN.sum()) if inlN is not None and len(inlN) else 0

    content_lock = n_in >= 20 and ratio >= 0.25
    beats_null = n_null > 0 and n_in > 3 * n_null
    id_ok = frac_near is not None and frac_near >= 0.5 and rmse_id is not None and rmse_id <= 3.0

    # loose-SIFT identity clustering
    ml = match_bf_ratio(d1, d2, ratio=0.9, verbose=False)
    frac_l = med_d = None
    if ml:
        disp = np.array([np.asarray(kp2[x.trainIdx].pt) - np.asarray(kp1[x.queryIdx].pt) for x in ml])
        frac_l = float((np.abs(disp).max(axis=1) <= 3).mean())
        med_d = float(np.median(np.linalg.norm(disp, axis=1)))

    return {
        "front": front, "n_matches": len(m), "inliers": n_in, "ratio": ratio,
        "rmse_identity": rmse_id, "frac_near3px": frac_near,
        "null_inliers": n_null, "beats_null_x3": bool(beats_null),
        "content_lock": bool(content_lock), "beats_null": bool(beats_null), "identity_ok": bool(id_ok),
        "loose": {"n": len(ml), "frac_disp<3px": frac_l, "med_disp_px": med_d},
        "resolved": bool(content_lock and beats_null and id_ok),
    }


def main() -> int:
    d = np.load(os.path.join(PROJECT_ROOT, NPZ), allow_pickle=True)
    lon60, lat60 = d["lon"], d["lat"]
    n_a60, n_c60 = lon60.shape
    lat0, lat1 = float(lat60.min()), float(lat60.max())
    lon0, lon1 = float(lon60.min()), float(lon60.max())
    n_along, n_across = n_a60 * 4, n_c60 * 4
    lats = np.linspace(lat0, lat1, n_along)
    lons = np.linspace(lon0, lon1, n_across)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    gsd = float((lat1 - lat0) * 111000.0 / (n_along - 1))

    spix, sscan, slon, slat = read_ground_grid(os.path.join(PROJECT_ROOT, OHRC_GRD))
    inv = GroundGridInverse(spix, sscan, slon, slat)
    scan, pix = inv.apply(LON.ravel(), LAT.ravel())
    ll = read_ch2_raw(os.path.join(PROJECT_ROOT, OHRC_IMG),
                      os.path.join(PROJECT_ROOT, OHRC_GRD), memmap=True)
    ok = np.isfinite(scan) & np.isfinite(pix)
    scan = np.where(ok, scan, 0); pix = np.where(ok, pix, 0)
    ohrc15 = map_coordinates(ll, np.vstack([scan, pix]), order=1,
                             mode="constant", cval=0).reshape(LAT.shape)
    ohrc15 = np.clip(ohrc15, 0, 255).astype(np.uint8)
    nac15 = cv2.resize(d["ref"], (n_across, n_along), interpolation=cv2.INTER_LINEAR)
    nac15 = as_u8(nac15)
    s15, r15 = as_u8(ohrc15), nac15

    rows = []
    for front in FRONTS:
        s = _apply_front(s15, front, ref=r15) if front != "histogram_match" \
            else _apply_front(s15, front, ref=r15)
        r = _apply_front(r15, front, ref=s15) if front != "histogram_match" \
            else _apply_front(r15, front, ref=s15)
        row = front_test(s, r, front)
        row["scan_grad"] = sig_scan(s15, r15, use_mag=True)
        row["scan_ridge"] = sig_scan(s15, r15, use_mag=False)
        rows.append(row)
        print(f"[{front}] inliers={row['inliers']} ratio={row['ratio']} "
              f"rmse_id={row['rmse_identity']} frac<3={row['frac_near3px']} "
              f"null={row['null_inliers']} loose<3px={row['loose']['frac_disp<3px']} "
              f"RESOLVED={row['resolved']}")

    n_res = sum(1 for r in rows if r["resolved"])
    status = "RESOLVED" if n_res else "NOT RESOLVED"
    out = {
        "status": status, "n_fronts_resolved": n_res,
        "gsd_m": round(gsd, 2), "grid": f"{n_along}x{n_across}",
        "note": ("OHRC re-staged at 15 m/px from ISRO geometry; NAC ortho "
                 "upsampled to same grid. True transform = identity."),
        "fronts": rows,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "results/logs"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_scale15_resolve.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_scale15_resolve.md"), "w") as fh:
        fh.write(f"# Pair-2 resolution at {gsd:.1f} m/px\n\nStatus: **{status}** "
                 f"({n_res}/{len(rows)} fronts).\n\n")
        fh.write("| front | matches | inl | ratio | rmse_id | frac<3px | null | "
                 "loose<3px | scan_grad.lock | scan_ridge.lock | RESOLVED |\n|---|---|---|---|---|---|---|---|---|---|---|\n")
        for row in rows:
            fh.write(f"| {row['front']} | {row['n_matches']} | {row['inliers']} | "
                     f"{row['ratio']} | {_f(row['rmse_identity'])} | "
                     f"{_f(row['frac_near3px'])} | {row['null_inliers']} | "
                     f"{_f(row['loose']['frac_disp<3px'])} | "
                     f"{_f(row['scan_grad']['leverage'])} | "
                     f"{_f(row['scan_ridge']['leverage'])} | "
                     f"{'YES' if row['resolved'] else 'no'} |\n")
    print("\nSTATUS:", status, "| gsd=%.2f m/px | grid %dx%d" % (gsd, n_along, n_across))
    return 0


def _f(x):
    return "n/a" if x is None else f"{x:.3f}"


if __name__ == "__main__":
    sys.exit(main())