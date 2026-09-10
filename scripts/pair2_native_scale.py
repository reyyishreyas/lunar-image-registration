"""Pair-2 native-scale leg: re-project BOTH sensors to a common ~10 m/px grid.

Why this leg is new:
  * 60 m/px leg collapsed both sensors.
  * 15 m/px leg re-staged OHRC at ~native but reference the NAC at 60 m (no
    sub-60 m NAC detail).
  * Here OHRC (native uint8, ~1 m/px) is bilinearly sampled from the ISRO
    geometry CSV and NAC (native int16, ~4-5 m/px, via rasterio + SPICE geom
    lon/lat->scan/pixel nearest) onto the SAME ~10 m/px grid. The true
    transform is again identity.

Battery (same acceptance as before):
  * phase-congruency lock @20 m/px (true vs row-reversed null vs positive
    control on relit+shifted NAC)
  * sparse bright-PC-arc chamfer vs NAC PC ridges over a shift grid
  * 5 cross-modal fronts x SIFT+BF(0.75)+USAC_MAGSAC(5): inliers, RMSE vs
    identity, null_inliers (difference from envelope)
Writes results/logs/pair2_native_scale.md + .json
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from scipy.ndimage import map_coordinates                       # noqa: E402
from scipy.spatial import cKDTree                               # noqa: E402

from src.preprocessing.tmc import read_ch2_raw, read_ground_grid  # noqa: E402
from src.preprocessing.ch2_staging import GroundGridInverse      # noqa: E402
from src.detection.classical import detect_sift                  # noqa: E402
from src.detection.cross_modal import _apply_front               # noqa: E402
from src.matching.classical_match import match_bf_ratio          # noqa: E402
from src.outlier_rejection.ransac import find_homography_ransac  # noqa: E402
from scripts.pair2_phase_law import phase_congruency             # noqa: E402

OHRC_IMG = "data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img"
OHRC_GRD = "data/PATCH-004/OHRC/geometry/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv"
NAC_IMG = "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG"
NAC_GEOM = "data/processed/nac_geom_M1127_full.csv"
FRONTS = ("none", "clahe", "log_ratio", "gradient_structure")
GSD = 10.0


def as_u8(a, lo=1.0, hi=99.5):
    a = np.asarray(a, np.float32)
    lo, hi = np.nanpercentile(a, (lo, hi))
    return np.clip((a - lo) * (255.0 / max(1e-9, hi - lo)), 0, 255).astype(np.uint8)


def lock_report(name, A, B, shift_t="00"):
    if isinstance(A, list):      # A,B already 2-tuples of labels? keep simple below
        pass
    from scripts.pair2_phase_law import scan_corr as SC
    m = SC(A, B, box=(-30, 30, -10, 10))
    mN = SC(A, B[::-1, :], box=(-30, 30, -10, 10))
    best = max(m, key=m.get); bestN = max(mN, key=mN.get)
    c00 = m[(0, 0)]
    lev = (m[best] - np.median(list(m.values()))) / max(
        1e-9, (mN[bestN] - np.median(list(mN.values()))))
    frac_beat = sum(1 for k in m if m[k] > c00) / len(m)
    d = {"name": name, "corr_true": round(float(c00), 4),
         "best": (list(best), round(m[best], 4)),
         "median": round(float(np.median(list(m.values()))), 4),
         "null_best": (list(bestN), round(mN[bestN], 4)),
         "leverage": round(float(lev), 3), "frac_beat_true": round(float(frac_beat), 3)}
    print(f"{name:26s} c_true={d['corr_true']:.4f} best={d['best']} "
          f"med={d['median']:.4f} null={d['null_best']} lev={d['leverage']} beat={d['frac_beat_true']}")
    return d


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
    print(f"  [{front}] inl={n_in} ratio={ratio} rmse_id={rmse_id} frac<3={frac_near} "
          f"null={n_null} RESOLVED={bool(content_lock and beats_null and id_ok)}")
    return {"front": front, "matches": len(m), "inliers": n_in, "ratio": ratio,
            "rmse_identity": rmse_id, "frac_near3px": frac_near,
            "null_inliers": n_null, "resolved": bool(content_lock and beats_null and id_ok)}


def main() -> int:
    npz = np.load(os.path.join(PROJECT_ROOT,
                  "data/processed/spice_georef/pair2_fullstrip.npz"), allow_pickle=True)
    lon0, lon1 = float(npz["lon"].min()), float(npz["lon"].max())
    lat0, lat1 = float(npz["lat"].min()), float(npz["lat"].max())
    n_a = int(round((lat1 - lat0) * 111000.0 / GSD))
    n_c = int(round((lon1 - lon0) * 111000.0 * np.cos(np.deg2rad(0.5 * (lat0 + lat1))) / GSD))
    lats = np.linspace(lat0, lat1, n_a); lons = np.linspace(lon0, lon1, n_c)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")
    print(f"grid {n_a}x{n_c} @ {GSD} m/px")

    print("staging OHRC at 10 m/px...")
    spix, sscan, slon, slat = read_ground_grid(os.path.join(PROJECT_ROOT, OHRC_GRD))
    inv = GroundGridInverse(spix, sscan, slon, slat)
    scan, pix = inv.apply(LON.ravel(), LAT.ravel())
    ll = read_ch2_raw(os.path.join(PROJECT_ROOT, OHRC_IMG),
                      os.path.join(PROJECT_ROOT, OHRC_GRD), memmap=True)
    ok = np.isfinite(scan) & np.isfinite(pix)
    scan = np.where(ok, scan, 0); pix = np.where(ok, pix, 0)
    ohrc = map_coordinates(ll, np.vstack([scan, pix]), order=1,
                           mode="constant", cval=0).reshape(LAT.shape)
    ok = ok.reshape(LAT.shape)
    ohrc80 = np.where(ok, np.clip(ohrc, 0, 255), 0).astype(np.uint8)

    print("staging NAC at 10 m/px from raw .IMG + SPICE geom...")
    import pandas as pd
    g = pd.read_csv(os.path.join(PROJECT_ROOT, NAC_GEOM), sep=r"\s+").astype(float)
    g.lon = g.Longitude % 360.0
    tree = cKDTree(np.c_[g.lon, g.Latitude])
    pts = np.c_[LON.ravel(), LAT.ravel()]
    _, idx = tree.query(pts, k=1)
    s_ = g.Scan.values[idx]; p_ = g.Pixel.values[idx]
    import rasterio
    with rasterio.open(os.path.join(PROJECT_ROOT, NAC_IMG)) as ds:
        n_im = ds.read(1)                    # int16 native
        nrows, ncols = n_im.shape
    s_i = np.clip(np.rint(s_).astype(np.int64), 0, nrows - 1)
    p_i = np.clip(np.rint(p_).astype(np.int64), 0, ncols - 1)
    nac = n_im[s_i, p_i].astype(np.float32).reshape(LAT.shape)
    nac = np.where(nac <= -30000, np.nan, nac)
    nac8 = as_u8(nac)

    print("phase-congruency lock @ 20 m/px...")
    ds = 2
    s_pc = phase_congruency(ohrc80.astype(np.float64)[::ds, ::ds])
    r_pc = phase_congruency(nac8.astype(np.float64)[::ds, ::ds])
    rows = [lock_report("native10 PC", s_pc, r_pc)]

    r2 = np.clip(nac8.astype(np.float64) ** 1.6 + 40 * (nac8 > 128), 0, 255)
    pc_r2 = phase_congruency(r2[::ds, ::ds])
    rows.append(lock_report("positive-control (relit+2px) [10m grid]",
                            np.roll(pc_r2, 2, axis=0), r_pc))

    print("sparse bright-PC-arc chamfer...")
    thr_s = np.percentile(s_pc, 99); thr_r = np.percentile(r_pc, 97)
    mask_s = (s_pc > thr_s).astype(np.uint8)
    dmap = cv2.distanceTransform((r_pc > thr_r).astype(np.uint8), cv2.DIST_L2, 5)
    cham = {(dx, dy): float((dmap * np.roll(mask_s, (dy, dx), axis=(0, 1))).sum() /
                    max(1, np.roll(mask_s, (dy, dx), axis=(0, 1)).sum()))
            for dx, dy in ((-40, -8), (-20, -4), (0, 0), (20, 4), (40, 8))}
    rows.append({"name": "chamfer bright-PC arcs", "at_true": round(cham[(0, 0)], 3),
                 "best": (list(min(cham, key=cham.get)), round(cham[min(cham, key=cham.get)], 3)),
                 "all": {f"{k[0]},{k[1]}": round(v, 3) for k, v in cham.items()}})
    print("  chamfer:", rows[-1]["all"])

    print("5-front SIFT...")
    for front in FRONTS:
        s = _apply_front(ohrc80, front, ref=nac8)
        r = _apply_front(nac8, front, ref=ohrc80)
        rows.append(front_test(s, r, front))

    n_res = sum(1 for r in rows if isinstance(r, dict) and r.get("resolved"))
    status = "RESOLVED" if n_res else "NOT RESOLVED"
    out = {"status": status, "gsd_m": GSD, "grid": f"{n_a}x{n_c}", "rows": rows}
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_native_scale.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)
    with open(os.path.join(PROJECT_ROOT, "results/logs/pair2_native_scale.md"), "w") as fh:
        fh.write(f"# Pair-2 native-scale leg ({GSD} m/px, both sensors raw)\n\n"
                 f"Status: **{status}**\n\n")
        for r in rows:
            fh.write(f"- {json.dumps(r, default=str)}\n")
    print("\nSTATUS:", status)
    return 0


if __name__ == "__main__":
    sys.exit(main())