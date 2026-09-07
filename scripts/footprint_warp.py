"""Footprint-initialized warping: resample an LROC NAC onto the OHRC ground grid.

Uses only PDS/geometry metadata (no learned or classical feature matching):
  * OHRC side: its geometry CSV (pixel, scan) -> (lon, lat) grid.
  * NAC side:  the ODE Footprint_geometry 4-corner polygon + a quasi-linear
    push-broom model (row ~ lat, col ~ lon), fitted by least squares.

Output crops are on the SAME (scan, pixel) ground grid -> same GSD, so later
stages (SIFT/match) only have to fine-adjust residual errors.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

OHRC_FULL = (93700, 12000)  # (scan, pixel) full resolution
NAC_FULL = (52224, 5064)    # (line, sample)
UTM_DEG_KM = 111.32

# lat lon per corner vertex of nac.m1127547939rc (from ODE Footprint_geometry)
NAC_CORNERS = ((296.2, 8.27), (296.27, 6.37), (296.1, 6.37), (296.02, 8.26))


def load_ohrc_geometry(csv_path: Path):
    scans = {}   # scan index -> list of (pixel, lon, lat)
    pixels = {}
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            px, sc = int(r["Pixel"]), int(r["Scan"])
            lon, lat = float(r["Longitude"]), float(r["Latitude"])
            scans.setdefault(sc, []).append((px, lon, lat))
            pixels.setdefault(px, []).append((sc, lon, lat))
    scan_ids = sorted(scans)
    px_ids = sorted(pixels)
    assert all(len(pixels[p]) == len(scan_ids) for p in px_ids), "ragged pixel column"
    assert all(len(scans[s]) == len(px_ids) for s in scan_ids), "ragged scan row"
    lon = np.empty((len(scan_ids), len(px_ids)), np.float64)
    lat = np.empty_like(lon)
    for i, sc in enumerate(scan_ids):
        row = sorted(scans[sc])  # by pixel
        lon[i, :] = [x[1] for x in row]
        lat[i, :] = [x[2] for x in row]
    return np.asarray(scan_ids, np.float64), np.asarray(px_ids, np.float64), lon, lat


def nac_affine_local(corners, shape, u0=0.5, v0=0.5):
    """Linearized inverse of the bilinear NAC push-broom model at (u0,v0).

    corners: ring of 4 (lon, lat); assigned in ring order to
             (u=0,v=0),(u=0,v=1),(u=1,v=1),(u=1,v=0)  [top-left, top-right,
             bottom-right, bottom-left in image space] when north_top=True.
    Returns map2 (2x3) such that [line;col] = map2 @ [1;lon;lat].
    """
    rows, cols = shape
    lon = np.array([c[0] for c in corners], float)
    lat = np.array([c[1] for c in corners], float)
    c00, c01, c11, c10 = lon  # top-left, top-right, bottom-right, bottom-left
    l00, l01, l11, l10 = lat

    def f(u, v):
        return (((1 - u) * (1 - v) * c00 + (1 - u) * v * c01 +
                 u * (1 - v) * c10 + u * v * c11),
                ((1 - u) * (1 - v) * l00 + (1 - u) * v * l01 +
                 u * (1 - v) * l10 + u * v * l11))

    e = 1e-7
    (lon0, lat0), (lon_u, lat_u), (lon_v, lat_v) = f(u0, v0), f(u0 + e, v0), f(u0, v0 + e)
    Jlon = np.array([[lon_u - lon0, lon_v - lon0],
                     [lat_u - lat0, lat_v - lat0]]) / e  # columns du, dv
    Jinv = np.linalg.inv(Jlon)
    # local affine: d(u,v) = Jinv @ (d lon, d lat)
    a_u = Jinv[0, 0]; b_u = Jinv[0, 1]; a_v = Jinv[1, 0]; b_v = Jinv[1, 1]
    map2 = np.zeros((2, 3))
    map2[0] = [rows * (u0 - a_u * lon0 - b_u * lat0), a_u * rows, b_u * rows]
    map2[1] = [cols * (v0 - a_v * lon0 - b_v * lat0), a_v * cols, b_v * cols]
    return map2


def as_u8(img16: np.ndarray) -> np.ndarray:
    if img16.dtype == np.uint8:
        return img16
    lo, hi = np.percentile(img16, (1, 99.5))
    out = np.clip((img16.astype(np.float64) - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255)
    return out.astype(np.uint8)


def warp_ohrc_crop(nac_img, map2, ohrc_lon, ohrc_lat, scan_ws, px_ws, size):
    """Resample NAC onto an OHRC ws crop (scan, pixel) grid of side `size`."""
    r = np.arange(size)
    scan_idx = (scan_ws - size // 2 + r) * 10      # ws step 10 -> full-res scan
    px_idx = (px_ws - size // 2 + r[:, None]) * 10  # ws step 10 -> full-res pixel
    lon = ohrc_lon((scan_idx, px_idx))              # shape (size, size)
    lat = ohrc_lat((scan_idx, px_idx))
    line = map2[0, 0] + map2[0, 1] * lon + map2[0, 2] * lat
    col = map2[1, 0] + map2[1, 1] * lon + map2[1, 2] * lat
    # slice the (possibly >32k-line) NAC to the needed range
    fin = np.isfinite(line) & np.isfinite(col)
    if not fin.any():
        raise ValueError("no NAC coverage in crop")
    lo_l, hi_l = int(np.floor(line[fin].min())), int(np.ceil(line[fin].max()))
    lo_c, hi_c = int(np.floor(col[fin].min())), int(np.ceil(col[fin].max()))
    assert hi_l - lo_l + 1 < 32767 and hi_c - lo_c + 1 < 32767, "tile too large"
    sub = nac_img[max(lo_l, 0): min(hi_l + 1, nac_img.shape[0]),
                  max(lo_c, 0): min(hi_c + 1, nac_img.shape[1])]
    mapx = (col - max(lo_c, 0)).astype(np.float32)
    mapy = (line - max(lo_l, 0)).astype(np.float32)
    warped = cv2.remap(sub, mapx, mapy, cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped, (lon[0, 0], lat[0, 0])


def sift_match(src, ref, ratio=0.8, thresh=5.0):
    sf = cv2.SIFT_create(nfeatures=4000, contrastThreshold=thresh)
    k1, d1 = sf.detectAndCompute(src, None)
    k2, d2 = sf.detectAndCompute(ref, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, 0.0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m0 = bf.knnMatch(d1, d2, k=2)
    good = [a[0] for a, b in m0
            if len(a) == 1 and len(b) == 1 and a[0].distance < ratio * b[0].distance]
    if len(good) < 8:
        return 0, 0.0, 0.0
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    if H is None:
        return 0, 0.0, 0.0
    nin = int(mask.sum())
    dsize = (ref.shape[1], ref.shape[0])
    ncc = 0.0
    if nin >= 12:
        warped = cv2.warpPerspective(src, H, dsize,
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        a = warped.astype(np.float32).ravel()
        b = ref.astype(np.float32).ravel()
        if nin < 40:  # heavy NCC over the whole crop is arguably fine; keep full
            pass
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt(np.mean(a * a) * np.mean(b * b))
        ncc = float(np.mean(a * b) / denom) if denom > 1e-6 else 0.0
    return nin, float(nin / len(good)), ncc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", action="store_true", help="validate a 1024x1024 crop")
    ap.add_argument("--scan", type=int, default=4600, help="ws scan-center of crop")
    ap.add_argument("--px", type=int, default=600, help="ws pixel-center of crop")
    ap.add_argument("--south-top", action="store_true", help="opposite NAC line orientation")
    ap.add_argument("--out", default="data/processed/fp_validate")
    args = ap.parse_args()

    t0 = time.time()
    from src.preprocessing.georeference import read_ohrc_raw, read_nac_img
    ohrc_img = read_ohrc_raw("data/PATCH-004/OHRC/data/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_d_img_d18.img")
    ohrc_ws = ohrc_img[::10, ::10]
    nac_img = read_nac_img("data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG")
    print(f"loaded OHRC {ohrc_img.shape} NAC {nac_img.shape} [{time.time()-t0:.0f}s]", flush=True)

    scan_ids, px_ids, lon_g, lat_g = load_ohrc_geometry(
        Path("data/PATCH-004/OHRC/geometry/calibrated/20260331/ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv"))
    from scipy.interpolate import RegularGridInterpolator
    ohrc_lon = RegularGridInterpolator((scan_ids, px_ids), lon_g)
    ohrc_lat = RegularGridInterpolator((scan_ids, px_ids), lat_g)

    map2 = nac_affine_local(NAC_CORNERS, NAC_FULL, u0=0.275, v0=0.72)
    print(f"NAC local affine:\n{map2.round(2)}", flush=True)

    if args.crop:
        size = 1024
        src = as_u8(ohrc_ws[args.scan - size // 2: args.scan + size // 2,
                            args.px - size // 2: args.px + size // 2])
        ref, _ = warp_ohrc_crop(nac_img, map2, ohrc_lon, ohrc_lat,
                                args.scan, args.px, size)
        ref = as_u8(ref)
        Path(args.out).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(f"{args.out}/src.png", src)
        cv2.imwrite(f"{args.out}/ref.png", ref)
        nin, ioc, ncc = sift_match(src, ref)
        print(f"crop src={src.shape} ref={ref.shape}")
        print(f"SIFT inliers={nin} inlier_over_ratio={ioc:.3f} NCC={ncc:+.3f}")
        print(f"images at {f'{args.out}/src.png'} / 'ref.png' [{time.time()-t0:.0f}s]")
    else:
        raise SystemExit("full-strip warp not implemented yet; use --crop")


if __name__ == "__main__":
    main()