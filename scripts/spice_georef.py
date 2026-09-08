"""SPICE-geometry initialized common-ground warp for pair 2 (OHRC-2026 x NAC).

Replaces the ODE 4-corner affine model in footprint_warp.py with the per-line
SPICE pointing (data/processed/nac_geom_M1127_full.csv). NAC (line, sample) ->
(lon, lat) is sampled per line; the inverse map (lon, lat) -> (line, sample) is
built with a per-quad Newton solve over the rectangular SPICE grid so the NAC
can be resampled onto the OHRC ground grid (same ground pixel grid = equal GSD
staging).

Crop-first workflow:
  * src = OHRC crop on its (scan, pixel) full-res grid.
  * ref = NAC resampled at the same ground positions (via lon/lat).
  * both written as u8 PNG + fine-match stats to validate alignment.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

OHRC = ("data/PATCH-004/OHRC/data/calibrated/20260331/"
        "ch2_ohr_ncp_20260331T1105235288_d_img_d18.img")
OHRC_GEO = ("data/PATCH-004/OHRC/geometry/calibrated/20260331/"
            "ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv")
NAC = "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG"
NAC_GEO = "data/processed/nac_geom_M1127_full.csv"

DEG_KM = 111.32


def load_ohrc_geom(csv_path):
    scan_ids = {}
    px_ids = {}
    with open(csv_path) as fh:
        for r in csv.DictReader(fh):
            px, sc = int(r["Pixel"]), int(r["Scan"])
            lon, lat = float(r["Longitude"]), float(r["Latitude"])
            scan_ids.setdefault(sc, []).append((px, lon, lat))
            px_ids.setdefault(px, []).append((sc, lon, lat))
    scan_axis = sorted(scan_ids)
    px_axis = sorted(px_ids)
    lon = np.empty((len(scan_axis), len(px_axis)))
    lat = np.empty_like(lon)
    for i, sc in enumerate(scan_axis):
        row = sorted(scan_ids[sc])
        lon[i, :] = [x[1] for x in row]
        lat[i, :] = [x[2] for x in row]
    return (np.asarray(scan_axis, float), np.asarray(px_axis, float), lon, lat)


def load_nac_geom(csv_path):
    rows = []
    with open(csv_path) as fh:
        for line in fh:
            parts = line.split()
            if len(parts) != 5 or not parts[0].replace("-", "").replace(".", "").isdigit():
                continue
            rows.append((float(parts[0]), float(parts[1]),
                         float(parts[3]), float(parts[4])))
    lon = np.array([r[0] for r in rows])
    lat = np.array([r[1] for r in rows])
    ln = np.array([r[2] for r in rows])
    sm = np.array([r[3] for r in rows])
    return lon, lat, ln, sm


class NacInverse:
    """(lon [E 0-360], lat) -> (line, sample) via per-quad Newton over the
    rectangular SPICE grid. Robust to thin-anisotropic strips where a scattered
    Delaunay inverse would fold over."""

    def __init__(self, lon, lat, line, sample):
        uniq_l = np.unique(line)
        uniq_s = np.unique(sample)
        assert len(uniq_l) * len(uniq_s) == len(line), "non-rectangular grid"
        self.line_ax = uniq_l
        self.samp_ax = uniq_s
        self.lon_g = (lon % 360.0).reshape(len(uniq_l), len(uniq_s))
        self.lat_g = lat.reshape(len(uniq_l), len(uniq_s))
        self.lon_f = RegularGridInterpolator(
            (self.line_ax, self.samp_ax), self.lon_g,
            bounds_error=False, fill_value=np.nan)
        self.lat_f = RegularGridInterpolator(
            (self.line_ax, self.samp_ax), self.lat_g,
            bounds_error=False, fill_value=np.nan)
        self._lonref = float(np.nanmedian(self.lon_g))

    @staticmethod
    def _unwrap(a, ref):
        return (a - ref + 180.0) % 360.0 - 180.0 + ref

    def _f_batch(self, line, samp):
        q = np.column_stack([line.ravel(), samp.ravel()])
        lo = np.asarray(self.lon_f(q), np.float64).ravel().reshape(line.shape)
        la = np.asarray(self.lat_f(q), np.float64).ravel().reshape(line.shape)
        lo = self._unwrap(lo, self._lonref)
        return lo, la

    def apply(self, lon, lat):
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        L = np.interp(-lat, -self.lat_g[:, 0], self.line_ax)
        mid = self.lat_g.shape[0] // 2
        S = np.interp(lon, self.lon_g[mid, :][::-1], self.samp_ax[::-1])
        lo_v, la_v = self._f_batch(L, S)
        for _ in range(10):
            e = 1e-3
            loE, laE = self._f_batch(L + e, S)
            loT, laT = self._f_batch(L, S + e)
            J00 = (loE - lo_v) / e
            J01 = (loT - lo_v) / e
            J10 = (laE - la_v) / e
            J11 = (laT - la_v) / e
            det = J00 * J11 - J01 * J10
            dlon = lon - lo_v
            dlat = lat - la_v
            L = L + (J11 * dlon - J01 * dlat) / det
            S = S + (-J10 * dlon + J00 * dlat) / det
            lo_v, la_v = self._f_batch(L, S)
        lo_v, la_v = self._f_batch(L, S)
        L = np.where(np.isfinite(lo_v) & np.isfinite(la_v), L, np.nan)
        S = np.where(np.isfinite(lo_v) & np.isfinite(la_v), S, np.nan)
        return S, L


def as_u8(img) -> np.ndarray:
    arr = np.asarray(img, np.float32)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    if np.isnan(arr).all():
        return np.zeros(arr.shape, np.uint8)
    lo, hi = np.nanpercentile(arr, (1, 99.5))
    out = np.clip((arr - lo) * (255.0 / max(hi - lo, 1e-9)), 0, 255)
    return np.nan_to_num(out).astype(np.uint8)


def remap_nac(nac_img, line, col):
    """Remap the (possibly >32k-line) NAC onto the given (line, col) grid."""
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
    return cv2.remap(sub, mapx, mapy, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def sift_match(src, ref, ratio=0.8, thresh=0.02):
    sf = cv2.SIFT_create(nfeatures=6000, contrastThreshold=thresh)
    k1, d1 = sf.detectAndCompute(src, None)
    k2, d2 = sf.detectAndCompute(ref, None)
    if d1 is None or d2 is None or len(k1) < 8 or len(k2) < 8:
        return 0, 0.0, 0.0
    bf = cv2.BFMatcher(cv2.NORM_L2)
    m0 = bf.knnMatch(d1, d2, k=2)
    good = []
    for mb in m0:
        if hasattr(mb, "distance"):          # cv2.5 may return single DMatch
            good.append(mb)
            continue
        if len(mb) >= 2 and mb[0].distance < ratio * mb[1].distance:
            good.append(mb[0])
        elif len(mb) == 1:
            good.append(mb[0])
    if len(good) < 8:
        return 0, 0.0, 0.0
    src_pts = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.USAC_MAGSAC, 5.0)
    if H is None:
        return 0, 0.0, 0.0
    nin = int(mask.sum())
    ncc = 0.0
    if nin >= 12:
        warped = cv2.warpPerspective(src, H, (ref.shape[1], ref.shape[0]),
                                     flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP)
        a = warped.astype(np.float32).ravel()
        b = ref.astype(np.float32).ravel()
        a, b = a - a.mean(), b - b.mean()
        den = np.sqrt(np.mean(a * a) * np.mean(b * b))
        ncc = float(np.mean(a * b) / den) if den > 1e-6 else 0.0
    return nin, float(nin / len(good)), ncc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=int, default=46000, help="full-res OHRC scan center")
    ap.add_argument("--px", type=int, default=6000, help="full-res OHRC pixel center")
    ap.add_argument("--size", type=int, default=1024, help="crop side (OHRC full-res px)")
    ap.add_argument("--stride", type=int, default=4, help="OHRC sampling stride (~GSD)")
    ap.add_argument("--out", default="data/processed/spice_georef")
    args = ap.parse_args()

    t0 = time.time()
    from src.preprocessing.georeference import read_ohrc_raw, read_nac_img
    ohrc = read_ohrc_raw(OHRC)
    nac = read_nac_img(NAC)
    print(f"loaded OHRC {ohrc.shape} NAC {nac.shape} [{time.time()-t0:.0f}s]", flush=True)

    scan_a, px_a, ohrc_lon_g, ohrc_lat_g = load_ohrc_geom(OHRC_GEO)
    ohrc_lon = RegularGridInterpolator((scan_a, px_a), ohrc_lon_g,
                                       bounds_error=False, fill_value=np.nan)
    ohrc_lat = RegularGridInterpolator((scan_a, px_a), ohrc_lat_g,
                                       bounds_error=False, fill_value=np.nan)

    lon, lat, ln, sm = load_nac_geom(NAC_GEO)
    inv = NacInverse(lon, lat, ln, sm)
    line0 = lon[0]
    lat0 = lat[0]
    lonN = lon[len(lon) - 1]
    latN = lat[len(lat) - 1]
    dLon = (lonN - line0) % 360.0
    if dLon > 180:
        dLon -= 360
    dLat = latN - lat0
    along_m = abs(dLat) * DEG_KM * 1000
    across_m = abs(dLon) * DEG_KM * 1000
    print(f"NAC across-track ~ {across_m/(sm[-1]-sm[0]):.2f} m/px, "
          f"along-track ~ {along_m/(ln[-1]-ln[0]):.2f} m/px", flush=True)

    s2 = args.size // 2
    sc_axis = np.arange(args.scan - s2 * args.stride,
                        args.scan + s2 * args.stride, args.stride)
    px_axis = np.arange(args.px - s2 * args.stride,
                        args.px + s2 * args.stride, args.stride)
    SC, PX = np.meshgrid(sc_axis, px_axis, indexing="ij")
    q = np.column_stack([SC.ravel(), PX.ravel()])
    lon_g = np.asarray(ohrc_lon(q).reshape(SC.shape), np.float64) % 360.0
    lat_g = np.asarray(ohrc_lat(q).reshape(SC.shape), np.float64)
    src = ohrc[SC.astype(int), PX.astype(int)]
    col, line = inv.apply(lon_g, lat_g)
    ref = remap_nac(nac, line, col)
    src_u8 = as_u8(src)
    ref_u8 = as_u8(ref)
    gc_lon = float(np.nanmean(lon_g))
    gc_lat = float(np.nanmean(lat_g))

    Path(args.out).mkdir(parents=True, exist_ok=True)
    cv2.imwrite(f"{args.out}/src.png", src_u8)
    cv2.imwrite(f"{args.out}/ref.png", ref_u8)
    nin, ioc, ncc = sift_match(src_u8, ref_u8)
    res_m = 1.17 * args.stride
    print(f"crop src={src_u8.shape} ref={ref_u8.shape} "
          f"ground center ({gc_lon:.4f}, {gc_lat:.4f})")
    print(f"grid res ~ {res_m:.1f} m/px")
    print(f"SIFT inliers={nin} inlier_over_ratio={ioc:.3f} NCC={ncc:+.3f} "
          f"[{time.time()-t0:.0f}s]")
    print(f"images at {args.out}/src.png and {args.out}/ref.png")


if __name__ == "__main__":
    main()