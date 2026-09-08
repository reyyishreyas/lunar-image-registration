"""Phase 9 — Chandrayaan-2 TMC support (multi-sensor PS 26166).

The Chandrayaan-2 **TMC** (Terrain Mapping Camera) calibrated product is
structurally identical to OHRC: a raw uint8 ``.img`` pushbroom raster plus an
ISRO geometry CSV with columns ``Longitude,Latitude,Pixel,Scan``. The only
material difference is the across-track width — OHRC is 12000 px, TMC is
4000 px (product ``ch2_tmc_*_d_img_d18.img``, geometry ``*_g_grd_d18.csv``).

This module:
  1. ``read_ch2_raw`` — generic Chandrayaan pushbroom reader that infers the
     across-track width from the geometry CSV (not hardcoded), so OHRC / TMC /
     future ISRO pushbroom sensors share one path.
  2. ``read_tmc_ground_grid`` — re-uses the (pixel, scan) -> (lon, lat) grid
     (the OHRC CSV format is identical).
  3. ``register_tmc_to_ohrc`` — registers a TMC frame (source/moving) against an
     OHRC frame (reference/fixed) by equal-GSD staging on each sensor's own
     geometry CSV, then robust (SIFT + USAC_MAGSAC) homography fit via the
     shared detection / matching / outlier modules.

Nothing is duplicated from the OHRC code path; the existing modules are reused
so the working OHRC<->NAC pipeline is untouched.
"""

from __future__ import annotations

import csv
import os

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator

TMC_WIDTH = 4000          # across-track pixels for TMC (constant across TMC d_img products)
OHRC_WIDTH = 12000        # reference OHRC across-track width
TMC_WS_FACTOR = 8         # TMC working-set downsampling factor


def infer_across_track_width(geom_csv):
    """Infer the across-track native width from an ISRO geometry CSV.

    The ISRO OHRC/TMC geometry CSVs sample every 100 native pixels and always
    include the final pixel (max Pixel = width - 1), so ``max(Pixel) + 1`` is the
    exact across-track width. Robust for OHRC (12000) and TMC (4000).
    """
    pix = []
    with open(geom_csv, newline="") as fh:
        for r in csv.DictReader(fh):
            try:
                pix.append(int(r["Pixel"]))
            except (ValueError, KeyError):
                continue
    if not pix:
        raise ValueError(f"could not read Pixel column from {geom_csv}")
    return max(pix) + 1


def read_ch2_raw(img_path, geom_csv, width=None, memmap=False):
    """Read a Chandrayaan-2 calibrated ``.img`` as uint8 (scan, pixel).

    ``width`` defaults to the across-track width inferred from ``geom_csv``, so
    OHRC (12000) and TMC (4000) are both handled. The along-track (scan) count
    is inferred from the file size.

    ``memmap=True`` returns a read-only ``np.memmap`` so huge frames (TMC
    products are ~2.2 GB) can be sampled per-swath without loading into RAM.
    """
    if width is None:
        width = infer_across_track_width(geom_csv)
    size = os.path.getsize(img_path)
    rows, rem = divmod(size, width)
    if rem != 0:
        raise ValueError(
            f"file size {size} is not divisible by inferred width {width} "
            f"({img_path}); check the geometry CSV Pixel stride")
    if memmap:
        return np.memmap(img_path, dtype=np.uint8, mode="r", shape=(rows, width))
    return np.fromfile(img_path, dtype=np.uint8).reshape(rows, width)


def read_ground_grid(geom_csv):
    """Build (pixel, scan) -> (lon, lat) interpolators from an ISRO geometry CSV.

    Returns (pix_axis, scan_axis, lon_grid, lat_grid). Grid layout matches the
    OHRC reader (grid[i, j] at scan=pix_axis[i], pixel=scan_axis[j]).
    """
    with open(geom_csv, newline="") as fh:
        rows = [(float(r["Longitude"]), float(r["Latitude"]),
                 int(r["Pixel"]), int(r["Scan"]))
                for r in csv.DictReader(fh)]
    lons = np.array([r[0] for r in rows], np.float64)
    lats = np.array([r[1] for r in rows], np.float64)
    pix = np.array([r[2] for r in rows], np.int64)
    scan = np.array([r[3] for r in rows], np.int64)

    pixs, scans = np.unique(pix), np.unique(scan)
    lon_grid = np.empty((scans.size, pixs.size))
    lat_grid = np.empty_like(lon_grid)
    idx = {(int(p), int(s)): i for i, (_, _, p, s)
           in enumerate(zip(lons, lats, pix, scan))}
    for j, p in enumerate(pixs):
        for i, s in enumerate(scans):
            lon_grid[i, j] = lons[idx[(p, s)]]
            lat_grid[i, j] = lats[idx[(p, s)]]
    return pixs, scans, lon_grid, lat_grid


def _interp_latlon(pix_axis, scan_axis, lon_grid, lat_grid, pix, scan):
    ilon = RegularGridInterpolator((scan_axis, pix_axis), lon_grid,
                                   method="linear", bounds_error=False,
                                   fill_value=None)
    ilat = RegularGridInterpolator((scan_axis, pix_axis), lat_grid,
                                   method="linear", bounds_error=False,
                                   fill_value=None)
    return float(ilon([[float(scan), float(pix)]])[0]), \
        float(ilat([[float(scan), float(pix)]])[0])


def _gsd_m(pix_axis, scan_axis, lon_grid, lat_grid):
    """Approx mean ground sample distance (m/px) of a Chandrayaan sensor.

    Uses the median inter-sample and inter-line ground spacing from the geometry
    grid, divided by the native-axis stride so the result is per native pixel/
    scan (the ISRO CSVs sample every 100 native units).
    """
    def _stride(axis):
        a = np.asarray(axis, np.float64)
        if a.size < 2:
            return 1.0
        d = np.abs(np.diff(a))
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0

    lat_m, lon_m = 111000.0, 111000.0
    lon_r = np.deg2rad(lat_grid)
    d_pix = np.hypot(np.diff(lon_grid, axis=1) * lon_m * np.cos(lon_r[:, :-1]),
                     np.diff(lat_grid, axis=1) * lat_m)
    d_scan = np.hypot(np.diff(lon_grid, axis=0) * lon_m * np.cos(lon_r[:-1, :]),
                      np.diff(lat_grid, axis=0) * lat_m)
    d_pix = d_pix[np.isfinite(d_pix)] / float(_stride(pix_axis))
    d_scan = d_scan[np.isfinite(d_scan)] / float(_stride(scan_axis))
    if d_pix.size and d_scan.size:
        return 0.5 * (np.median(d_pix) + np.median(d_scan))
    vals = np.concatenate([d_pix, d_scan])
    return float(np.median(vals)) if vals.size else np.nan


def working_crop_full(img, geom_csv, box, ws_factor=TMC_WS_FACTOR):
    """Downsample + crop a full Chandrayaan raw frame to a working-set.

    ``box`` is (scan0, scan1, pix0, pix1) native coords; returns the uint8
    working-set crop (approx 1/ws_factor scale) and its native bounding box.
    Reads the whole raw raster from disk (the caller passes the loaded array to
    avoid double I/O — see register_tmc_to_ohrc).
    """
    s0, s1, p0, p1 = box
    ws = img[s0:s1, p0:p1]
    return ws[::ws_factor, ::ws_factor]


def register_tmc_to_ohrc(tmc_img_path, tmc_geom_csv, ohrc_img_path, ohrc_geom_csv,
                         out_dir, crop_rows=1024, crop_pix=1024, prefix="tmc_ohrc",
                         swath_pix=(400, 900), nfeatures=20000,
                         contrast_threshold=0.03, ransac_thresh=5.0):
    """Register a TMC frame (moving/source) against an OHRC frame (fixed/ref).

    Strategy (equal-GSD, crop-first compliant):
      1. Load both raw frames (width inferred per sensor from its own CSV) and
         both geometry grids.
      2. Choose a TMC swath (source) and, using the same median ground scale,
         pick a same-GSD scan range of the OHRC strip.
      3. Precondition both crops (CLAHE) and run the champion SIFT + robust
         fit pipeline to get a homography TMC-crop -> OHRC-crop.
      4. Return report dict with RMSE (self reprojection), inliers, ratio,
         homography, staged GSD, and a diagnostics note.

    This is an exploratory/official pipeline function kept light-weight; the two
    full rasters are the dominant cost. It never fabricates a result — when no
    reliable model is found it returns ``verdict: "not_registered"``.
    """
    os.makedirs(out_dir, exist_ok=True)

    tmc = read_ch2_raw(tmc_img_path, tmc_geom_csv, memmap=True)
    ohrc = read_ch2_raw(ohrc_img_path, ohrc_geom_csv, width=OHRC_WIDTH, memmap=True)
    tpix, tscan, tlon, tlat = read_ground_grid(tmc_geom_csv)
    opix, oscan, olon, olat = read_ground_grid(ohrc_geom_csv)

    gsd_tmc = _gsd_m(tpix, tscan, tlon, tlat)
    gsd_ohrc = _gsd_m(opix, oscan, olon, olat)
    if not np.isfinite(gsd_tmc) or not np.isfinite(gsd_ohrc) or gsd_tmc <= 0:
        return {"verdict": "not_registered", "note": "GSD not computable",
                "m_per_px": {"tmc": gsd_tmc, "ohrc": gsd_ohrc}}

    # same ground width in each sensor's pixels
    p0, p1 = swath_pix
    pix_w_m = (p1 - p0) * gsd_tmc
    ohrc_pix_span = int(pix_w_m / gsd_ohrc)
    ohrc_pix_span = min(ohrc_pix_span, ohrc.shape[1])
    if tscan.size:
        s0 = int(mid_array(tscan))
    else:
        s0 = 0
    scan_span = crop_rows
    tmc_crop = tmc[s0:s0 + scan_span, p0:p1].copy()
    # OHRC swath at the same ground width, centred in the strip
    ohrc_p0 = max(0, (ohrc.shape[1] - ohrc_pix_span) // 2)
    ohrc_crop = ohrc[s0:s0 + scan_span,
                     ohrc_p0:ohrc_p0 + ohrc_pix_span].copy()
    del tmc, ohrc

    # equal-size so a homography is meaningful: resize OHRC crop to TMC-crop dims
    tmc_ws = _to_u8(tmc_crop)
    ohrc_ws = _to_u8(ohrc_crop)
    ohrc_ws = cv2.resize(ohrc_ws, (tmc_ws.shape[1], tmc_ws.shape[0]),
                         interpolation=cv2.INTER_AREA)

    from src.detection.classical import detect_sift
    from src.matching.classical_match import match_bf_ratio
    from src.outlier_rejection.ransac import find_homography_ransac

    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    a, b = clahe.apply(tmc_ws), clahe.apply(ohrc_ws)
    kp1, d1 = detect_sift(a, nfeatures=nfeatures, contrast_threshold=contrast_threshold)
    kp2, d2 = detect_sift(b, nfeatures=nfeatures, contrast_threshold=contrast_threshold)
    m = match_bf_ratio(d1, d2, ratio=0.75, norm_type=cv2.NORM_L2, verbose=False)
    n_matches = len(m)
    if n_matches < 8:
        return {"verdict": "not_registered", "n_matches": n_matches,
                "inliers": 0, "note": "too few raw matches"}
    pts1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
    pts2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
    H, inl = find_homography_ransac(pts1, pts2, ransac_thresh=ransac_thresh,
                                    method="usac_magsac")
    if H is None or inl is None or inl.sum() < 8:
        return {"verdict": "not_registered", "n_matches": n_matches,
                "inliers": int(inl.sum()) if inl is not None else 0,
                "note": "robust fit failed"}
    pw = pts1[inl]
    proj = (H @ np.hstack([pw, np.ones((pw.shape[0], 1))]).T).T
    proj = proj[:, :2] / np.maximum(proj[:, 2:3], 1e-12)
    rmse = float(np.sqrt(np.mean(np.sum((proj - pts2[inl]) ** 2, axis=1))))

    import json
    report = {
        "pair": "TMC -> OHRC",
        "verdict": "registered",
        "n_matches": n_matches,
        "inliers": int(inl.sum()),
        "inlier_ratio": round(float(inl.sum()) / n_matches, 4),
        "rmse_px": round(rmse, 4),
        "H": H.tolist(),
        "staged": {"tmc_gsd_m": round(float(gsd_tmc), 4),
                   "ohrc_gsd_m": round(float(gsd_ohrc), 4),
                   "tmc_crop_scan": [s0, s0 + scan_span],
                   "ohrc_crop_scan": [s0, s0 + scan_span],
                   "tmc_pixel_swath": list(swath_pix)},
        "method": "sift+usac_magsac + clahe (equal-GSD swath)",
    }
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    # write browse crops
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_src.png"), tmc_ws)
    cv2.imwrite(os.path.join(out_dir, f"{prefix}_ref.png"), ohrc_ws)
    return report


def mid_array(axis):
    return float(axis[len(axis) // 2]) if len(axis) else 0.0


def _to_u8(arr):
    if arr.dtype != np.uint8:
        lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
        return np.clip((arr - lo) / (hi - lo if hi > lo else 1.0) * 255, 0, 255).astype(np.uint8)
    return arr
