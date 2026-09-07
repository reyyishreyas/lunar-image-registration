"""Phase 1 preprocessing: georeference an OHRC / LRO NAC pair.

Stage contract (AI_EXECUTION_PLAN.md Step 1.1):
    crop/resize both images to the same approximate ground area so that the
    feature-matching stages (Steps 1.2-1.5) receive aligned 1024x1024 inputs.

OHRC (raw .img, 12000 px x 93693 scan, uint8) is georeferenced directly from the
ISRO geometry CSV (pixel, scan) -> (lon, lat) regular grid.
LRO NAC (.IMG, PDS3 EDR/CDR) carries no embedded georeference; its placement
relative to the OHRC scene is derived by feature co-registration (SIFT) between
the OHRC working image and the NAC working image, including vertical-mirror
handling (LROC and OHRC are pushbroom products commonly delivered mirrored).
The resulting homography maps a chosen OHRC crop onto the NAC footprint.

Outputs (written by georeference_pair):
    pair1_src.png  - 1024x1024 OHRC crop (source image for the pipeline)
    pair1_ref.png  - 1024x1024 NAC crop warped to the same ground area (reference)
    georef_pair1.json - derived transform + NAC corner ground coords for records
"""

from __future__ import annotations

import csv
import json
import os
from typing import Tuple

import cv2
import numpy as np
import rasterio
from rasterio.errors import NotGeoreferencedWarning

import warnings

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

OHRC_SHAPE = (93693, 12000)  # (scan, pixel), uint8
OHRC_WS_FACTOR = 10  # working-set downsampling factor for OHRC
NAC_WS_FACTOR = 8  # working-set downsampling factor for LRO NAC


def read_ohrc_raw(img_path: str) -> np.ndarray:
    """Read the full raw OHRC .img as uint8 (scan, pixel)."""
    arr = np.fromfile(img_path, dtype=np.uint8)
    arr = arr.reshape(OHRC_SHAPE)
    return arr


def read_ohrc_ground_grid(csv_path: str):
    """Build (pixel, scan) -> (lon, lat) interpolators from the ISRO geometry CSV.

    Returns (pix_axis, scan_axis, lon_grid, lat_grid) where grid[i, j] is the
    value at (scan=pix_axis[i], pixel=scan_axis[j])? -- see docstring below.

    Columns: Longitude, Latitude, Pixel, Scan -> sampled every 100 px / scan.
    """
    with open(csv_path, newline="") as fh:
        rows = [
            (float(r["Longitude"]), float(r["Latitude"]), int(r["Pixel"]), int(r["Scan"]))
            for r in csv.DictReader(fh)
        ]
    lons = np.array([r[0] for r in rows], dtype=np.float64)
    lats = np.array([r[1] for r in rows], dtype=np.float64)
    pix = np.array([r[2] for r in rows], dtype=np.int64)
    scan = np.array([r[3] for r in rows], dtype=np.int64)

    pixs = np.unique(pix)
    scans = np.unique(scan)
    lon_grid = np.empty((scans.size, pixs.size))
    lat_grid = np.empty_like(lon_grid)
    idx = {(int(p), int(s)): i for i, (_, _, p, s) in enumerate(zip(lons, lats, pix, scan))}
    for j, p in enumerate(pixs):
        for i, s in enumerate(scans):
            lon_grid[i, j] = lons[idx[(p, s)]]
            lat_grid[i, j] = lats[idx[(p, s)]]
    return pixs, scans, lon_grid, lat_grid


def ground_from_ohrc(pix_axis, scan_axis, lon_grid, lat_grid, pixel, scan):
    """Interpolated (lon, lat) at float (pixel, scan)."""
    from scipy.interpolate import RegularGridInterpolator

    ilon = RegularGridInterpolator(
        (scan_axis, pix_axis), lon_grid, method="linear", bounds_error=False, fill_value=None
    )
    ilat = RegularGridInterpolator(
        (scan_axis, pix_axis), lat_grid, method="linear", bounds_error=False, fill_value=None
    )
    pt = np.array([[float(scan), float(pixel)]])
    return float(ilon(pt)[0]), float(ilat(pt)[0])


def _normalize8(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    out = np.where(np.isnan(arr), 0.0, arr)
    out = np.clip((out - lo) / (hi - lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def read_nac_img(img_path: str) -> np.ndarray:
    """Read LRO NAC .IMG; null == -32768 -> NaN mask (kept out of stats)."""
    with rasterio.open(img_path) as ds:
        arr = ds.read(1).astype(np.float32)
    arr = np.where(arr <= -30000, np.nan, arr)
    return arr


def _prep_sift(im: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(32, 32)).apply(im)


def find_pair_transform(ohrc_ws, nac, nac_fac=NAC_WS_FACTOR):
    """Estimate mapping OHRC working-set -> NAC working-set.

    ohrc_ws: (rows, cols) uint8 working image of the OHRC scene (1/10 res).
    nac: full-resolution NAC array with NaN nulls.

    The true mirror orientation is decided by normalized cross-correlation of the
    warped NAC against the OHRC working image (SIFT inlier counts alone can tie
    between flipH/flipV because both produce a fitted RANSAC model).

    Returns dict with H (homography ohrc_ws px -> nac working px), nac_fac,
    flip (str), inliers, good_matches, ncc, nac_ws_shape.
    """
    b = _prep_sift(ohrc_ws)
    sub = nac[::nac_fac, ::nac_fac]
    lo, hi = np.nanpercentile(sub, [2, 98])
    n8 = _normalize8(sub, lo, hi)
    n8[np.isnan(sub)] = 0
    n8_mask = (~np.isnan(sub)).astype(np.float32)

    sift = cv2.SIFT_create(nfeatures=15000, contrastThreshold=0.01, edgeThreshold=10)
    kpb, db = sift.detectAndCompute(b, None)
    bfm = cv2.BFMatcher()

    def ncc_overlap(H, nn, mask_nn):
        Hinv = np.linalg.inv(H)
        warped = cv2.warpPerspective(nn, Hinv, (b.shape[1], b.shape[0]),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wmask = cv2.warpPerspective(mask_nn, Hinv, (b.shape[1], b.shape[0]),
                                    flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ok = (wmask > 0.5) & (b > 0)
        if ok.sum() < 2000:
            return -1.0
        x = b[ok].astype(np.float32)
        y = warped[ok].astype(np.float32)
        x = (x - x.mean()) / (x.std() + 1e-6)
        y = (y - y.mean()) / (y.std() + 1e-6)
        return float((x * y).mean())

    candidates = []
    for name, flips in (("none", (False, False)), ("flipV", (True, False)),
                        ("flipH", (False, True)), ("flipHV", (True, True))):
        nn = n8
        if flips[0]:
            nn = np.flipud(nn)
        if flips[1]:
            nn = np.fliplr(nn)
        kpn, dn = sift.detectAndCompute(_prep_sift(nn), None)
        if dn is None or len(dn) < 8:
            continue
        raw = bfm.knnMatch(db, dn, k=2)
        good = [m for m, z in raw if m.distance < 0.75 * z.distance]
        if len(good) < 8:
            continue
        src = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kpn[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if mask is None:
            continue
        inl = int(mask.sum())
        ncc = ncc_overlap(H, nn, n8_mask)
        candidates.append(dict(flip=name, H=H, inliers=inl, good=len(good),
                               ncc=ncc, nac_ws_shape=nn.shape, nac_fac=nac_fac))
    if not candidates:
        return None
    best_inl = max(c["inliers"] for c in candidates)
    pools = [c for c in candidates if c["inliers"] >= max(50, 0.6 * best_inl)]
    best = max(pools, key=lambda c: c["ncc"]) if pools else max(candidates,
                                                                key=lambda c: c["inliers"])
    # overlap region of the NAC (as warped into OHRC working space) -> bbox
    Hinv = np.linalg.inv(best["H"])
    wmask = cv2.warpPerspective(n8_mask, Hinv, (b.shape[1], b.shape[0]),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.where(wmask > 0.5)
    best["overlap_bbox"] = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return best


def georeference_pair(ohrc_img_path, ohrc_csv_path, nac_img_path, out_dir,
                      crop_px=1024):
    """Crop pair1 to 1024x1024 same-ground-area; write PNGs + JSON.

    Returns dict of outputs + derived NAC corner ground coords.
    """
    os.makedirs(out_dir, exist_ok=True)

    ohrc = read_ohrc_raw(ohrc_img_path)
    ohrc_ws = ohrc[::OHRC_WS_FACTOR, ::OHRC_WS_FACTOR]  # (9369, 1200)
    nac = read_nac_img(nac_img_path)
    pix_axis, scan_axis, lon_grid, lat_grid = read_ohrc_ground_grid(ohrc_csv_path)

    t = find_pair_transform(ohrc_ws, nac)
    if t is None or t["inliers"] < 20:
        raise RuntimeError(
            f"georeference: no reliable OHRC<->NAC overlap (inliers="
            f"{(t or {}).get('inliers')}). Cannot produce same-ground-area crops."
        )

    H = t["H"]
    h_ws, w_ws = t["nac_ws_shape"]
    fac = t["nac_fac"]
    flipv = t["flip"] in ("flipV", "flipHV")
    fliph = t["flip"] in ("flipH", "flipHV")

    # Crop centre in OHRC working-set space = centre of the NAC overlap region.
    x_min, y_min, x_max, y_max = t["overlap_bbox"]
    cx = float(np.clip(0.5 * (x_min + x_max), 51, w_ws - 51))
    cy = float(np.clip(0.5 * (y_min + y_max), 51, h_ws - 51))
    half = 0.5 * (crop_px / OHRC_WS_FACTOR)  # 51.2 px working-set

    corners_ws = np.float32(
        [[cx - half, cy - half], [cx + half, cy - half],
         [cx - half, cy + half], [cx + half, cy + half]]
    ).reshape(-1, 1, 2)

    # ---- OHRC full-res crop ----
    p0 = int(round((cx - half) * OHRC_WS_FACTOR))
    s0 = int(round((cy - half) * OHRC_WS_FACTOR))
    p0 = max(0, min(p0, OHRC_SHAPE[1] - crop_px))
    s0 = max(0, min(s0, OHRC_SHAPE[0] - crop_px))
    src_crop = ohrc[s0:s0 + crop_px, p0:p0 + crop_px]
    del ohrc
    src_png = np.clip(src_crop.astype(np.float32) / 255.0, 0, 1)
    src_png = (src_png * 255).astype(np.uint8)

    # ---- Map crop corners -> NAC working set -> NAC native pixels ----
    corners_nac_ws = cv2.perspectiveTransform(corners_ws, H).reshape(-1, 2)
    r_rows = corners_nac_ws[:, 1]
    c_cols = corners_nac_ws[:, 0]
    if flipv:
        row_native = (h_ws - 1 - r_rows) * fac
    else:
        row_native = r_rows * fac
    col_native = (w_ws - 1 - c_cols) * fac if fliph else c_cols * fac
    nac_corners = np.stack([col_native, row_native], axis=1).astype(np.float32)

    dst_grid = np.float32(
        [[0, 0], [crop_px - 1, 0], [0, crop_px - 1], [crop_px - 1, crop_px - 1]]
    )
    # warpPerspective rejects source dims >= SHRT_MAX; the full NAC height (46080)
    # exceeds that, so warp only a sub-region around the mapped corners.
    margin = 64
    x0 = int(max(0, np.floor(nac_corners[:, 0].min()) - margin))
    y0 = int(max(0, np.floor(nac_corners[:, 1].min()) - margin))
    x1 = int(min(nac.shape[1], np.ceil(nac_corners[:, 0].max()) + margin))
    y1 = int(min(nac.shape[0], np.ceil(nac_corners[:, 1].max()) + margin))
    region = nac[y0:y1, x0:x1]
    nac_corners_local = nac_corners - np.float32([x0, y0])

    P = cv2.getPerspectiveTransform(nac_corners_local, dst_grid)
    nac_filled = np.where(np.isnan(region), 0.0, region)
    lo, hi = np.nanpercentile(region, [2, 98])
    nac_8 = _normalize8(nac_filled, lo, hi)
    ref_crop = cv2.warpPerspective(nac_8, P, (crop_px, crop_px),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # ---- Outputs ----
    src_path = os.path.join(out_dir, "pair1_src.png")
    ref_path = os.path.join(out_dir, "pair1_ref.png")
    cv2.imwrite(src_path, src_png)
    cv2.imwrite(ref_path, ref_crop)

    # Derived NAC corner ground coords via the OHRC grid at the 4 crop corners.
    ground_corners = [
        ground_from_ohrc(pix_axis, scan_axis, lon_grid, lat_grid,
                         p0 + (cx - half) * OHRC_WS_FACTOR, s0 + (cy - half) * OHRC_WS_FACTOR),
        ground_from_ohrc(pix_axis, scan_axis, lon_grid, lat_grid,
                         p0 + (cx + half) * OHRC_WS_FACTOR, s0 + (cy - half) * OHRC_WS_FACTOR),
        ground_from_ohrc(pix_axis, scan_axis, lon_grid, lat_grid,
                         p0 + (cx - half) * OHRC_WS_FACTOR, s0 + (cy + half) * OHRC_WS_FACTOR),
        ground_from_ohrc(pix_axis, scan_axis, lon_grid, lat_grid,
                         p0 + (cx + half) * OHRC_WS_FACTOR, s0 + (cy + half) * OHRC_WS_FACTOR),
    ]

    meta = {
        "ohrc_crop": {"pixel": p0, "scan": s0, "width": crop_px, "height": crop_px},
        "ohrc_ws_factor": OHRC_WS_FACTOR,
        "nac_ws_factor": fac,
        "nac_flip": t["flip"],
        "inliers": t["inliers"],
        "good_matches": t["good"],
        "H_ohrc_ws_to_nac_ws": H.tolist(),
        "nac_crop_corners_native": nac_corners.tolist(),
        "ground_corners_lon_lat": ground_corners,
        "src_png": src_path,
        "ref_png": ref_path,
    }
    with open(os.path.join(out_dir, "georef_pair1.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


if __name__ == "__main__":
    import sys

    m = georeference_pair(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    )
    print(json.dumps(m, indent=2))