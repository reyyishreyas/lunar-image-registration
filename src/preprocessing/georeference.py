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

OHRC_SHAPE = (93693, 12000)  # (scan, pixel), uint8 — Phase 1 default; actual rows inferred from file size
OHRC_WIDTH = 12000  # across-track pixels (constant across OHRC products)
OHRC_WS_FACTOR = 10  # working-set downsampling factor for OHRC
NAC_WS_FACTOR = 8  # working-set downsampling factor for LRO NAC


def ws_scale(H):
    """Mean ground scale of H (ohrc_ws px -> nac_ws px).

    For a homography, the local linear part is H[:2, :2]; the length of an
    OHRC working-set unit vector after mapping equals the number of NAC
    working-set pixels it spans. mean(|H*u_x|, |H*u_y|) is that factor.
    """
    a = np.asarray(H, np.float64)[:2, :2]
    vx = np.linalg.norm(a[:, 0])
    vy = np.linalg.norm(a[:, 1])
    return float(0.5 * (vx + vy))


def refine_ws_factor(fac, scale, lo=1, hi=256):
    """Next NAC working-set factor so OHRC and NAC working sets share a GSD.

    An OHRC working set at native scale s_ohrc and NAC set at s_nac*fac are
    equal-GSD iff s_ohrc = s_nac*fac*scale(H), i.e. fac' = fac*scale(H).
    """
    f = int(round(float(fac) * float(scale)))
    return int(min(max(f, lo), hi))


def read_ohrc_raw(img_path: str) -> np.ndarray:
    """Read the full raw OHRC .img as uint8 (scan, pixel).

    The across-track width is constant (12000 px); the along-track scan-line
    count is inferred from the file size so products of different lengths (Phase
    1: 93693, Phase 5: 93692) are read correctly.
    """
    size = os.path.getsize(img_path)
    rows, rem = divmod(size, OHRC_WIDTH)
    if rem != 0:
        raise ValueError(
            f"OHRC file size {size} is not divisible by {OHRC_WIDTH} px; "
            f"unexpected product layout for {img_path}"
        )
    arr = np.fromfile(img_path, dtype=np.uint8)
    return arr.reshape(rows, OHRC_WIDTH)


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


def _ncc_overlap(H, nn, mask_nn, b):
    """Normalized cross-correlation of the warped nn over the OHRC working image.

    Returns (ncc, overlap_bbox) where bbox is [x_min, y_min, x_max, y_max] of the
    overlap in OHRC working-set pixel space (None if too little overlap).
    """
    Hinv = np.linalg.inv(H)
    warped = cv2.warpPerspective(nn, Hinv, (b.shape[1], b.shape[0]),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    wmask = cv2.warpPerspective(mask_nn, Hinv, (b.shape[1], b.shape[0]),
                                flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ok = (wmask > 0.5) & (b > 0)
    if ok.sum() < 2000:
        return -1.0, None
    x = b[ok].astype(np.float32)
    y = warped[ok].astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-6)
    y = (y - y.mean()) / (y.std() + 1e-6)
    ys, xs = np.where(wmask > 0.5)
    bbox = [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return float((x * y).mean()), bbox


def candidate_stats(ohrc_ws, nac, nac_fac=NAC_WS_FACTOR, nfeatures=15000,
                    contrast_threshold=0.01, edge_threshold=10,
                    ratio=0.75, ransac_thresh=5.0):
    """Run the 4-orientation SIFT overlap search and return per-flip statistics.

    Returns a dict {candidates: [...], sift_params: {...}}. Each candidate holds
    flip, good, inliers, inlier_ratio, ncc, bbox. Pure function used by both the
    primary search (find_pair_transform) and the robust fallback, and dumped to
    georef diagnostics on failure.
    """
    b = _prep_sift(ohrc_ws)
    sub = nac[::nac_fac, ::nac_fac]
    lo, hi = np.nanpercentile(sub, [2, 98])
    n8 = _normalize8(sub, lo, hi)
    n8[np.isnan(sub)] = 0
    n8_mask = (~np.isnan(sub)).astype(np.float32)

    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold)
    kpb, db = sift.detectAndCompute(b, None)
    bfm = cv2.BFMatcher()

    out = []
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
        good = [m for m, z in raw if m.distance < ratio * z.distance]
        if len(good) < 4:
            out.append(dict(flip=name, good=len(good), inliers=0, inlier_ratio=0.0,
                            ncc=-1.0, bbox=None))
            continue
        src = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kpn[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        best = None
        for method in (cv2.RANSAC, cv2.USAC_MAGSAC):
            H, mask = cv2.findHomography(src, dst, method, ransac_thresh)
            inl = int(mask.sum()) if mask is not None else 0
            if best is None or inl > best[0]:
                best = (inl, H, mask, method)
        inl, H, mask, used = best
        if H is None:
            out.append(dict(flip=name, good=len(good), inliers=inl,
                            inlier_ratio=inl / len(good), ncc=-1.0, bbox=None))
            continue
        ncc, bbox = _ncc_overlap(H, nn, n8_mask, b)
        out.append(dict(flip=name, good=len(good), inliers=inl,
                        inlier_ratio=inl / len(good), ncc=ncc, bbox=bbox))
    return {"candidates": out,
            "sift_params": dict(nfeatures=nfeatures, contrast_threshold=contrast_threshold,
                                edge_threshold=edge_threshold, ratio=ratio,
                                ransac_thresh=ransac_thresh, nac_fac=nac_fac)}


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
        if H is None:
            continue
        inl = int(mask.sum()) if mask is not None else 0
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


def find_pair_transform_robust(ohrc_ws, nac, nac_fac=NAC_WS_FACTOR,
                               nfeatures=30000, contrast_threshold=0.006,
                               edge_threshold=12, ratio=0.8):
    """Boosted fallback overlap search with proof-based acceptance.

    Used when the primary search finds <20 inliers. Runs the same 4-orientation
    SIFT search with a denser extractor and USAC_MAGSAC, then ACCEPTS a
    candidate only when it passes all of:
      * winning-orientation inliers >= 20 and inlier ratio >= 0.25
      * MIRROR-GROUP dominance: the winning orientation's flip-group
        (id/HV = 0/180 deg rotation vs V/H = transpositions) must hold >= 2x the
        inliers of the losing group. SIFT + craters are rotation-invariant, so a
        genuine overlap produces a strong id/HV (or V/H) pair while the other
        group stays near noise; on a truly non-overlapping pair every orientation
        stays weak and no group dominates.
      * warped-overlap NCC of the winner >= 0.15 (structural, not just
        keypoint, agreement)
      * overlap bbox is a sane sub-region (not the fully-degenerate whole frame)

    Returns (t_dict_or_None, stats). t-dict has same schema as
    find_pair_transform; stats is the full candidate table used for the decision
    (also written to diagnostics).
    """
    stats = candidate_stats(ohrc_ws, nac, nac_fac=nac_fac, nfeatures=nfeatures,
                            contrast_threshold=contrast_threshold,
                            edge_threshold=edge_threshold, ratio=ratio)
    cands = [c for c in stats["candidates"]]
    if len(cands) < 2:
        return None, stats

    def _group(c):
        return 0 if c["flip"] in ("none", "flipHV") else 1

    by_group = {}
    for c in cands:
        by_group.setdefault(_group(c), []).append(c)
    if len(by_group) < 2:
        return None, stats
    g_max = {}
    for g, cs in by_group.items():
        g_max[g] = max(cs, key=lambda c: (c["inliers"], c["ncc"]))
    loser, winner = sorted(g_max.items(), key=lambda kv: kv[1]["inliers"])
    win, lose = winner[1], loser[1]
    if win["inliers"] < 20 or win["good"] == 0:
        return None, stats
    if (win["inliers"] / win["good"]) < 0.25:
        return None, stats
    if win["inliers"] < 2.0 * max(lose["inliers"], 1):
        return None, stats
    if win["ncc"] < 0.15 or win["bbox"] is None:
        return None, stats
    area = (win["bbox"][2] - win["bbox"][0]) * (win["bbox"][3] - win["bbox"][1])
    if not (0.01 < area / (ohrc_ws.shape[0] * ohrc_ws.shape[1]) < 0.9):
        return None, stats
    # rebuild the winning candidate transform (deterministic, single flip)
    b = _prep_sift(ohrc_ws)
    sub = nac[::nac_fac, ::nac_fac]
    lo, hi = np.nanpercentile(sub, [2, 98])
    n8 = _normalize8(sub, lo, hi)
    n8[np.isnan(sub)] = 0
    nn = n8
    if win["flip"] in ("flipV", "flipHV"):
        nn = np.flipud(nn)
    if win["flip"] in ("flipH", "flipHV"):
        nn = np.fliplr(nn)
    sift = cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast_threshold,
                           edgeThreshold=edge_threshold)
    kpb, db = sift.detectAndCompute(b, None)
    kpn, dn = sift.detectAndCompute(_prep_sift(nn), None)
    bfm = cv2.BFMatcher()
    raw = bfm.knnMatch(db, dn, k=2)
    good = [m for m, z in raw if m.distance < ratio * z.distance]
    src = np.float32([kpb[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kpn[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 5.0)
    if H is None:
        return None, stats
    inl = int(mask.sum()) if mask is not None else 0
    ncc, bbox = _ncc_overlap(H, nn, (~np.isnan(sub)).astype(np.float32), b)
    return (dict(flip=win["flip"], H=H, inliers=inl, good=len(good), ncc=ncc,
                 overlap_bbox=bbox, nac_ws_shape=nn.shape, nac_fac=nac_fac,
                 method="sparse-robust", stats=stats), stats)


def _set_georeference_pair(ohrc, ohrc_ws, nac, pix_axis, scan_axis, lon_grid,
                           lat_grid, t, out_dir, crop_px, prefix):

    def join(p):
        return os.path.join(out_dir, p)

    os.makedirs(out_dir, exist_ok=True)
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
    p0 = max(0, min(p0, ohrc.shape[1] - crop_px))
    s0 = max(0, min(s0, ohrc.shape[0] - crop_px))
    src_crop = ohrc[s0:s0 + crop_px, p0:p0 + crop_px]
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
    src_path = join(f"{prefix}_src.png")
    ref_path = join(f"{prefix}_ref.png")
    cv2.imwrite(src_path, src_png)
    cv2.imwrite(ref_path, ref_crop)

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
        "crop_gsd_m": _crop_gsd_metres(pix_axis, scan_axis, lon_grid, lat_grid),
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
    return meta


def _crop_gsd_metres(pix_axis, scan_axis, lon_grid, lat_grid):
    """Common ground resolution of the staged crops (native OHRC 1:1 pixels)."""
    from src.preprocessing.geometry import _gsd_from_geometry

    m_scan, m_px = _gsd_from_geometry(scan_axis, pix_axis, lon_grid, lat_grid)
    return round(0.5 * (m_scan + m_px), 2)


def georeference_pair(ohrc_img_path, ohrc_csv_path, nac_img_path, out_dir,
                      crop_px=1024, prefix="pair1", equal_gsd=False,
                      nac_geom_csv=""):
    """Crop a pair to 1024x1024 same-ground-area; write PNGs + JSON.

    Outputs are named `{prefix}_src.png`, `{prefix}_ref.png` and
    `georef_{prefix}.json` so several pairs can be processed.

    Overlap search (two stages):
      1. primary sparse SIFT (find_pair_transform)       -- unchanged Phase 1 path
      2. boosted sparse fallback (find_pair_transform_robust) with
         proof-based acceptance (inlier dominance + NCC) when stage 1 finds
         <20 inliers.
    If both fail, a full candidate-statistics diagnostic is written to
    `georef_{prefix}_diagnostics.json` and RuntimeError is raised.

    equal_gsd=True self-calibrates the NAC working-set factor (no fixed
    `nac_fac` guess): the homography's scale tells the OHRC<->NAC working-set
    ground-scale ratio, so the NAC working set is re-sampled until both
    working sets share the same ground resolution. nac_geom_csv (NAC per-line
    SPICE geometry CSV) seeds the first guess from ground GSDs when given.
    """
    from src.preprocessing.geometry import _gsd_from_geometry

    os.makedirs(out_dir, exist_ok=True)

    ohrc = read_ohrc_raw(ohrc_img_path)
    ohrc_ws = ohrc[::OHRC_WS_FACTOR, ::OHRC_WS_FACTOR]  # (~9369, 1200)
    nac = read_nac_img(nac_img_path)
    pix_axis, scan_axis, lon_grid, lat_grid = read_ohrc_ground_grid(ohrc_csv_path)

    fac = NAC_WS_FACTOR
    ws_iters = None
    if equal_gsd:
        m_scan, m_px = _gsd_from_geometry(scan_axis, pix_axis,
                                          lon_grid, lat_grid)
        gsd_ohrc_ws = OHRC_WS_FACTOR * 0.5 * (m_scan + m_px)
        if nac_geom_csv:
            lo, la, ln, sm = _nac_geom_rows(nac_geom_csv)
            fac0 = _nac_gsd_seed(lo, la, ln, sm, gsd_ohrc_ws)
        else:
            fac0 = NAC_WS_FACTOR
        fac = fac0
        ws_iters = []
        for _ in range(4):
            t = find_pair_transform(ohrc_ws, nac, nac_fac=fac)
            if t is None:
                ws_iters.append(dict(fac=fac, scale=None))
                break
            scale = ws_scale(t["H"])
            f_new = refine_ws_factor(fac, scale)
            ws_iters.append(dict(fac=fac, scale=round(scale, 4), next_fac=f_new))
            if f_new == fac or abs(f_new - fac) <= 1:
                fac = f_new
                break
            fac = f_new
        else:
            ws_iters.append(dict(fac=fac, scale=None, not_converged=True))
            fac = refine_ws_factor(fac, ws_iters[-2]["scale"])

    t = find_pair_transform(ohrc_ws, nac, nac_fac=fac)
    method = "sparse-sift"
    diag = {}
    if t is None or t["inliers"] < 20:
        diag["primary"] = candidate_stats(ohrc_ws, nac, nac_fac=fac)
        t2, robust_stats = find_pair_transform_robust(ohrc_ws, nac, nac_fac=fac)
        if t2 is None:
            diag["robust"] = robust_stats
            diag["robust_rejected"] = (
                "no candidate passed acceptance: inliers>=20, ratio>=0.25, "
                "winning-ratio >= 2x second, ncc>=0.15, sane bbox"
            )
            with open(os.path.join(out_dir, f"georef_{prefix}_diagnostics.json"), "w") as fh:
                json.dump(diag, fh, indent=2)
            n = (t or {}).get('inliers')
            raise RuntimeError(
                f"georeference: no reliable OHRC<->NAC overlap after primary "
                f"(inliers={n}) and boosted fallback. Diagnostics written to "
                f"georef_{prefix}_diagnostics.json (all orientations, ratios, NCC)."
            )
        t = t2
        method = t.get("method", "sparse-robust")

    meta = _set_georeference_pair(ohrc, ohrc_ws, nac, pix_axis, scan_axis,
                                  lon_grid, lat_grid, t, out_dir, crop_px, prefix)
    meta["georef_method"] = method
    if equal_gsd:
        meta["equal_gsd"] = True
        meta["ws_factor_iters"] = ws_iters
        meta["nac_ws_factor_final"] = fac
    with open(os.path.join(out_dir, f"georef_{prefix}.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def _nac_geom_rows(csv_path):
    """Read the NAC per-line SPICE geometry CSV back into raw column arrays."""
    with open(csv_path, newline="") as fh:
        rr = [[float(v) for v in r.split()] for r in fh if r.strip()]
    a = np.asarray(rr, np.float64)
    return a[:, 0], a[:, 1], a[:, 3], a[:, 4]


def _nac_gsd_seed(lo, la, ln, sm, gsd_ohrc_ws):
    """Seed NAC working-set factor so GSD ~= OHRC working-set GSD.

    Uses the median across-track (sample) and along-track (line) ground
    spacing within the NAC geometry grid as the per-pixel meters.
    """
    lo_r = np.asarray(lo, np.float64) % 360.0
    lat_r = np.asarray(la, np.float64)
    if lo_r.size < 2 or lat_r.size < 2:
        return NAC_WS_FACTOR
    lat_m = 111000.0
    lon_m = 111000.0
    d = np.sqrt((np.diff(lo_r) * lon_m * np.cos(np.deg2rad(lat_r[:-1]))) ** 2
                + (np.diff(lat_r) * lat_m) ** 2)
    order = np.argsort(np.abs(ln[:-1] - ln[1:]))
    d_line = np.median(d[order[:100]]) if d.size else np.nan
    order_s = np.argsort(np.abs(sm[:-1] - sm[1:]))
    d_samp = np.median(d[order_s[:100]]) if d.size else np.nan
    gsd_nac = np.nanmedian([d_line, d_samp])
    if not np.isfinite(gsd_nac) or gsd_nac <= 0:
        return NAC_WS_FACTOR
    return int(min(max(int(round(gsd_ohrc_ws / gsd_nac)), 1), 256))


if __name__ == "__main__":
    import sys

    m = georeference_pair(
        sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],
    )
    print(json.dumps(m, indent=2))