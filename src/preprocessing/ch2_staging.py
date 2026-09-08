"""Phase 9 (fix) — Robust Chandrayaan-2 sensor-to-sensor registration.

Dark, low-sun and polar lunar regions (e.g. OHRC-2026, mean ~5/255) defeat
content-based matching: enhancement lifts pixel values but cannot create spatial
detail that is not there. This module provides the "proper pipeline" for those
cases:

  1. **Enhance** both frames (percentile stretch + CLAHE) so *moderately* dark
     pairs become matchable.
  2. **Content** attempt: cross-modal / SIFT match on the enhanced pair.
  3. **Geometry fallback**: if content fails AND the two Chandrayaan-2 geometry
     CSVs (each ``(pixel, scan) -> (lon, lat)``) overlap on the ground, register
     both frames onto a **common ground grid** (equal-GSD), producing genuinely
     aligned crops and ``verdict: geometry_registered`` with the honest note
     "content correspondence not verifiable".
  4. **No overlap** / no geometry -> ``verdict: not_registered`` (never a
     fabricated model).

The geometry path uses each sensor's own ISRO geometry CSV directly (no SPICE),
so it works for any combination of OHRC/TMC (and future CH2 pushbroom sensors).
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import map_coordinates
from scipy.spatial import cKDTree

from src.preprocessing.tmc import (
    read_ch2_raw,
    read_ground_grid,
    _gsd_m,
)
from src.visuals import frame_img, highlight_box

ENHANCE_PERCENTILES = (1, 99)
CLAHE_CLIP = 4.0
CLAHE_TILE = 8


# --------------------------------------------------------------------------- #
# enhancement
# --------------------------------------------------------------------------- #
def percentile_stretch(img, lo=ENHANCE_PERCENTILES[0], hi=ENHANCE_PERCENTILES[1]):
    """Robust percentile contrast stretch of a (possibly near-black) image."""
    a = img.astype(np.float32)
    p_lo, p_hi = np.percentile(a, [lo, hi])
    if p_hi - p_lo < 1e-6:
        return np.zeros_like(img)
    out = np.clip((a - p_lo) / (p_hi - p_lo) * 255.0, 0, 255)
    return out.astype(np.uint8)


def enhance_dark(img, clip_limit=CLAHE_CLIP, tile_grid=CLAHE_TILE):
    """Lift a low-dynamic-range lunar frame: stretch then local equalisation."""
    s = percentile_stretch(img)
    c = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    return c.apply(s)


def low_contrast_score(img):
    """0..1 heuristic: 1 = essentially flat/featureless (dark/polar), 0 = rich.

    Combines the percentile contrast span (relative to full scale) and the
    fraction of pixels carrying gradient (structure). A shadowed moon frame (e.g.
    OHRC-2026, span ~18/255, few strong gradients) scores high; a sunlit cratered
    scene scores low.
    """
    a = img.astype(np.float32)
    p_lo, p_hi = np.percentile(a, [1, 99])
    span = float(p_hi - p_lo) / 255.0                 # 0 (flat) .. ~1 (full scale)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    strong = float((grad > 2.0).mean())               # fraction with real edges
    span_term = float(np.clip(1.0 - 3.0 * span, 0, 1))
    struct_term = float(np.clip(1.0 - 8.0 * strong, 0, 1))
    score = 0.5 * span_term + 0.5 * struct_term
    return float(np.clip(score, 0, 1))


# --------------------------------------------------------------------------- #
# ground staging (geometry registration / equal-GSD pair builder)
# --------------------------------------------------------------------------- #
def overlap_box(gridA_lon, gridA_lat, gridB_lon, gridB_lat):
    """Intersection of two footprint boxes -> (lon0, lon1, lat0, lat1) or None."""
    lon0 = max(float(gridA_lon.min()), float(gridB_lon.min()))
    lon1 = min(float(gridA_lon.max()), float(gridB_lon.max()))
    lat0 = max(float(gridA_lat.min()), float(gridB_lat.min()))
    lat1 = min(float(gridA_lat.max()), float(gridB_lat.max()))
    if lon1 <= lon0 or lat1 <= lat0:
        return None
    return (lon0, lon1, lat0, lat1)


class GroundGridInverse:
    """2-D (lon, lat) -> (scan, pixel) inverse for an ISRO-style grid.

    A separable 1-D model (two independent monotonic maps lat->scan,
    lon->pixel, previously ``_inverse_maps``) is exact only when the ground
    grid is a rectangle. Chandrayaan strips sweep (lat is not a function of
    scan alone), so the separable model can be off by thousands of native
    pixels — the TMC-2026 -> NAC content failure. This class instead seeds
    each query with a nearest-grid-point and refines with a full 2x2
    Gauss-Newton step, exactly like ``NacInverse`` but for ISRO (scan, pixel)
    grids.
    """

    def __init__(self, pix_axis, scan_axis, lon_grid, lat_grid):
        self._scan_ax = np.asarray(scan_axis, np.float64)
        self._pix_ax = np.asarray(pix_axis, np.float64)
        self.lon_f = RegularGridInterpolator(
            (self._scan_ax, self._pix_ax), lon_grid,
            method="linear", bounds_error=False, fill_value=np.nan)
        self.lat_f = RegularGridInterpolator(
            (self._scan_ax, self._pix_ax), lat_grid,
            method="linear", bounds_error=False, fill_value=np.nan)
        SL, SP = np.meshgrid(self._scan_ax, self._pix_ax, indexing="ij")
        self._lonref = float(np.nanmedian(lon_grid))
        rel_lon = (lon_grid - self._lonref + 180.0) % 360.0 - 180.0
        self._tree_pts = np.column_stack([rel_lon.ravel(), lat_grid.ravel()])
        self._seed_scan = SL.ravel()
        self._seed_pix = SP.ravel()

    def forward(self, scan, pixel):
        """(scan, pixel) -> (lon [roughly near _lonref], lat)."""
        q = np.column_stack([np.asarray(scan).ravel(), np.asarray(pixel).ravel()])
        lo = np.asarray(self.lon_f(q), np.float64).ravel()
        la = np.asarray(self.lat_f(q), np.float64).ravel()
        lo = (lo - self._lonref + 180.0) % 360.0 - 180.0 + self._lonref
        if np.ndim(scan) == 0:
            return float(lo[0]), float(la[0])
        return lo.reshape(np.shape(scan)), la.reshape(np.shape(scan))

    def apply(self, lon, lat):
        """(lon, lat) -> (scan, pixel) arrays (same shape as inputs).

        For each query a nearest-grid-point seed is refined with a full 2x2
        Gauss-Newton step. Swept (folded) strips contain regions where the
        forward map is locally non-invertible and the Newton iterate can
        diverge; in that case the iterate with the lowest residual — or the
        seed itself — is returned so the output is always a finite,
        best-available prediction.
        """
        lon = np.asarray(lon, np.float64)
        lat = np.asarray(lat, np.float64)
        rel = (lon - self._lonref + 180.0) % 360.0 - 180.0
        d, idx = cKDTree(self._tree_pts).query(
            np.column_stack([rel.ravel(), lat.ravel()]))
        scan = self._seed_scan[idx].reshape(lon.shape).astype(np.float64)
        pix = self._seed_pix[idx].reshape(lon.shape).astype(np.float64)
        best_s, best_p = scan.copy(), pix.copy()
        best_res = np.full(scan.shape, np.inf)
        lo_v, la_v = self.forward(scan, pix)
        for _ in range(12):
            fin = np.isfinite(lo_v) & np.isfinite(la_v)
            rl = (lo_v - self._lonref + 180.0) % 360.0 - 180.0
            res = (np.abs(rl - rel) + np.abs(la_v - lat))
            take = fin & (res < best_res)
            best_s[take] = scan[take]
            best_p[take] = pix[take]
            best_res[take] = res[take]
            e = 1e-3
            loE, laE = self.forward(scan + e, pix)
            loT, laT = self.forward(scan, pix + e)
            J00 = (loE - lo_v) / e
            J01 = (loT - lo_v) / e
            J10 = (laE - la_v) / e
            J11 = (laT - la_v) / e
            det = J00 * J11 - J01 * J10
            dlon = rel - rl
            dlat = lat - la_v
            nxt_s = scan + (J11 * dlon - J01 * dlat) / det
            nxt_p = pix + (-J10 * dlon + J00 * dlat) / det
            bad = ~np.isfinite(det) | (np.abs(det) < 1e-15)
            nxt_s = np.where(bad, scan, nxt_s)
            nxt_p = np.where(bad, pix, nxt_p)
            scan, pix = nxt_s, nxt_p
            lo_v, la_v = self.forward(scan, pix)
        return best_s, best_p


def _make_original_preview(ll, scan_vals, pix_vals, max_w=640, max_h=1600,
                           max_aspect=3.5):
    """Build a display-ready 'before' crop: a context window around the native
    overlap region the geometry maps onto, bin-downsampled so it fits a wide
    panel (aspect <= ``max_aspect`` — strips are otherwise 10:1+ slivers that
    render off-screen in the UI).

    Returns ``(preview_gray_uint8, box)`` where ``box = (x0, y0, x1, y1)`` is the
    inner overlap window in preview-pixel coordinates (so callers can draw a
    highlight rectangle). Returns ``(None, None)`` when no finite window exists."""
    finite = np.isfinite(scan_vals) & np.isfinite(pix_vals)
    scan, pix = scan_vals[finite], pix_vals[finite]
    if scan.size == 0 or pix.size == 0 or ll is None:
        return None, None
    rows, cols = ll.shape
    s0 = max(int(np.floor(scan.min())) - 4, 0)
    s1 = min(int(np.ceil(scan.max())) + 4, rows - 1)
    p0 = max(int(np.floor(pix.min())) - 4, 0)
    p1 = min(int(np.ceil(pix.max())) + 4, cols - 1)
    if s1 <= s0 or p1 <= p0:
        return None, None
    span_s, span_p = s1 - s0, p1 - p0
    c0 = max(0, s0 - int(span_s * 0.4)); c1 = min(rows, s1 + int(span_s * 0.4))
    d0 = max(0, p0 - int(span_p * 0.4)); d1 = min(cols, p1 + int(span_p * 0.4))
    crop = np.ascontiguousarray(ll[c0:c1, d0:d1]).astype(np.float32)
    fac_s, fac_p = 1, 1
    if crop.shape[1] > max_w:
        fac_p = max(int(np.ceil(crop.shape[1] / max_w)), 1)
    if crop.shape[0] > max_h:
        fac_s = max(int(np.ceil(crop.shape[0] / max_h)), 1)
    while (crop.shape[0] / fac_s) / max(crop.shape[1] / fac_p, 1) > max_aspect:
        fac_s += 1
    hs, ws = crop.shape[0] // fac_s, crop.shape[1] // fac_p
    crop = crop[:hs * fac_s, :ws * fac_p].reshape(hs, fac_s, ws, fac_p).mean(axis=(1, 3))
    prev = (percentile_stretch(crop) if crop.max() > crop.min()
            else np.zeros_like(crop)).astype(np.uint8)
    box = (int((p0 - d0) / fac_p), int((s0 - c0) / fac_s),
           int((p1 - d0) / fac_p), int((s1 - c0) / fac_s))
    return prev, box


def _draw_highlight_box(gray, box):
    """Red rectangle + 'OVERLAP SWATH' label around the extracted window; the
    context outside the box is dimmed hard so the change cannot be missed."""
    return highlight_box(gray, box, tag="OVERLAP SWATH")


def _outline_registered(gray, tag="REGISTERED"):
    """Red outline + ``tag`` around a registered product (shared visuals)."""
    return frame_img(gray, tag=tag)


def stage_ch2_ground_pair(src_img, src_geom, ref_img, ref_geom, *,
                          src_width=None, ref_width=None, gsd=None,
                          n_along=1400, out_dir=None, prefix="ch2_pair"):
    """Stage two CH2 pushbroom sensors onto a common ground grid (equal GSD).

    Uses each sensor's ISRO geometry CSV to inverse-map every point of a shared
    lon/lat grid into that sensor's native (scan, pixel) space and bilinearly
    resamples both raw frames there. Returns two aligned uint8 crops on the same
    ground grid, i.e. a **geometry registration**.

    The destination ground GSD defaults to the coarser of the two sensors so the
    fine sensor is sensibly downsampled and neither is upsampled.

    Returns a dict: {src, ref (uint8 arrays), gsd_m, lon0, lon1, lat0, lat1,
    n_along, n_across, overlap: bool, note}.
    """
    spix, sscan, slon, slat = read_ground_grid(src_geom)
    rpix, rscan, rlon, rlat = read_ground_grid(ref_geom)
    box = overlap_box(slon, slat, rlon, rlat)
    if box is None:
        return {"overlap": False, "note": "footprints do not overlap on the ground",
                "src": None, "ref": None}

    gsd_src = float(_gsd_m(spix, sscan, slon, slat))
    gsd_ref = float(_gsd_m(rpix, rscan, rlon, rlat))
    if gsd is None:
        gsd = max(gsd_src, gsd_ref) if np.isfinite(gsd_src) and np.isfinite(gsd_ref) else 21.7

    lon0, lon1, lat0, lat1 = box
    mlat = 0.5 * (lat0 + lat1)
    lat_m = 111000.0
    lon_m = 111000.0 * np.cos(np.deg2rad(mlat))

    d_lat = (lat1 - lat0) * lat_m / n_along
    n_across = max(64, int(round((lon1 - lon0) * lon_m / d_lat)))

    lats = np.linspace(lat0, lat1, n_along)
    lons = np.linspace(lon0, lon1, n_across)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    s_inv = GroundGridInverse(spix, sscan, slon, slat)
    r_inv = GroundGridInverse(rpix, rscan, rlon, rlat)

    s_scan, s_pix = s_inv.apply(LON.ravel(), LAT.ravel())
    r_scan, r_pix = r_inv.apply(LON.ravel(), LAT.ravel())

    s_ll = read_ch2_raw(src_img, src_geom, width=src_width, memmap=True)
    r_ll = read_ch2_raw(ref_img, ref_geom, width=ref_width, memmap=True)
    finite_s = np.isfinite(s_scan) & np.isfinite(s_pix)
    finite_r = np.isfinite(r_scan) & np.isfinite(r_pix)
    swath_s = {"row_min": int(s_scan[finite_s].min()), "row_max": int(s_scan[finite_s].max()),
               "col_min": int(s_pix[finite_s].min()), "col_max": int(s_pix[finite_s].max())}
    swath_r = {"row_min": int(r_scan[finite_r].min()), "row_max": int(r_scan[finite_r].max()),
               "col_min": int(r_pix[finite_r].min()), "col_max": int(r_pix[finite_r].max())}

    def sample(ll, scan, pix):
        ok = np.isfinite(scan) & np.isfinite(pix)
        scan = np.where(ok, scan, 0); pix = np.where(ok, pix, 0)
        out = map_coordinates(ll, np.vstack([scan, pix]), order=1,
                              mode="constant", cval=0)
        out[~ok] = 0
        return out.reshape(LAT.shape).astype(np.uint8)

    src_g = sample(s_ll, s_scan, s_pix)
    ref_g = sample(r_ll, r_scan, r_pix)

    result = {
        "overlap": True, "src": src_g, "ref": ref_g,
        "gsd_m": float(gsd), "lon0": lon0, "lon1": lon1,
        "lat0": lat0, "lat1": lat1, "n_along": n_along,
        "n_across": n_across,
        "native_src": {"rows": int(s_ll.shape[0]), "cols": int(s_ll.shape[1])},
        "native_ref": {"rows": int(r_ll.shape[0]), "cols": int(r_ll.shape[1])},
        "swath_src": swath_s, "swath_ref": swath_r,
        "note": f"ground staged at {gsd:.2f} m/px over "
                f"[{lon0:.4f},{lon1:.4f}]x[{lat0:.4f},{lat1:.4f}]",
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_src.png"), src_g)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_ref.png"), ref_g)
        before_src = os.path.join(out_dir, f"{prefix}_before_src.png")
        before_ref = os.path.join(out_dir, f"{prefix}_before_ref.png")
        after_src = os.path.join(out_dir, f"{prefix}_after_src.png")
        after_ref = os.path.join(out_dir, f"{prefix}_after_ref.png")
        prev_src, box_s = _make_original_preview(s_ll, s_scan, s_pix)
        prev_ref, box_r = _make_original_preview(r_ll, r_scan, r_pix)
        if prev_src is not None:
            cv2.imwrite(before_src, _draw_highlight_box(prev_src, box_s))
        if prev_ref is not None:
            cv2.imwrite(before_ref, _draw_highlight_box(prev_ref, box_r))
        cv2.imwrite(after_src, _outline_registered(src_g))
        cv2.imwrite(after_ref, _outline_registered(ref_g))
        result["original_src"] = before_src if prev_src is not None else None
        result["original_ref"] = before_ref if prev_ref is not None else None
        result["after_src"] = after_src
        result["after_ref"] = after_ref
        diff = np.abs(src_g.astype(np.int16) - ref_g.astype(np.int16)).astype(np.uint8)
        diff = percentile_stretch(diff) if diff.max() > diff.min() else diff
        result["change_map"] = os.path.join(out_dir, f"{prefix}_change.png")
        cv2.imwrite(result["change_map"], cv2.applyColorMap(diff, cv2.COLORMAP_TURBO))
        result["montage"] = os.path.join(out_dir, f"{prefix}_montage.png")
        cv2.imwrite(result["montage"], np.hstack([
            cv2.imread(result["after_src"]),
            cv2.imread(result["after_ref"]),
            cv2.imread(result["change_map"]),
        ]))
        with open(os.path.join(out_dir, f"{prefix}_stage.json"), "w") as fh:
            json.dump({k: v for k, v in result.items()
                       if not isinstance(v, np.ndarray)}, fh, indent=2)
    return result


# --------------------------------------------------------------------------- #
# TMC / CH2 -> LRO NAC staging (PS 26166: "lunar reference images")
# --------------------------------------------------------------------------- #
def _nac_gsd_m(line_axis, samp_axis, lon_g, lat_g):
    """Mean NAC ground sample distance (m per line, m per sample)."""
    def _step(axis):
        a = np.asarray(axis, np.float64)
        if a.size < 2:
            return 1.0
        d = np.abs(np.diff(a))
        d = d[d > 0]
        return float(np.median(d)) if d.size else 1.0
    lat_m, lon_m = 111000.0, 111000.0
    lon_r = np.deg2rad(lat_g)
    d_samp = np.hypot(np.diff(lon_g, axis=1) * lon_m * np.cos(lon_r[:, :-1]),
                      np.diff(lat_g, axis=1) * lat_m) / _step(samp_axis)
    d_line = np.hypot(np.diff(lon_g, axis=0) * lon_m * np.cos(lon_r[:-1, :]),
                      np.diff(lat_g, axis=0) * lat_m) / _step(line_axis)
    vals = np.concatenate([d_samp[np.isfinite(d_samp)],
                           d_line[np.isfinite(d_line)]])
    return float(np.median(vals)) if vals.size else np.nan


def _sample_subgrid(arr, row, col, pad=8):
    """Bilinear-sample ``arr`` at float (row, col) positions, 0 outside.

    Uses ``scipy.ndimage.map_coordinates`` (no 32k row limit, unlike
    ``cv2.remap``), so multi-GB / >32k-row rasters (TMC memmaps, LRO NAC strips)
    are sampled at a bounded cost without loading full RAM.
    """
    row = np.asarray(row, np.float64)
    col = np.asarray(col, np.float64)
    fin = np.isfinite(row) & np.isfinite(col)
    if not fin.any():
        return np.zeros(row.shape, np.uint8)
    rr = np.where(fin, row, 0)
    cc = np.where(fin, col, 0)
    a = np.asarray(arr)  # uint8 (TMC memmap / as_u8(NAC)); NaN already 0
    out = map_coordinates(a, np.vstack([rr.ravel(), cc.ravel()]), order=1,
                          mode="constant", cval=0.0)
    out = out.reshape(row.shape)
    out[~fin] = 0.0
    return np.clip(out, 0, 255).astype(np.uint8)


def stage_ch2_nac_pair(src_img, src_geom, nac_img, nac_geom_csv, *,
                       src_width=None, gsd=None, n_along=1400, out_dir=None,
                       prefix="ch2_nac"):
    """Stage a CH2 pushbroom sensor (TMC/OHRC source) onto a common ground grid
    with an **LRO NAC lunar reference** (SPICE per-line geometry).

    The source is inverse-mapped from its ISRO geometry CSV; the NAC is
    inverse-mapped with the ``NacInverse`` SPICE Newton solve. Both raw frames
    are resampled onto the shared equal-GSD lon/lat grid -> a genuine
    source→lunar-reference geometry registration without any feature matching.

    Returns the same dict contract as ``stage_ch2_ground_pair``.
    """
    from src.preprocessing.geometry import load_nac_geom, NacInverse, as_u8
    from src.preprocessing.georeference import read_nac_img

    spix, sscan, slon, slat = read_ground_grid(src_geom)
    lon, lat, line, samp = load_nac_geom(nac_geom_csv)
    nac = read_nac_img(nac_img)
    # wrap NAC lon into [0, 360) to compare with the ISRO grid convention
    nlon = lon % 360.0
    box = overlap_box(slon, slat, nlon, lat)
    if box is None:
        return {"overlap": False, "note": "footprints do not overlap on the ground",
                "src": None, "ref": None}

    gsd_src = float(_gsd_m(spix, sscan, slon, slat))
    uline = np.unique(line); usamp = np.unique(samp)
    lon_g = nlon.reshape(len(uline), len(usamp))
    lat_g = lat.reshape(len(uline), len(usamp))
    gsd_ref = float(_nac_gsd_m(uline, usamp, lon_g, lat_g))
    if gsd is None:
        gsd = max(gsd_src, gsd_ref) if np.isfinite(gsd_src) and np.isfinite(gsd_ref) else 21.7

    lon0, lon1, lat0, lat1 = box
    mlat = 0.5 * (lat0 + lat1)
    lat_m = 111000.0
    lon_m = 111000.0 * np.cos(np.deg2rad(mlat))
    d_lat = (lat1 - lat0) * lat_m / n_along
    n_across = max(64, int(round((lon1 - lon0) * lon_m / d_lat)))
    lats = np.linspace(lat0, lat1, n_along)
    lons = np.linspace(lon0, lon1, n_across)
    LAT, LON = np.meshgrid(lats, lons, indexing="ij")

    s_inv = GroundGridInverse(spix, sscan, slon, slat)
    src_scan, src_pix = s_inv.apply(LON.ravel(), LAT.ravel())
    inv = NacInverse(lon, lat, line, samp)
    nac_col, nac_row = inv.apply(LON.ravel(), LAT.ravel())
    nac_col = nac_col.reshape(LAT.shape); nac_row = nac_row.reshape(LAT.shape)

    s_ll = read_ch2_raw(src_img, src_geom, width=src_width, memmap=True)
    finite_s = np.isfinite(src_scan) & np.isfinite(src_pix)
    swath_s = {"row_min": int(src_scan[finite_s].min()),
               "row_max": int(src_scan[finite_s].max()),
               "col_min": int(src_pix[finite_s].min()),
               "col_max": int(src_pix[finite_s].max())}
    src_g = _sample_subgrid(s_ll, src_scan.reshape(LAT.shape), src_pix.reshape(LAT.shape))

    nac_u8 = as_u8(nac)
    finite_r = np.isfinite(nac_row) & np.isfinite(nac_col)
    swath_r = {"row_min": int(nac_row[finite_r].min()),
               "row_max": int(nac_row[finite_r].max()),
               "col_min": int(nac_col[finite_r].min()),
               "col_max": int(nac_col[finite_r].max())}
    ref_g = _sample_subgrid(nac_u8, nac_row, nac_col)

    result = {
        "overlap": True, "src": src_g, "ref": ref_g,
        "gsd_m": float(gsd), "lon0": lon0, "lon1": lon1,
        "lat0": lat0, "lat1": lat1, "n_along": n_along,
        "n_across": n_across,
        "native_src": {"rows": int(s_ll.shape[0]), "cols": int(s_ll.shape[1])},
        "native_ref": {"rows": int(nac.shape[0]), "cols": int(nac.shape[1])},
        "swath_src": swath_s, "swath_ref": swath_r,
        "reference": "LRO NAC",
        "note": f"ground staged at {gsd:.2f} m/px over "
                f"[{lon0:.4f},{lon1:.4f}]x[{lat0:.4f},{lat1:.4f}] vs LRO NAC",
    }
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_src.png"), src_g)
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_ref.png"), ref_g)
        prev_src, box_s = _make_original_preview(s_ll, src_scan, src_pix)
        prev_ref, box_r = _make_original_preview(nac_u8, nac_row, nac_col)
        if prev_src is not None:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_before_src.png"),
                        _draw_highlight_box(prev_src, box_s))
        if prev_ref is not None:
            cv2.imwrite(os.path.join(out_dir, f"{prefix}_before_ref.png"),
                        _draw_highlight_box(prev_ref, box_r))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_after_src.png"),
                    _outline_registered(src_g))
        cv2.imwrite(os.path.join(out_dir, f"{prefix}_after_ref.png"),
                    _outline_registered(ref_g))
        result["original_src"] = (os.path.join(out_dir, f"{prefix}_before_src.png")
                                  if prev_src is not None else None)
        result["original_ref"] = (os.path.join(out_dir, f"{prefix}_before_ref.png")
                                  if prev_ref is not None else None)
        result["after_src"] = os.path.join(out_dir, f"{prefix}_after_src.png")
        result["after_ref"] = os.path.join(out_dir, f"{prefix}_after_ref.png")
        diff = np.abs(src_g.astype(np.int16) - ref_g.astype(np.int16)).astype(np.uint8)
        diff = percentile_stretch(diff) if diff.max() > diff.min() else diff
        result["change_map"] = os.path.join(out_dir, f"{prefix}_change.png")
        cv2.imwrite(result["change_map"], cv2.applyColorMap(diff, cv2.COLORMAP_TURBO))
        result["montage"] = os.path.join(out_dir, f"{prefix}_montage.png")
        cv2.imwrite(result["montage"], np.hstack([
            cv2.imread(result["after_src"]),
            cv2.imread(result["after_ref"]),
            cv2.imread(result["change_map"]),
        ]))
        with open(os.path.join(out_dir, f"{prefix}_stage.json"), "w") as fh:
            json.dump({k: v for k, v in result.items()
                       if not isinstance(v, np.ndarray)}, fh, indent=2)
    return result


def register_ch2_to_nac(src_img, src_geom, nac_img, nac_geom_csv, *, out_dir,
                        src_width=None, prefix="ch2_nac", n_along=1400,
                        min_inliers=10, min_ratio=0.02, nfeatures=10000,
                        contrast_threshold=0.02, verbose=True, **match_kw):
    """Register a CH2 source (TMC/OHRC) against the **LRO NAC lunar reference**.

    Content-first (enhanced cross-modal pair, sub-pixel RMSE + inlier
    correspondences CSV when it succeeds); SPICE + ISRO ground-grid geometry
    registration as the honest fallback for dark/low-sun/polar pairs (aligned
    products + the explicit "content NOT verifiable" note). Never fabricates a
    model.
    """
    from src.detection.cross_modal import cross_modal_register

    os.makedirs(out_dir, exist_ok=True)
    report = {
        "pair": {"src": os.path.basename(src_img),
                 "ref": os.path.basename(nac_img)},
        "reference": "LRO NAC",
        "method": None, "verdict": None, "rmse_px": None, "inliers": 0,
        "inlier_ratio": 0.0, "n_matches": 0, "homography": None,
        "gsd_m": None, "artifacts": {}, "notes": [],
    }

    spix, sscan, slon, slat = read_ground_grid(src_geom)
    from src.preprocessing.geometry import load_nac_geom
    nlon_arr, nlat_arr, _l, _s = load_nac_geom(nac_geom_csv)
    box = overlap_box(slon, slat, nlon_arr % 360.0, nlat_arr)
    report["georef"] = {
        "src": {"lon": [float(slon.min()), float(slon.max())],
                "lat": [float(slat.min()), float(slat.max())],
                "gsd_m": _gsd_m(spix, sscan, slon, slat)},
        "ref": {"lon": [float(nlon_arr.min()), float(nlon_arr.max())],
                "lat": [float(nlat_arr.min()), float(nlat_arr.max())],
                "gsd_m": None, "sensor": "LRO NAC"},
        "overlap": bool(box is not None),
    }
    if box is None:
        report.update(verdict="not_registered",
                      notes=["footprints do not overlap on the ground"])
        _dump(report, out_dir, prefix)
        return report

    st = stage_ch2_nac_pair(src_img, src_geom, nac_img, nac_geom_csv,
                            src_width=src_width, n_along=n_along,
                            out_dir=out_dir, prefix=prefix)
    if not st["overlap"]:
        report.update(verdict="not_registered", notes=["ground staging failed"])
        _dump(report, out_dir, prefix)
        return report
    report["gsd_m"] = st["gsd_m"]
    report["dimensions"] = {
        "native_src": st["native_src"], "native_ref": st["native_ref"],
        "swath_src": st["swath_src"], "swath_ref": st["swath_ref"],
        "grid_along": st["n_along"], "grid_across": st["n_across"],
        "gsd_m": st["gsd_m"],
    }
    src, ref = st["src"], st["ref"]
    src_e, ref_e = enhance_dark(src), enhance_dark(ref)
    union = (src_e > 0) & (ref_e > 0)
    if union.sum() > 100:
        a = src_e[union].astype(np.float64)
        b = ref_e[union].astype(np.float64)
        overlap_ncc = {"overlap_ncc": float(np.corrcoef(a, b)[0, 1]),
                       "overlap_px": int(union.sum())}
    else:
        overlap_ncc = {"overlap_ncc": None, "overlap_px": int(union.sum())}
    report["diagnostics"] = {
        "src_low_contrast": round(low_contrast_score(src), 3),
        "ref_low_contrast": round(low_contrast_score(ref), 3),
        "src_mean": float(src.mean()), "ref_mean": float(ref.mean()),
        **overlap_ncc,
    }
    for k in ("src", "ref", "original_src", "original_ref",
              "after_src", "after_ref", "change_map", "montage"):
        if st.get(k) is not None:
            report["artifacts"][k] = st[k]
    report["notes"].append(st["note"])

    res = cross_modal_register(src_e, ref_e, min_inliers=min_inliers,
                               min_ratio=min_ratio, nfeatures=nfeatures,
                               contrast_threshold=contrast_threshold,
                               verbose=False, **match_kw)
    report["n_matches"] = res["n_matches"] if res else 0
    if res is not None:
        report.update(
            method=f"content:{res['front']}",
            verdict="registered",
            inliers=int(res["inliers"]),
            inlier_ratio=float(res["inlier_ratio"]),
            rmse_px=float(res["rmse"]),
            homography=res["H"].tolist(),
        )
        report["notes"].append(
            f"content registration on enhanced CH2<->NAC pair; "
            f"front={res['front']}")
        _write_content_fig(report, src_e, ref_e, res, out_dir, prefix)
        _save_correspondences(report, res, out_dir, prefix)
    else:
        report.update(
            method="geometry",
            verdict="geometry_registered",
            notes=report["notes"]
            + ["content correspondence NOT verifiable; registered by ISRO CSV "
               "+ LRO NAC SPICE on a common ground grid"],
        )
    if verbose:
        print(f"[register_ch2_to_nac] verdict={report['verdict']} "
              f"method={report['method']} inliers={report['inliers']}")
    _dump(report, out_dir, prefix)
    return report


# --------------------------------------------------------------------------- #
# layered registration decision
# --------------------------------------------------------------------------- #
def register_ch2_pair(src_img, src_geom, ref_img, ref_geom, *, out_dir,
                      src_width=None, ref_width=None, prefix="ch2_pair",
                      n_along=1400, min_inliers=10, min_ratio=0.02,
                      nfeatures=10000, contrast_threshold=0.02, verbose=True,
                      **match_kw):
    """Content-first, geometry-fallback registration for two CH2 sensors.

    Returns a JSON report. Content path enhances both frames and runs the
    cross-modal matcher; a successful content fit yields ``registered`` with a
    homography + RMSE. When content fails, if the footprints overlap the pair is
    ground-staged (geometry registration) -> ``geometry_registered``; otherwise
    ``not_registered``. Never fabricates a model.
    """
    from src.detection.cross_modal import cross_modal_register
    from src.outlier_rejection.ransac import find_homography_ransac
    from src.preprocessing.tmc import read_ground_grid  # noqa: F401 (re-export used below)

    os.makedirs(out_dir, exist_ok=True)
    report = {
        "pair": {"src": os.path.basename(src_img), "ref": os.path.basename(ref_img)},
        "method": None, "verdict": None, "rmse_px": None, "inliers": 0,
        "inlier_ratio": 0.0, "n_matches": 0, "homography": None,
        "gsd_m": None, "artifacts": {}, "notes": [],
    }

    # --- footprints + geometry ---
    spix, sscan, slon, slat = read_ground_grid(src_geom)
    rpix, rscan, rlon, rlat = read_ground_grid(ref_geom)
    box = overlap_box(slon, slat, rlon, rlat)
    report["georef"] = {
        "src": {"lon": [float(slon.min()), float(slon.max())],
                "lat": [float(slat.min()), float(slat.max())],
                "gsd_m": _gsd_m(spix, sscan, slon, slat)},
        "ref": {"lon": [float(rlon.min()), float(rlon.max())],
                "lat": [float(rlat.min()), float(rlat.max())],
                "gsd_m": _gsd_m(rpix, rscan, rlon, rlat)},
        "overlap": bool(box is not None),
    }
    if box is None:
        report.update(verdict="not_registered",
                      notes=["footprints do not overlap on the ground"])
        _dump(report, out_dir, prefix)
        return report

    # --- 1. enhance both frames on the shared ground grid ---
    st = stage_ch2_ground_pair(src_img, src_geom, ref_img, ref_geom,
                               src_width=src_width, ref_width=ref_width,
                               n_along=n_along, out_dir=out_dir, prefix=prefix)
    if not st["overlap"]:
        report.update(verdict="not_registered", notes=["ground staging failed"])
        _dump(report, out_dir, prefix)
        return report
    report["gsd_m"] = st["gsd_m"]
    report["dimensions"] = {
        "native_src": st["native_src"], "native_ref": st["native_ref"],
        "swath_src": st["swath_src"], "swath_ref": st["swath_ref"],
        "grid_along": st["n_along"], "grid_across": st["n_across"],
        "gsd_m": st["gsd_m"],
    }
    src, ref = st["src"], st["ref"]
    src_e, ref_e = enhance_dark(src), enhance_dark(ref)
    report["diagnostics"] = {
        "src_low_contrast": round(low_contrast_score(src), 3),
        "ref_low_contrast": round(low_contrast_score(ref), 3),
        "src_mean": float(src.mean()), "ref_mean": float(ref.mean()),
    }
    report["artifacts"]["src"] = os.path.join(out_dir, f"{prefix}_src.png")
    report["artifacts"]["ref"] = os.path.join(out_dir, f"{prefix}_ref.png")
    if st.get("original_src"):
        report["artifacts"]["original_src"] = st["original_src"]
    if st.get("original_ref"):
        report["artifacts"]["original_ref"] = st["original_ref"]
    if st.get("after_src"):
        report["artifacts"]["after_src"] = st["after_src"]
    if st.get("after_ref"):
        report["artifacts"]["after_ref"] = st["after_ref"]
    if st.get("change_map"):
        report["artifacts"]["change_map"] = st["change_map"]
    if st.get("montage"):
        report["artifacts"]["montage"] = st["montage"]
    report["notes"].append(st["note"])

    # --- 2. content attempt on enhanced pair ---
    res = cross_modal_register(src_e, ref_e, min_inliers=min_inliers,
                               min_ratio=min_ratio, nfeatures=nfeatures,
                               contrast_threshold=contrast_threshold,
                               verbose=False, **match_kw)
    report["n_matches"] = res["n_matches"] if res else 0
    if res is not None:
        report.update(
            method=f"content:{res['front']}",
            verdict="registered",
            inliers=int(res["inliers"]),
            inlier_ratio=float(res["inlier_ratio"]),
            rmse_px=float(res["rmse"]),
            homography=res["H"].tolist(),
        )
        report["notes"].append(
            f"content registration on enhanced pair; front={res['front']}")
        _write_content_fig(report, src_e, ref_e, res, out_dir, prefix)
        _save_correspondences(report, res, out_dir, prefix)
    else:
        # --- 3. geometry fallback (dark / polar / featureless) ---
        report.update(
            method="geometry",
            verdict="geometry_registered",
            notes=report["notes"]
            + ["content correspondence NOT verifiable (dark/low-sun/polar or "
               "featureless); registered by ISRO geometry on a common ground grid"],
        )
    if verbose:
        print(f"[register_ch2_pair] verdict={report['verdict']} "
              f"method={report['method']} inliers={report['inliers']}")
    _dump(report, out_dir, prefix)
    return report


def _write_content_fig(report, src, ref, res, out_dir, prefix):
    try:
        from src.evaluation.visualize import draw_matches
        kp1, kp2 = res.get("kp1"), res.get("kp2")
        if kp1 is None or kp2 is None:
            return
        pts1, pts2 = res["pts1"], res["pts2"]
        path = os.path.join(out_dir, f"{prefix}_matches.png")
        draw_matches(src, kp1, ref, kp2,
                     [cv2.DMatch(i, i, 0.0) for i in range(len(pts1))],
                     path, inlier_mask=(res["inl"] > 0) if "inl" in res else None)
        report["artifacts"]["matches"] = path
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"match figure failed: {exc}")


def _save_correspondences(report, res, out_dir, prefix):
    """Persist the inlier (verified) source<->reference point pairs as CSV — the
    'registered product with corresponding match points' deliverable."""
    try:
        p1, p2, inl = res.get("pts1"), res.get("pts2"), res.get("inl")
        if p1 is None or p2 is None or inl is None or int(np.count_nonzero(inl)) == 0:
            return
        p1i, p2i = p1[inl > 0], p2[inl > 0]
        path = os.path.join(out_dir, f"{prefix}_inliers.csv")
        with open(path, "w", newline="") as fh:
            fh.write("src_x,src_y,ref_x,ref_y\n")
            for (x1, y1), (x2, y2) in zip(p1i, p2i):
                fh.write(f"{float(x1):.4f},{float(y1):.4f},"
                         f"{float(x2):.4f},{float(y2):.4f}\n")
        report["artifacts"]["correspondences"] = path
        report["inlier_points_saved"] = int(len(p1i))
    except Exception as exc:  # noqa: BLE001
        report["notes"].append(f"correspondence save failed: {exc}")


def _dump(report, out_dir, prefix):
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
