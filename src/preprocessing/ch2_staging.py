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

from src.preprocessing.tmc import (
    read_ch2_raw,
    read_ground_grid,
    _gsd_m,
)

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


def _inverse_maps(pix_axis, scan_axis, lon_grid, lat_grid):
    """Return (lat->scan, lon->pixel) monotonic inverse interpolators."""
    lat_scan = np.median(lat_grid, axis=1)      # lat as function of scan index
    lon_pix = np.median(lon_grid, axis=0)       # lon as function of pixel index
    f_scan = RegularGridInterpolator((lat_scan,), scan_axis.astype(float),
                                     bounds_error=False, fill_value=None)
    f_pix = RegularGridInterpolator((lon_pix,), pix_axis.astype(float),
                                    bounds_error=False, fill_value=None)
    return f_scan, f_pix


RECT_COLOR = (0, 0, 255)  # BGR red


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
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if box:
        x0, y0, x1, y1 = box
        h, w = bgr.shape[:2]
        inside = np.zeros((h, w), bool)
        inside[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
        dimmed = bgr[~inside]
        bgr[~inside] = (dimmed * 0.35).astype(np.uint8)
        t = max(4, int(min(h, w) / 30))
        cv2.rectangle(bgr, (x0, y0), (x1, y1), RECT_COLOR, t)
        text = "OVERLAP SWATH"
        fs = max(0.7, min(h, w) / 420)
        th = max(2, int(fs * 3.2))
        (tw, thb), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(2, min(x0, max(2, w - tw - 8)))
        ty = max(thb + 6, y0 - thb - 10)
        cv2.rectangle(bgr, (tx - 8, ty - thb - 8), (tx + tw + 8, ty + 6),
                      (0, 0, 0), -1)
        cv2.putText(bgr, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs,
                    (255, 255, 255), th, cv2.LINE_AA)
        cv2.putText(bgr, text, (tx + max(1, th // 6), ty), cv2.FONT_HERSHEY_SIMPLEX,
                    fs, RECT_COLOR, max(1, th // 5), cv2.LINE_AA)
    return bgr


def _outline_registered(gray, tag="REGISTERED"):
    """Red outline + 'REGISTERED' tag around a registered product."""
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    h, w = bgr.shape[:2]
    t = max(4, int(min(h, w) / 30))
    cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), RECT_COLOR, t)
    glow_t = max(1, t // 3)
    cv2.rectangle(bgr, (t + glow_t, t + glow_t),
                  (w - 1 - t - glow_t, h - 1 - t - glow_t), RECT_COLOR, glow_t)
    fs = max(0.7, min(h, w) / 420)
    th = max(2, int(fs * 3.2))
    (tw, thb), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    pad = max(6, t // 2)
    cv2.rectangle(bgr, (pad - 2, pad - 2),
                  (pad + tw + 10, pad + thb + 6), (0, 0, 0), -1)
    cv2.putText(bgr, tag, (pad + 8, pad + thb + 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), th, cv2.LINE_AA)
    cv2.putText(bgr, tag, (pad + 8 + max(1, th // 6), pad + thb + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, RECT_COLOR, max(1, th // 5),
                cv2.LINE_AA)
    return bgr


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

    s_map_scan, s_map_pix = _inverse_maps(spix, sscan, slon, slat)
    r_map_scan, r_map_pix = _inverse_maps(rpix, rscan, rlon, rlat)

    s_scan = s_map_scan(LAT.ravel()[:, None]).ravel()
    s_pix = s_map_pix(LON.ravel()[:, None]).ravel()
    r_scan = r_map_scan(LAT.ravel()[:, None]).ravel()
    r_pix = r_map_pix(LON.ravel()[:, None]).ravel()

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


def _dump(report, out_dir, prefix):
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
