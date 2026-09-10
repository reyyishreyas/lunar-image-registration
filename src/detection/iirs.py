"""Phase 9 — Chandrayaan-2 IIRS support (multi-spectral source, PS 26166).

The Chandrayaan-2 **IIRS** (Imaging IR Spectrometer) is a hyperspectral imager.
Its calibrated product is delivered in **ENVI** format (BSQ interleave):

    ch2_iir_nri_*.qub   (binary raster)
    ch2_iir_nri_*.hdr   (ENVI header: samples=250, lines=12620, bands=256,
                         data type=2 i.e. int16, interleave=bsq)

IIRS is radiometrically very different from the visible OHRC/TMC/NAC sensors, so
it is the archetypal "multi-modal" case in PS 26166. This module provides:

  * ``read_envi`` — bytes-bounded ENVI BSQ reader (samples/lines/bands from the
    header) that never loads more than one band at a time (256 bands × 250 ×
    12620 × 2 B would exceed RAM if read whole).
  * ``read_iirs_raster`` — convenience wrapper for IIRS `.qub`+`.hdr`.
  * ``choose_representative_band`` — pick a handful of well-contrasted bands to
    carry into cross-modal matching (IIRS Band ~40/116 are useful vs visible).
  * ``register_iirs_to_ohrc`` — stages a few representative IIRS bands, warps
    them via ``cross_modal_register`` against an OHRC frame, and returns the
    best candidate. It is intentionally proof-based: a `verdict: not_registered`
    is returned (never a fabricated homography) if no band yields a reliable
    model.
"""

from __future__ import annotations

import os

import cv2
import numpy as np


def parse_envi_header(hdr_path):
    """Parse an ENVI header file into a dict of key -> value.

    Handles the common ENVI header format (``key = value`` lines, values may
    contain commas for arrays). Lower-cases keys for tolerant lookup.
    """
    meta = {}
    with open(hdr_path) as fh:
        for line in fh:
            line = line.strip()
            if not line or "=" not in line:
                continue
            k, _, v = line.partition("=")
            meta[k.strip().lower()] = v.strip()
    return meta


def read_envi(raster_path, hdr_path=None, bands=None, memmap=False):
    """Read an ENVI BSQ raster, band-selectively.

    Args:
        raster_path: path to the binary (.qub / .raw).
        hdr_path: path to the .hdr; if None, derived from raster_path (.hdr).
        bands: optional list of 0-based band indices to load (None = first band
            only by default).
        memmap: if True, return a memmap shaped (bands, lines, samples) so the
            caller can sample without loading everything.

    Returns a dict with keys ``data`` (uint8 or int16 array per requested bands),
    plus the parsed header ``meta``.
    """
    hdr_path = hdr_path or (raster_path.rsplit(".", 1)[0] + ".hdr")
    meta = parse_envi_header(hdr_path)
    samples = int(meta.get("samples", "0"))
    lines = int(meta.get("lines", "0"))
    nbands = int(meta.get("bands", "0"))
    dtype = int(meta.get("data type", "1"))
    interleave = meta.get("interleave", "bsq").strip().lower()

    dtype_map = {1: np.uint8, 2: np.int16, 3: np.int32, 4: np.float32,
                 5: np.float64, 12: np.uint16, 13: np.uint32, 14: np.int64}
    dt = dtype_map.get(dtype, np.uint8)
    expected = samples * lines * nbands * np.dtype(dt).itemsize
    size = os.path.getsize(raster_path)
    if size < expected:
        raise ValueError(
            f"IIRS raster too small: got {size} B, expected ~{expected} B "
            f"({samples}x{lines}x{nbands} {dt})")

    if interleave != "bsq":
        raise NotImplementedError(
            f"only BSQ interleave supported, got {interleave!r}")

    if bands is None:
        bands = [0]
    bands = [int(b) for b in bands]

    if memmap:
        mm = np.memmap(raster_path, dtype=dt, mode="r",
                       shape=(nbands, lines, samples))
        return {"data": mm[bands], "meta": meta, "bands": bands}

    # BSQ: each band is a contiguous lines*samples block; read only the requested
    # bands by seeking, so a 1.6 GB 256-band product costs ~band-block bytes.
    band_bytes = lines * samples * np.dtype(dt).itemsize
    out = np.empty((len(bands), lines, samples), dtype=dt)
    with open(raster_path, "rb") as fh:
        for out_i, b in enumerate(bands):
            fh.seek(b * band_bytes)
            raw = fh.read(band_bytes)
            out[out_i] = np.frombuffer(raw, dtype=dt).reshape(lines, samples)
    return {"data": out, "meta": meta, "bands": bands}


def read_iirs_raster(qub_path, hdr_path=None, bands=None, memmap=False):
    """Convenience: read_envi for IIRS .qub + .hdr."""
    return read_envi(qub_path, hdr_path or (qub_path.rsplit(".", 1)[0] + ".hdr"),
                     bands=bands, memmap=memmap)


def choose_representative_bands(meta, n=4, contrast_kernel=(3, 3)):
    """Choose well-contrasted IIRS bands to carry into cross-modal matching.

    Returns a list of band indices (0-based). If the header lacks band info this
    falls back to a fixed, physically-reasonable set (IR + NIR + SWIR).
    """
    nbands = int(meta.get("bands", "256"))
    if nbands <= n:
        return list(range(min(n, max(1, nbands))))
    # physically useful IIRS band windows (approx): visible/blue ~1-40,
    # NIR ~40-100, SWIR later. Sample a spread.
    lo, hi = max(0, nbands // 8), max(1, nbands - 1)
    step = max(1, (hi - lo) // n)
    return [min(hi, lo + i * step) for i in range(n)]


def as_u8(arr):
    """Normalise an arbitrary 2-D band to uint8 (per-array min/max stretch)."""
    a = np.asarray(arr, np.float32)
    if a.size == 0:
        return np.zeros_like(arr, np.uint8)
    lo, hi = float(np.nanpercentile(a, 2)), float(np.nanpercentile(a, 98))
    if hi <= lo:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def register_iirs_to_ohrc(iirs_qub, iirs_hdr, ohrc_img, ohrc_geom, out_dir,
                          bands=None, n_bands=4, prefix="iirs_ohrc",
                          min_inliers=12, min_ratio=0.02, **match_kw):
    """Cross-modal registration of selected IIRS bands against an OHRC frame.

    Workflow:
      1. Read N representative IIRS bands (BSQ, band-selective).
      2. Downsample the OHRC raw to a grayscale working crop (native uint8).
      3. For each IIRS band, run ``cross_modal_register`` (geometry-dominant
         front-ends) against the OHRC crop and keep the best candidate.
      4. Record a per-band table + best model; return a JSON report. Never
         fabricates a model.

    Requires both full rasters; callers should pass bounded crops to keep the
    IIRS and OHRC cost low (crop-first rule).
    """
    from src.preprocessing.tmc import read_ch2_raw, OHRC_WIDTH
    from src.preprocessing.ch2_staging import enhance_dark
    os.makedirs(out_dir, exist_ok=True)
    if bands is None:
        meta0 = parse_envi_header(iirs_hdr)
        bands = choose_representative_bands(meta0, n=n_bands)
    r = read_envi(iirs_qub, iirs_hdr, bands=bands)
    iirs_bands = r["data"]  # (len(bands), lines, samples)
    meta = r["meta"]

    ohrc = read_ch2_raw(ohrc_img, ohrc_geom, width=OHRC_WIDTH)
    rows, cols = ohrc.shape
    mid = rows // 2
    slice_rows = min(1024, rows)
    ohrc_crop = ohrc[mid - slice_rows // 2: mid + slice_rows // 2, :]
    ohrc_ws = cv2.resize(ohrc_crop, (256, 256), interpolation=cv2.INTER_AREA)
    ohrc_ref = enhance_dark(ohrc_ws)
    native_ref = {"rows": int(rows), "cols": int(cols)}

    return _register_iirs_vs_reference(
        iirs_bands, bands, ohrc_ref, meta, out_dir, prefix,
        pair="IIRS -> OHRC", reference="OHRC",
        min_inliers=min_inliers, min_ratio=min_ratio, match_kw=match_kw,
        native_ref=native_ref, ref_before=ohrc_crop, ref_before_lab="OHRC strip",
    )


def register_iirs_to_nac(iirs_qub, iirs_hdr, nac_img, out_dir, bands=None,
                         n_bands=4, prefix="iirs_nac", min_inliers=12,
                         min_ratio=0.02, **match_kw):
    """Content-only registration of IIRS bands against an **LRO NAC** reference.

    IIRS products carry no ISRO geometry CSV / ENVI map-info, so a ground-grid
    (geometry) registration is impossible for IIRS — this is a content attempt
    only, and the verdict is an honest ``not_registered`` when no band yields a
    reliable model (never a fabricated homography). The NAC is read as a
    grayscale working crop via the shared rasters reader.
    """
    from src.preprocessing.georeference import read_nac_img
    from src.preprocessing.geometry import as_u8
    from src.preprocessing.ch2_staging import enhance_dark
    os.makedirs(out_dir, exist_ok=True)
    if bands is None:
        meta0 = parse_envi_header(iirs_hdr)
        bands = choose_representative_bands(meta0, n=n_bands)
    r = read_envi(iirs_qub, iirs_hdr, bands=bands)
    iirs_bands = r["data"]
    meta = r["meta"]

    nac = read_nac_img(nac_img)
    rows, cols = nac.shape
    mid = rows // 2
    slice_rows = min(1024, rows)
    nac_crop = nac[mid - slice_rows // 2: mid + slice_rows // 2, :]
    nac_ws = cv2.resize(as_u8(nac_crop), (256, 256),
                        interpolation=cv2.INTER_AREA)
    nac_ref = enhance_dark(nac_ws)
    native_ref = {"rows": int(rows), "cols": int(cols)}

    return _register_iirs_vs_reference(
        iirs_bands, bands, nac_ref, meta, out_dir, prefix,
        pair="IIRS -> LRO NAC", reference="LRO NAC",
        min_inliers=min_inliers, min_ratio=min_ratio, match_kw=match_kw,
        native_ref=native_ref, ref_before=nac_crop, ref_before_lab="NAC working crop",
    )


def _register_iirs_vs_reference(iirs_bands, bands, ref_ws, meta, out_dir, prefix,
                                *, pair, reference, min_inliers, min_ratio,
                                match_kw, native_ref, ref_before, ref_before_lab):
    """Shared IIRS content-vs-(grayscale reference crop) register body."""
    from src.detection.cross_modal import cross_modal_register
    from src.preprocessing.ch2_staging import (enhance_dark, low_contrast_score,
                                                percentile_stretch, _outline_registered)

    per_band = []
    best = None
    for i, b in enumerate(bands):
        band_2d = iirs_bands[i] if iirs_bands.ndim == 3 else iirs_bands[0]
        u = as_u8(band_2d)
        u_ws = cv2.resize(u, (256, 256), interpolation=cv2.INTER_AREA)
        u_ref = enhance_dark(u_ws)
        res = cross_modal_register(u_ref, ref_ws, min_inliers=min_inliers,
                                   min_ratio=min_ratio, verbose=False, **match_kw)
        rec = {"band": int(b), "ok": res is not None,
               "low_contrast": round(low_contrast_score(u_ws), 3)}
        if res:
            rec.update(inliers=res["inliers"], inlier_ratio=res["inlier_ratio"],
                       front=res["front"], rmse=res.get("rmse"))
            if best is None or res["inliers"] > best["inliers"]:
                best = {**rec, "H": res["H"], "pts1": res.get("pts1"),
                        "pts2": res.get("pts2"), "inl": res.get("inl")}
        else:
            rec.update(inliers=0, inlier_ratio=0.0, front="n/a")
        per_band.append(rec)

    import json

    def _bin_preview(arr, max_w=640, max_h=2048):
        a = arr.astype(np.float32)
        h, w = a.shape
        s = max(1, -(-h // max_h), -(-w // max_w)) if (h > max_h or w > max_w) else 1
        if s > 1:
            h2, w2 = h // s * s, w // s * s
            a = a[:h2, :w2].reshape(h2 // s, s, w2 // s, s).mean(axis=(1, 3))
        return (percentile_stretch(a) if a.max() > a.min()
                else np.zeros_like(a)).astype(np.uint8)

    orig_iirs = iirs_bands[i] if iirs_bands.ndim == 3 else iirs_bands[0]
    if best is not None and iirs_bands.ndim == 3:
        bi = int(best.get("band"))
        bl = [int(x) for x in bands]
        if bi in bl:
            orig_iirs = iirs_bands[bl.index(bi)]
    prev_src = os.path.join(out_dir, f"{prefix}_original_src.png")
    prev_ref = os.path.join(out_dir, f"{prefix}_original_ref.png")
    cv2.imwrite(prev_src, _bin_preview(np.asarray(orig_iirs)))
    cv2.imwrite(prev_ref, _bin_preview(ref_before))
    proc_src = os.path.join(out_dir, f"{prefix}_src.png")
    proc_ref = os.path.join(out_dir, f"{prefix}_ref.png")
    cv2.imwrite(proc_src, enhance_dark(u_ws))
    cv2.imwrite(proc_ref, ref_ws)
    after_src_p = os.path.join(out_dir, f"{prefix}_after_src.png")
    after_ref_p = os.path.join(out_dir, f"{prefix}_after_ref.png")
    cv2.imwrite(after_src_p, _outline_registered(enhance_dark(u_ws)))
    cv2.imwrite(after_ref_p, _outline_registered(ref_ws))
    diff = np.abs(enhance_dark(u_ws).astype(np.int16)
                  - ref_ws.astype(np.int16)).astype(np.uint8)
    diff = percentile_stretch(diff) if diff.max() > diff.min() else diff
    change_p = os.path.join(out_dir, f"{prefix}_change.png")
    cv2.imwrite(change_p, cv2.applyColorMap(diff, cv2.COLORMAP_TURBO))
    montage = os.path.join(out_dir, f"{prefix}_montage.png")
    cv2.imwrite(montage, np.hstack([cv2.imread(after_src_p),
                                    cv2.imread(after_ref_p), cv2.imread(change_p)]))
    import src.preprocessing.ch2_staging as _cs
    _save_iirs_correspondences = _cs._save_correspondences
    saved_csv = None
    if best is not None:
        report_proxy = {"artifacts": {}, "notes": []}
        _save_iirs_correspondences(
            report_proxy,
            {"pts1": best.get("pts1"), "pts2": best.get("pts2"),
             "inl": best.get("inl")},
            out_dir, prefix)
        saved_csv = report_proxy["artifacts"].get("correspondences")

    report = {
        "pair": pair,
        "reference": reference,
        "verdict": "registered" if best else "not_registered",
        "n_bands": len(bands),
        "bands_used": [int(b) for b in bands],
        "per_band": per_band,
        "method": "cross_modal (geometry-dominant front-ends)",
        "ref_low_contrast": round(low_contrast_score(ref_ws), 3),
        "geometry_available": False,
        "dimensions": {
            "native_src": {"rows": int(iirs_bands.shape[1]),
                           "cols": int(iirs_bands.shape[2])},
            "native_ref": native_ref,
            "workspace": {"rows": 256, "cols": 256},
            "gsd_m": None,
        },
        "artifacts": {
            "original_src": prev_src,
            "original_ref": prev_ref,
            "after_src": after_src_p,
            "after_ref": after_ref_p,
            "change_map": change_p,
            "montage": montage,
            "src": proc_src,
            "ref": proc_ref,
        },
        "notes": [
            "IIRS products carry no ISRO geometry CSV / ENVI map-info: ground-grid "
            "(geometry) registration is not possible for IIRS. Registration is "
            "content-based only; if content matching fails on a dark/polar/IR-vs-"
            f"visible pair the verdict is an honest not_registered (never a "
            "fabricated model). Contrast enhancement is applied before matching."
            f" Reference: {reference}."
        ],
    }
    if saved_csv:
        report["artifacts"]["correspondences"] = saved_csv
    if best:
        # carry full match sets so the CSV/proxy was written with actual arrays
        report.update(best_inliers=best.get("inliers"),
                      best_front=best.get("front"),
                      best_band=best.get("band"),
                      best_rmse=best.get("rmse"),
                      H=best.get("H"))
    with open(os.path.join(out_dir, f"{prefix}_report.json"), "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    return report
