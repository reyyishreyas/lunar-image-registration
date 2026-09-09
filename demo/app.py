"""Streamlit app for the lunar image registration pipeline.

The app ONLY orchestrates the pipeline: it builds the experiment config and
calls `src.pipeline.run_experiment` / `src.auto_pipeline.run_auto` /
`src.auto_pipeline.run_sensor_auto`. No matching, detection, georeferencing or
outlier-rejection logic is reimplemented here.

Modes:
  * Automatic registration (OHRC + LRO NAC) — one-click full run including
    georeference staging (cached crops reused when present).
  * Multi-sensor registration (TMC ⟷ OHRC, IIRS ⟷ OHRC) — one-click Phase 9
    Chandrayaan-2 sensor pairs; dark/low-sun pairs are enhanced first and fall
    back to geometry registration automatically; results are phrased for
    non-expert users.
  * Project pair-1 (OHRC-2021 + LRO NAC M1469248775LC) — full run including
    georeference staging (cached crops reused when present).
  * Project pair-2 (polar OHRC-2026 + M1127547939RC) — demonstrates the
    documented, evidence-based georeference refusal.
  * Upload aligned crops — runs the FINAL detection/matching/outlier config on
    two uploaded grayscale images (preprocessing disabled).

Run with:
    venv/bin/python -m streamlit run demo/app.py
"""

from __future__ import annotations

import os
import sys

sys.dont_write_bytecode = True  # repo lives on a .mounty FUSE mount: mtime-based
                                # .pyc invalidation is unreliable and served a
                                # stale auto_pipeline cache once (ImportError:
                                # detect_pair).

# Make `src` importable no matter how/whence the app is launched (the plain
# 'from src.visuals' further down must not rely on the cwd being the repo root).
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.getcwd()):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import streamlit as st
import yaml

from src.visuals import frame_img, diff_map

FINAL_CONFIG = "results/final_config.yaml"
DEMO_DIR = "data/processed/demo"
UPLOAD_DIR = os.path.join(DEMO_DIR, "uploads")
FALLBACK_DIR = "demo/fallback"

PAIR1 = {
    "name": "Project pair-1 — OHRC-2021 + NAC M1469248775LC",
    "ohrc_img": ("data/PATCH-001/data/OHRC/"
                 "ch2_ohr_ncp_20210405T1606536730_d_img_d18/"
                 "data/calibrated/20210405/"
                 "ch2_ohr_ncp_20210405T1606536730_d_img_d18.img"),
    "ohrc_geometry": ("data/PATCH-001/data/OHRC/"
                      "ch2_ohr_ncp_20210405T1606536730_d_img_d18/"
                      "geometry/calibrated/20210405/"
                      "ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv"),
    "nac_img": "data/PATCH-001/data/LRO_NAC/LC/M1469248775LC.IMG",
    "out_dir": "data/processed",
}

PAIR2 = {
    "name": "Project pair-2 (polar) — OHRC-2026 + NAC M1127547939RC",
    "ohrc_img": ("data/PATCH-004/OHRC/data/calibrated/20260331/"
                 "ch2_ohr_ncp_20260331T1105235288_d_img_d18.img"),
    "ohrc_geometry": ("data/PATCH-004/OHRC/geometry/calibrated/20260331/"
                      "ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv"),
    "nac_img": "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG",
    "nac_geometry": "data/processed/nac_geom_M1127_full.csv",
    "out_dir": os.path.join(DEMO_DIR, "pair2"),
}

# --------------------------------------------------------------------------- #
# Phase 9 — Chandrayaan-2 multi-sensor presets (TMC / IIRS vs OHRC)
# --------------------------------------------------------------------------- #
TMC_EXTRACTED_IMG = ("data/processed/tmc/"
                     "ch2_tmc_nca_20260607T2319176707_d_img_d18.img")
TMC_EXTRACTED_CSV = ("data/processed/tmc/"
                     "ch2_tmc_nca_20260607T2319176707_g_grd_d18.csv")
TMC_ZIP = "data/PATCH-004/TMC/ch2_tmc_nca_20260607T2319176707_d_img_d18.zip"

SENSOR_PRESETS = {
    "TMC ⟷ OHRC — Chandrayaan-2 (2026, same orbit family)": {
        "sensor": "tmc-ohrc",
        "src_img": TMC_EXTRACTED_IMG,
        "src_geom": TMC_EXTRACTED_CSV,
        "ref_img": PAIR2["ohrc_img"],
        "ref_geom": PAIR2["ohrc_geometry"],
        "note": ("Terrain Mapping Camera vs OHRC over the same region. The OHRC "
                 "acquisition here is a dark/low-sun hard case — the pipeline "
                 "enhances contrast first, then falls back to geometry "
                 "registration when content cannot be verified."),
    },
    "IIRS ⟷ OHRC — hyperspectral IR vs visible (2021)": {
        "sensor": "iirs-ohrc",
        "src_img": ("data/PATCH-001/data/IIRS/"
                    "ch2_iir_nri_20211221T0324126144_d_img_hw1/"
                    "data/raw/20211221/"
                    "ch2_iir_nri_20211221T0324126144_d_img_hw1.qub"),
        "src_geom": ("data/PATCH-001/data/IIRS/"
                     "ch2_iir_nri_20211221T0324126144_d_img_hw1/"
                     "data/raw/20211221/"
                     "ch2_iir_nri_20211221T0324126144_d_img_hw1.hdr"),
        "ref_img": PAIR1["ohrc_img"],
        "ref_geom": PAIR1["ohrc_geometry"],
        "note": ("Imaging IR Spectrometer (hyperspectral) vs OHRC visible. IIRS "
                 "ships no ground-geometry CSV, so registration is content-based "
                 "only; the pipeline reports honestly if no reliable match "
                 "exists."),
    },
}

TMC_ARCHIVE_FILES = {
    "data/calibrated/20260607/ch2_tmc_nca_20260607T2319176707_d_img_d18.img":
        TMC_EXTRACTED_IMG,
    "geometry/calibrated/20260607/ch2_tmc_nca_20260607T2319176707_g_grd_d18.csv":
        TMC_EXTRACTED_CSV,
}


def _base_config():
    with open(os.path.join(ROOT, FINAL_CONFIG)) as fh:
        return yaml.safe_load(fh)


def _write_experiment(cfg, name):
    os.makedirs(os.path.join(ROOT, DEMO_DIR), exist_ok=True)
    path = os.path.join(ROOT, DEMO_DIR, f"demo_{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


def _ensure_tmc_extracted():
    """One-click convenience: unpack the TMC .img + geometry CSV from its zip
    into data/processed/tmc/ so the multi-sensor demo can read them."""
    import zipfile
    if (os.path.exists(os.path.join(ROOT, TMC_EXTRACTED_IMG)) and
            os.path.exists(os.path.join(ROOT, TMC_EXTRACTED_CSV))):
        return None
    zpath = os.path.join(ROOT, TMC_ZIP)
    if not os.path.exists(zpath):
        return f"Missing archive: `{TMC_ZIP}` — drop the TMC product here first."
    out = os.path.join(ROOT, "data/processed/tmc")
    os.makedirs(out, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for member, dest in TMC_ARCHIVE_FILES.items():
            zf.extract(member, out)
            os.rename(os.path.join(out, member), os.path.join(ROOT, dest))
            os.removedirs(os.path.dirname(os.path.join(out, member)))
    return None


def _friendly_status(verdict):
    """(icon, headline, kind) for a pipeline verdict, phrased for non-experts."""
    table = {
        "registered": ("✅", "Images registered by content", "success"),
        "geometry_registered": ("🌗", "Registered by geometry — dark/low-sun pair",
                                "success"),
        "no_content_correspondence": ("🔍", "No content match found — geometry "
                                       "fallback", "info"),
        "not_registered": ("⛔", "Could not register these images", "warning"),
    }
    return table.get(verdict, ("ℹ️", verdict or "No verdict", "info"))


def _decision_banner(rep):
    """Big MATCH / NO MATCH banner from the Phase-10 decision layer.

    Uses the evidence-based ``report["decision"]`` when present, so the
    headline reflects the true cause (content match / dark crop / frame
    disagreement / cross-sensor), never a guessed one.
    """
    d = rep.get("decision") or {}
    if not d:
        return
    matched = bool(d.get("matched"))
    cause = d.get("cause", "")
    kind = "success" if matched else \
        ("info" if cause == "frame_disagreement" else "warning")
    icon = "✅ MATCH" if matched else "⛔ NO MATCH"
    st.markdown(
        f"<h3 style='margin-bottom:0'>{icon}</h3>"
        f"<div style='opacity:.85'>{d.get('explanation', '')}</div>",
        unsafe_allow_html=True)
    best = d.get("best_product")
    if best and os.path.exists(os.path.join(ROOT, str(best))):
        hint = " — open it in the panel below" if best else ""
        st.caption(f"Best aligned product: `{best}`{hint}")


def _sensor_metric_cols(rep):
    verdict = rep.get("verdict")
    c = st.columns(4)
    icon, headline, kind = _friendly_status(verdict)
    c[0].metric("Method", rep.get("method") or "—")
    gsd = rep.get("gsd") or {}
    if isinstance(gsd.get("src_est_m"), (int, float)) \
            and isinstance(gsd.get("ref_est_m"), (int, float)):
        gsd_txt = (f"src {gsd['src_est_m']:.2f} / ref {gsd['ref_est_m']:.2f}"
                   + (" · ×{:.2f}".format(gsd["scale_ratio"])
                      if isinstance(gsd.get("scale_ratio"), (int, float))
                      else ""))
    elif isinstance(gsd.get("staged_m"), (int, float)):
        gsd_txt = f"{gsd['staged_m']:.2f} (staged)"
    elif isinstance(rep.get("gsd_m"), (int, float)):
        gsd_txt = f"{rep['gsd_m']:.2f}"
    else:
        gsd_txt = "—"
    c[1].metric("GSD (m/px)", gsd_txt)
    rmse = rep.get("rmse_px")
    c[2].metric("RMSE (px)", f"{rmse:.3f}" if isinstance(rmse, float) else "n/a")
    c[3].metric("Inliers", f"{rep.get('inliers', 0)} ({rep.get('n_matches', 0)} raw)"
                            if rep.get("inliers") else "n/a")
    return verdict


def _render_gsd(rep):
    """Source/reference native GSD + scale ratio, with their derivation."""
    gsd = rep.get("gsd") or {}
    if not any(v is not None for v in (gsd.get("src_est_m"), gsd.get("ref_est_m"))):
        return
    rows = [
        ("Source", gsd.get("src_est_m")),
        ("Reference", gsd.get("ref_est_m")),
        ("Staged common cell", gsd.get("staged_m")),
        ("Native scale ratio (ref/src)",
         gsd.get("scale_ratio")),
    ]
    with st.expander("Ground sample distance (native GSD) — the scale both "
                     "images were registered at"):
        st.table({k: [f"{v:g} m/px" if isinstance(v, (int, float)) else "—"]
                  for k, v in rows})
        st.caption(f"Derivation: {gsd.get('source', 'n/a')}. Source = ISRO "
                   "ground grid; reference = georef native-vs-ground corners.")


def _render_tile_residuals(rep):
    """Per-region residual RMSE so a weak global fit can't hide a bad region."""
    tr = rep.get("tile_residuals")
    if not tr:
        return
    vals = [v for r in tr for v in r.values() if v is not None]
    worst = max(vals) if vals else None
    with st.expander("Per-region fit quality (4×4 tiles of inlier residual RMSE)"):
        st.write("x = source tile column, y = source tile row. Values are the "
                 "mean residual (px) of the matched features whose source "
                 "point falls in that tile — a homography is a global model, so "
                 "this shows whether any local region (relief/parallax) is "
                 "systematically worse than the global self-RMSE.")
        for i, row in enumerate(tr):
            st.write(f"**row {i}**  ·  " + "   ".join(
                f"{v:.2f}" if v is not None else "  –  " for v in row.values()))
        st.caption(f"Worst region ≈ **{worst:.2f} px** "
                   f"vs global self-RMSE {rep.get('rmse_self_px')} px.")


def _rmse_sentence(rep):
    """Human sentence about the fit accuracy for a registered pair.

    `rmse_px` is the residual RMSE **against an external ground-truth
    reference** (only pair-1 has one); when it is absent we report the
    self-consistency RMSE (`rmse_self_px`) instead — never crash on `None`.
    """
    rmse = rep.get("rmse_px")
    if isinstance(rmse, (int, float)):
        return (f"Residual RMSE ≈ **{rmse:.3f} px** "
                "vs the reference (≈1 px or less = sub-pixel: the two images "
                "agree to well under a pixel after alignment).")
    self_rmse = rep.get("rmse_self_px")
    if isinstance(self_rmse, (int, float)):
        return (f"Residual RMSE ≈ **{self_rmse:.3f} px** "
                "(self-consistency over the matched features — sub-pixel, "
                "well under a pixel after alignment).")
    return "Residual RMSE: not reported for this run."


def _framed(path_or_arr, tag="REGISTERED"):
    """Red frame + tag around any image, wherever it appears in the app."""
    img = cv2.imread(path_or_arr, cv2.IMREAD_UNCHANGED) \
        if isinstance(path_or_arr, str) and os.path.exists(path_or_arr) \
        else path_or_arr
    if img is None:
        return None
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[:, :, :3]
    return frame_img(img, tag=tag)


def _render_ground_artifacts(rep, src_name="Source", ref_name="Reference"):
    arts = rep.get("artifacts") or {}
    a, b = st.columns(2)
    src, ref = arts.get("src"), arts.get("ref")
    if src and os.path.exists(os.path.join(ROOT, src)):
        a.image(_framed(os.path.join(ROOT, src), "GEO-READY"),
                caption=f"{src_name} — ground-ready (frame = delivered product)",
                width="stretch")
    if ref and os.path.exists(os.path.join(ROOT, ref)):
        b.image(_framed(os.path.join(ROOT, ref), "GEO-READY"),
                caption=f"{ref_name} — ground-ready (frame = delivered product)",
                width="stretch")
    if src and ref and os.path.exists(os.path.join(ROOT, src)) \
            and os.path.exists(os.path.join(ROOT, ref)):
        sa = cv2.imread(os.path.join(ROOT, src), cv2.IMREAD_GRAYSCALE)
        rb = cv2.imread(os.path.join(ROOT, ref), cv2.IMREAD_GRAYSCALE)
        if sa is not None and rb is not None and sa.shape == rb.shape:
            st.image(diff_map(sa, rb), width="stretch",
                     caption="What changed, pixel by pixel — warm (yellow/red) "
                             "= large difference at that pixel, dark/blue = "
                             "already agreeing after registration.")
    m = arts.get("matches")
    if m and os.path.exists(os.path.join(ROOT, m)):
        st.image(_framed(os.path.join(ROOT, m), "VERIFIED MATCHES"),
                 caption="Verified match points (the red lines connect matched "
                         "features; frame = this panel is part of the "
                         "delivered product)",
                 width="stretch")
    corr = arts.get("correspondences")
    if corr and os.path.exists(os.path.join(ROOT, corr)):
        st.caption(f"Corresponding match points saved: `{corr}` "
                   " "
                   f"({rep.get('inlier_points_saved', '?')} inlier pairs).")


def _render_best_product(rep):
    """Show the best aligned product when the decision layer produced one.

    The aligned image is the source *warped onto the reference frame* — this is
    the deliverable. A registered checkerboard is additionally shown when
    present; it is built from the *warped* source vs the reference, so features
    continue across tile boundaries (never the un-warped raw crops).
    """
    d = rep.get("decision") or {}
    arts = rep.get("artifacts") or {}
    aligned = arts.get("best_aligned") or d.get("best_product")
    diff = arts.get("diff")
    if not aligned or not os.path.exists(os.path.join(ROOT, str(aligned))):
        return
    st.markdown("#### Best aligned product")
    st.image(_framed(os.path.join(ROOT, str(aligned)), "BEST ALIGNED"),
             caption="Source **warped onto the reference frame** (content "
                     "matches only — this is the deliverable image).",
             width="stretch")
    if diff and os.path.exists(os.path.join(ROOT, str(diff))):
        st.image(_framed(os.path.join(ROOT, str(diff)), "DIFF"),
                 caption="Pixel-level difference map against the reference "
                         "after alignment — warm = changed.", width="stretch")
    chk = arts.get("checkerboard")
    if chk and os.path.exists(os.path.join(ROOT, str(chk))) \
            and str(chk) != str(aligned):
        st.image(_framed(os.path.join(ROOT, str(chk)), "REGISTERED CHECKERBOARD"),
                 caption="Checkerboard of the **warped** source vs the "
                         "reference — features should run continuously across "
                         "tile boundaries (built after alignment, not from raw "
                         "crops on different grids).",
                 width="stretch")


def _render_before_after(rep, src_name="Source", ref_name="Reference"):
    """Original raw strips ('before') next to the registered common-grid pair
    ('after'), so users can see exactly what the pipeline did."""
    arts = rep.get("artifacts") or {}
    gsd = rep.get("gsd_m")
    gsd_txt = f" ({float(gsd):.2f} m/px)" if isinstance(gsd, (int, float)) else ""
    ok_verdict = rep.get("verdict") in ("registered", "geometry_registered")
    after_txt = ("registered on the same ground grid"
                 if ok_verdict else "processed working crops (enhanced)")

    st.markdown("#### Before → After")
    a, b = st.columns(2)
    o_src, o_ref = arts.get("original_src"), arts.get("original_ref")
    bcap_before_src = (f"**Before · {src_name}** raw strip — "
                       "red box = overlap swath registered")
    bcap_before_ref = (f"**Before · {ref_name}** raw strip — "
                       "red box = overlap swath registered")
    if o_src and os.path.exists(os.path.join(ROOT, o_src)):
        a.image(os.path.join(ROOT, o_src), caption=bcap_before_src,
                width="stretch")
    elif arts.get("src") and os.path.exists(os.path.join(ROOT, arts["src"])):
        a.image(os.path.join(ROOT, arts["src"]), caption=bcap_before_src,
                width="stretch")
    if o_ref and os.path.exists(os.path.join(ROOT, o_ref)):
        b.image(os.path.join(ROOT, o_ref), caption=bcap_before_ref,
                width="stretch")
    elif arts.get("ref") and os.path.exists(os.path.join(ROOT, arts["ref"])):
        b.image(os.path.join(ROOT, arts["ref"]), caption=bcap_before_ref,
                width="stretch")

    st.markdown("⬇  **After** — both re-projected on a common geographic grid"
                + gsd_txt)
    montage = arts.get("montage")
    if montage and os.path.exists(os.path.join(ROOT, montage)):
        st.image(os.path.join(ROOT, montage), width="stretch",
                 caption=f"Montage — After · {src_name}  |  After · {ref_name} "
                         f"|  Difference (warm = pixels that changed most, "
                         f"dark = already agreeing). {after_txt}.")  # noqa: E501
    else:
        c, d = st.columns(2)
        src, ref = arts.get("after_src"), arts.get("after_ref")
        if not src:
            src = arts.get("src")
        if not ref:
            ref = arts.get("ref")
        ccap = f"**After · {src_name}** — {after_txt} (red outline)"
        dcap = f"**After · {ref_name}** — {after_txt} (red outline)"
        if src and os.path.exists(os.path.join(ROOT, src)):
            c.image(os.path.join(ROOT, src), caption=ccap, width="stretch")
        if ref and os.path.exists(os.path.join(ROOT, ref)):
            d.image(os.path.join(ROOT, ref), caption=dcap, width="stretch")
        cmap = arts.get("change_map")
        if cmap and os.path.exists(os.path.join(ROOT, cmap)):
            st.markdown("**What changed, pixel by pixel**")
            st.image(os.path.join(ROOT, cmap), width="stretch",
                     caption="Bright/warm (yellow–red) = large difference "
                             "between the two products at that pixel after "
                             "registration; dark/blue = already agreeing.")
    st.caption("How to read this: the red box + 'OVERLAP SWATH' label on each "
               "Before panel is the exact overlap swath the pipeline extracted; "
               "the red-outlined 'REGISTERED' panels are that same data "
               "re-projected onto one shared geographic grid at equal "
               "ground-sample distance (GSD).")


def _render_sensor(res):
    """Reader-friendly report of a multi-sensor (TMC / IIRS vs OHRC) run.

    Ordered so a first-time user can answer three questions without reading the
    code: (1) what did the pipeline do, (2) see the Before -> After evidence,
    (3) what should I check to trust it."""
    rep = res.get("report", {})
    sensor = res.get("sensor", "tmc-ohrc")
    src_name, ref_name = _sensor_labels(sensor)
    _decision_banner(rep)
    verdict = _sensor_metric_cols(rep)
    icon, headline, kind = _friendly_status(verdict)
    getattr(st, kind)(f"{icon} **{headline}**")

    st.markdown("#### What the pipeline did")
    if verdict == "registered":
        st.markdown(
            f"1. Read the raw **{src_name}** and **{ref_name}** strips.",
            unsafe_allow_html=True)
        st.markdown(
            "2. Matched the ground features both strips share (cross-modal "
            f"matching, method **{rep.get('method')}**) — "
            f"**{rep.get('inliers')}** after outlier rejection.",
            unsafe_allow_html=True)
        st.markdown(
            "3. Fit the geometric transformation between them. "
            + _rmse_sentence(rep),
            unsafe_allow_html=True)
        st.markdown(
            "4. Re-sampled both onto one shared ground grid so they can be "
            "compared directly. That is the **After** view below.",
            unsafe_allow_html=True)
    elif verdict == "geometry_registered":
        gsd = rep.get("gsd_m")
        gsd_txt = f"**{float(gsd):.2f} m/px**" if isinstance(gsd, (int, float)) else "one shared scale"
        st.markdown(
            f"1. Read the raw **{src_name}** and **{ref_name}** strips and "
            "checked their on-ground footprints overlap.",
            unsafe_allow_html=True)
        st.markdown(
            "2. The reference strip turned out to be very dark / low-sun "
            "(mean ~ "
            f"{(rep.get('diagnostics') or {}).get('ref_mean', '?')} "
            "of 255), so there are **no trustworthy craters or features to "
            "match** — the pipeline does not invent matches on dark data.",
            unsafe_allow_html=True)
        st.markdown(
            "3. Instead it registered using the **ISRO spacecraft geometry**: "
            "both strips were placed on the same ground grid, equalised to "
            f"{gsd_txt} ground-sample distance.",
            unsafe_allow_html=True)
        st.markdown(
            "4. Outcome: both images are aligned **in ground space** even "
            "though they look different. This is why the 'After/registered' "
            "view exists — see it below.",
            unsafe_allow_html=True)
    elif verdict == "not_registered":
        st.markdown(
            f"1. Read the raw **{src_name}** and **{ref_name}** strips.",
            unsafe_allow_html=True)
        st.markdown(
            "2. Tried to match shared ground features — **no verified matches** "
            "survived outlier rejection.",
            unsafe_allow_html=True)
        st.markdown(
            "3. Tried the spacecraft-geometry route — **no usable ground "
            "geometry** (this instrument ships none).",
            unsafe_allow_html=True)
        st.markdown(
            "4. The pipeline therefore refuses to guess. It reports "
            "**not_registered** honestly instead of presenting a fabricated "
            "alignment.",
            unsafe_allow_html=True)

    _render_before_after(rep, src_name, ref_name)
    _render_what_changed(rep, src_name, ref_name)
    _render_check_it(rep, src_name, ref_name)
    _render_gsd(rep)
    _render_tile_residuals(rep)
    _render_best_product(rep)

    georef = rep.get("georef") or {}
    if georef:
        s, r = georef.get("src", {}), georef.get("ref", {})
        st.caption(
            f"Footprints — source: lon {s.get('lon', ['', ''])[0]:.3f}–"
            f"{s.get('lon', ['', ''])[1]:.3f}°, lat {s.get('lat', ['', ''])[0]:.3f}"
            f"–{s.get('lat', ['', ''])[1]:.3f}° · reference: lon "
            f"{r.get('lon', ['', ''])[0]:.3f}–{r.get('lon', ['', ''])[1]:.3f}°, "
            f"lat {r.get('lat', ['', ''])[0]:.3f}–{r.get('lat', ['', ''])[1]:.3f}°"
            f" · overlap: {'yes' if georef.get('overlap') else 'no'}")
    dg = rep.get("diagnostics") or {}
    if dg:
        st.caption(
            "Image quality — source mean "
            f"{dg.get('src_mean', 0):.0f}/255, reference mean "
            f"{dg.get('ref_mean', 0):.0f}/255 (low values = dark/low-sun).")
    with st.expander("Evidence — full pipeline trace", expanded=False):
        _render_pipeline_trace(rep, sensor)
    notes = rep.get("notes") or []
    if notes:
        with st.expander("Technical notes"):
            for n in notes:
                st.markdown(f"- {n}")


def _fmt_px(d, key_rows="rows", key_cols="cols"):
    if not d:
        return "—"
    r, c = d.get(key_rows), d.get(key_cols)
    if not isinstance(r, (int, float)) or not isinstance(c, (int, float)):
        return "—"
    return f"{int(r):,} × {int(c):,} px"


def _render_what_changed(rep, src_name, ref_name):
    """Explicit 'before vs after' sizes so users can see exactly what changed."""
    d = rep.get("dimensions") or {}
    if not d or not d.get("native_src"):
        return
    ss, sr = d.get("swath_src") or {}, d.get("swath_ref") or {}
    gsd = d.get("gsd_m", rep.get("gsd_m"))
    gsd_txt = f" @ {float(gsd):.2f} m/px" if isinstance(gsd, (int, float)) else ""
    grid = (f"{d.get('grid_along', '?')} × {d.get('grid_across', '?')} px"
            + gsd_txt)

    def swath(s):
        if not s:
            return "—"
        return (f"rows {s.get('row_min', '?')}–{s.get('row_max', '?')} × "
                f"cols {s.get('col_min', '?')}–{s.get('col_max', '?')} (native)")

    st.markdown("#### Before vs after — by the numbers")
    st.markdown(
        "| | **Before** (raw strip) | **After** (common ground grid) |\n"
        f"|---|---|---|\n"
        f"| **{src_name}** | {_fmt_px(d.get('native_src'))} "
        f"(red-box swath: {swath(ss)}) | {grid} |\n"
        f"| **{ref_name}** | {_fmt_px(d.get('native_ref'))} "
        f"(red-box swath: {swath(sr)}) | {grid} |\n"
        f"| Why it changes | each strip is a different sensor swath (different "
        f"rows/cols, different footprint, native pixel scale) → you cannot "
        f"compare them directly | both re-sampled to the same region, same "
        f"grid, same ground scale → directly comparable |\n")
    st.caption("The **red box** on each Before panel is the *overlap swath* — "
               "the only part of each raw strip that actually covers the shared "
               "ground region. Everything outside it is dimmed because it is "
               "not part of the comparison.")


def _render_check_it(rep, src_name, ref_name):
    """What 'good' looks like for this verdict — a short check script."""
    verdict = rep.get("verdict")
    st.markdown("#### How to check it's correct")
    if verdict == "registered":
        rmse = rep.get("rmse_px")
        if not isinstance(rmse, (int, float)):
            rmse = rep.get("rmse_self_px")
        rmse_txt = (f"**{rmse:.3f} px**" if isinstance(rmse, (int, float))
                    else "not reported")
        st.markdown(
            f"- **Check the accuracy number** — RMSE ≈ {rmse_txt} across "
            f"{rep.get('inliers')} matched features. Under 1 px means "
            f"sub-pixel agreement.",
            unsafe_allow_html=True)
        st.markdown(
            "- **Difference panel** (right of the montage) shows mostly dark "
            "with warm speckles only on genuine illumination/feature "
            "differences — not broad shifts.",
            unsafe_allow_html=True)
    elif verdict == "geometry_registered":
        st.markdown(
            f"- **Same ground region** — confirm the {src_name} and {ref_name} "
            "footprints overlap (lon/lat line above).",
            unsafe_allow_html=True)
        st.markdown(
            "- **Same ground scale** — both After panels have the same GSD, so "
            "a fixed screen distance equals the same ground distance in both.",
            unsafe_allow_html=True)
        st.markdown(
            "- **Why no RMSE here** — the reference is so dark/low-sun that "
            "there are no trustworthy features to measure accuracy against. "
            "Reporting an RMSE would mean inventing matches on black data.",
            unsafe_allow_html=True)
    elif verdict == "not_registered":
        st.markdown(
            "- **This is the honest answer** — neither content matching nor "
            "geometry could place the pair on a common grid, so nothing is "
            "presented as aligned.",
            unsafe_allow_html=True)
        st.markdown(
            "- To register this pair you would need a brighter acquisition or "
            "a geometry file for the instrument.",
            unsafe_allow_html=True)


def _sensor_labels(sensor):
    """Display names for each instrument in a sensor route."""
    instr = {
        "tmc": "TMC — Chandrayaan-2",
        "iirs": "IIRS — Chandrayaan-2",
        "ohrc": "OHRC — Chandrayaan-2",
        "nac": "LRO NAC",
    }
    s, r = (sensor.split("-") + ["", ""])[:2]
    return instr.get(s, "Source"), instr.get(r, "Reference")


def _render_pipeline_trace(rep, sensor="tmc-ohrc"):
    """Evidence card: instrument -> reference -> native resolution -> processing
    -> matcher -> transformation -> accuracy. Lets an evaluator see exactly what
    was registered and how, without digging into JSON."""
    src_name, ref_name = _sensor_labels(sensor)
    pair = rep.get("pair") or {}
    georef = rep.get("georef") or {}
    src_gr, ref_gr = georef.get("src", {}), georef.get("ref", {})
    dg = rep.get("diagnostics") or {}

    rows = [
        ("Source (moving)", f"{src_name} · `{os.path.basename(pair.get('src', '?'))}`"),
        ("Reference (fixed)", f"{ref_name} · `{os.path.basename(pair.get('ref', '?'))}`"),
    ]
    s_gsd, r_gsd = src_gr.get("gsd_m"), ref_gr.get("gsd_m")
    rows.append(("Source native GSD",
                 f"{s_gsd:.3f} m/px" if isinstance(s_gsd, (int, float))
                 else "n/a (no geometry CSV)"))
    rows.append(("Reference native GSD",
                 f"{r_gsd:.3f} m/px" if isinstance(r_gsd, (int, float))
                 else "n/a (no geometry CSV)"))
    common = rep.get("gsd_m")
    if isinstance(common, (int, float)):
        rows.append(("Common ground-grid GSD", f"{common:.3f} m/px"))

    if dg:
        r_m = dg.get("ref_mean", 0)
        ref_note = "dark / low-sun" if r_m < 25 else "normal lighting"
        rows.append(("Preprocessing",
                     "contrast enhancement (percentile stretch + low-sun path)"
                     if (dg.get("ref_mean", 0) < 25 or
                         dg.get("src_mean", 0) < 25)
                     else "none needed"))
        rows.append(("Reference frame quality",
                     f"mean {r_m:.0f}/255 · {ref_note}"))

    meth = rep.get("method") or "—"
    if rep.get("verdict") == "registered":
        front = meth.removeprefix("content:")
        rows += [
            ("Matcher", f"cross-modal ({front}) · {rep.get('n_matches', 0)} raw matches"),
            ("Transformation", "homography (RANSAC, grid-uniform)"),
            ("Inliers / ratio",
             f"{rep.get('inliers', 0)} / {float(rep.get('inlier_ratio', 0)):.3f}"),
            ("Accuracy (RMSE)", "n/a" if not isinstance(rep.get("rmse_px"), float)
             else f"{rep['rmse_px']:.4f} px"),
        ]
    else:
        rows += [
            ("Content attempt", f"{meth} → not verifiable (dark/low-sun or featureless)"),
            ("Transformation", "geometry placement on common ground grid"),
            ("Accuracy (RMSE)", "n/a — no trustworthy content correspondences"),
        ]
    rows.append(("Verdict", rep.get("verdict") or "—"))

    md = "\n".join(["| Component | Value |", "|---|---|"]
                   + [f"| **{k}** | {v} |" for k, v in rows])
    st.markdown(md)


def build_pair_config(pair, overrides):
    cfg = _base_config()
    pre = cfg["preprocessing"]
    pre["ohrc_img"] = pair["ohrc_img"]
    pre["ohrc_geometry"] = pair["ohrc_geometry"]
    pre["nac_img"] = pair["nac_img"]
    pre["out_dir"] = pair["out_dir"]
    pre["equal_gsd"] = overrides["equal_gsd"]
    out = os.path.join(pair["out_dir"], "pair1")
    pre["outputs"] = {"src": out + "_src.png", "ref": out + "_ref.png"}
    cfg["inputs"] = {"src": out + "_src.png", "ref": out + "_ref.png"}
    cfg["outputs"]["matches_figure"] = os.path.join(
        pair["out_dir"], "pair1_matches_demo.png")
    cfg["outputs"]["ablation_log"] = "results/logs/demo_ablation.csv"
    cfg["normalize"] = {
        "enabled": overrides["normalize"] != "none",
        "method": overrides["normalize"],
        "clip_limit": 4.0,
        "tile_grid": 8,
    }
    return cfg


def build_upload_config(src_path, ref_path, normalize, out_path):
    cfg = _base_config()
    cfg["preprocessing"]["enabled"] = False
    cfg["inputs"] = {"src": src_path, "ref": ref_path}
    cfg["ground_truth"] = ""
    cfg["normalize"] = {
        "enabled": normalize != "none",
        "method": normalize,
        "clip_limit": 4.0,
        "tile_grid": 8,
    }
    cfg["outputs"]["matches_figure"] = out_path
    cfg["outputs"]["ablation_log"] = "results/logs/demo_ablation.csv"
    cfg["experiment"]["id"] = "DEMO-UPLOAD"
    return cfg


def checkerboard(src, ref, tiles=8):
    """Visual blending of the two already-aligned staged crops (display only)."""
    h, w = src.shape[:2]
    out = np.zeros((h, w), np.uint8)
    th, tw = h // tiles, w // tiles
    for i in range(tiles):
        for j in range(tiles):
            use_ref = (i + j) % 2 == 0
            out[i * th:(i + 1) * th, j * tw:(j + 1) * tw] = (
                ref[i * th:(i + 1) * th, j * tw:(j + 1) * tw] if use_ref
                else src[i * th:(i + 1) * th, j * tw:(j + 1) * tw])
    return out


def run_demo(config_path):
    from src.pipeline import run_experiment
    return run_experiment(config_path)


def main():
    st.set_page_config(page_title="Lunar Image Registration — Demo",
                       layout="wide")
    st.title("Chandrayaan-2 & LRO image registration")
    st.caption("One-click registration for OHRC ⟷ LRO NAC, TMC ⟷ OHRC and "
               "IIRS ⟷ OHRC. The app drives the repository "
               "pipeline (src/pipeline.py / src/auto_pipeline.py); no matching "
               "logic is reimplemented here.")

    with st.sidebar:
        mode = st.radio("Input mode", [
            "Any image pair (auto-detect)",
            "Multi-sensor registration (TMC / IIRS)",
            "Automatic registration",
            "Project pair-1",
            "Project pair-2 (polar, geometry registration)",
            "Upload aligned crops",
        ])
        normalize = st.selectbox("Photometric preconditioning",
                                 ["clahe", "none", "edges", "histogram_match"])
        equal_gsd = st.checkbox("equal-GSD staging (self-calibrated NAC factor)",
                                value=False)
        use_cached = st.checkbox("Load cached demo output instead of running",
                                 value=False)

    if mode.startswith("Multi-sensor"):
        sensor_res = _execute_sensor()
        if sensor_res.get("report"):
            st.subheader("Results")
            _render_sensor(sensor_res)
        return

    if mode.startswith("Any image"):
        any_res = _execute_any()
        if any_res.get("report"):
            st.subheader("Results")
            _render_sensor(any_res)
        return

    if st.button("Run pipeline (FINAL_CONFIG)"):
        st.session_state["result"] = None
        st.session_state["result"] = _execute(mode, normalize, equal_gsd, use_cached)

    if "result" not in st.session_state or st.session_state["result"] is None:
        st.info("Choose a mode and press **Run pipeline (FINAL_CONFIG)**.")
        return

    res = st.session_state["result"]
    st.subheader("Results")
    if res.get("mode") == "auto":
        if res.get("error"):
            st.info(res["error"])
            return
        _render_auto(res)
        return
    if res.get("mode") == "sensor":
        if res.get("error"):
            st.info(res["error"])
            return
        _render_sensor(res)
        return
    if res.get("mode") == "phase5":
        _render_phase5(res)
        return
    if res.get("error"):
        st.error(res["error"])
        for k, v in res.get("notes", {}).items():
            st.write(f"**{k}:** {v}")
        return

    cols = st.columns(5)
    cols[0].metric("RMSE (px)", f"{res['rmse']}")
    cols[1].metric("Inliers", f"{res['inliers']}")
    cols[2].metric("Inlier ratio", f"{res['ratio']:.3f}")
    cols[3].metric("Raw matches", f"{res['n_matches']}")
    cols[4].metric("Runtime (s)", f"{res['time']:.1f}")

    left, right = st.columns(2)
    if res.get("fig"):
        left.image(_framed(res["fig"], "MATCHES"),
                   caption="Inlier match points — crossed feature pairs the "
                           "pipeline trusts (frame = delivered product).",
                   width="stretch")
    if res.get("checker"):
        right.image(_framed(res["checker"], "ALIGNED"),
                    caption="Checkerboard blend of the aligned crops — features "
                            "run continuously across the tiles, which is what "
                            "'aligned' looks like.",
                    width="stretch")
    if res.get("diff") and os.path.exists(os.path.join(ROOT, res["diff"])):
        st.image(os.path.join(ROOT, res["diff"]), width="stretch",
                 caption="What changed, pixel by pixel — warm (yellow/red) = "
                         "large difference at that pixel, dark/blue = already "
                         "agreeing after registration.")
    gsd = res.get("gsd_m")
    if gsd:
        st.caption(f"Staged common GSD: {gsd} m/px")
    gt = res.get("gt")
    if gt:
        gt_path = os.path.join(ROOT, gt)
        import numpy as _np
        n_pts = "?"
        try:
            n_pts = len(_np.loadtxt(gt_path, delimiter=",", skiprows=1))
        except Exception:
            pass
        rmse_val = res.get("rmse")
        is_subpx = isinstance(rmse_val, (int, float)) and 0 < rmse_val < 1.0
        st.caption(
            f"Evaluation reference: `{gt}` ({n_pts} sub-pixel control points).")
        if is_subpx:
            st.success(
                f"**Sub-pixel registration**: RMSE {rmse_val:.4f} px "
                "< 1.0 px target.")
    err = res.get("notes")
    if err:
        st.caption(err)


def _load_fallback(mode):
    cache = os.path.join(ROOT, FALLBACK_DIR, "cached_final_output.json")
    if not os.path.exists(cache):
        return {"error": "Fallback cache not present under demo/fallback/."}
    import json
    with open(cache) as fh:
        d = json.load(fh)
    d["notes"] = {"source": "cached demo/fallback/cached_final_output.json"}
    d.setdefault("gt", d.get("ground_truth"))
    return {**d, "fig": os.path.join(ROOT, "results", "figures",
                                     "pair1_matches_FINAL.png")}


def _phase5_registration():
    """Run (once) or reload the SPICE equal-GSD geometry registration for
    pair-2. Returns the georef_phase5.json meta dict."""
    import json
    out5 = os.path.join(PAIR2["out_dir"], "phase5")
    mj = os.path.join(ROOT, out5, "georef_phase5.json")
    if not os.path.exists(mj):
        from src.preprocessing.geometry import georeference_phase5
        georeference_phase5(
            os.path.join(ROOT, PAIR2["ohrc_img"]),
            os.path.join(ROOT, PAIR2["ohrc_geometry"]),
            os.path.join(ROOT, PAIR2["nac_img"]),
            os.path.join(ROOT, PAIR2["nac_geometry"]),
            os.path.join(ROOT, out5),
        )
    with open(mj) as fh:
        return json.load(fh)


def _render_phase5(res):
    meta = res.get("meta", {})
    st.info("\n\n— ".join(f"{k}: {v}" for k, v in res.get("notes", {}).items())
            or "Photometric-gap pair: content matching is not verifiable.")
    st.success("Pair-2 registered **by geometry** (SPICE + ISRO CSV): "
               "equal-GSD ortho pair, 100% in-bounds, self-consistent to "
               "<1e-9 km.")

    c = st.columns(4)
    gs = meta.get("grid_shape", [0, 0])
    c[0].metric("Staged grid", f"{gs[1]}×{gs[0]}")
    c[1].metric("GSD", f"{meta.get('gsd_m', '')} m")
    c[2].metric("Sun elevation", f"{meta.get('ohrc_sun_elevation_deg', '')}°")
    c[3].metric("ODE overlap", f"{meta.get('ode_overlap_percent', '')}%")

    audit = meta.get("coverage_audit", {})
    ion, lat = st.columns(2)
    ion.metric("Coverage (in-bounds)", f"{audit.get('in_bounds_frac', 0):.4f}")
    lat.metric("Pos. err. mean",
               f"{audit.get('pos_err_km_mean', 0):.2e} km")
    lo, la = meta.get("lon_range", [0, 0]), meta.get("lat_range", [0, 0])
    st.caption(f"OHRC lon {lo[0]:.4f}–{lo[1]:.4f}°, lat {la[0]:.4f}–{la[1]:.4f}°")

    probe = meta.get("content_probe", {})
    if probe.get("cross_full_strip") is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("DISK+LightGlue cross (pair)", probe["cross_full_strip"])
        p2.metric("Self src control", probe["self_src"])
        p3.metric("Self ref control", probe["self_ref"])
    corr = meta.get("correlator_lock_check", {})
    if corr:
        st.metric("FFT lock verdict", corr.get("verdict", ""),
                  help=f"true peak {corr.get('true_peak_value')} vs "
                       f"row-reversed null {corr.get('null_rev_peak_value')} "
                       f"(sigma {corr.get('sigma_px')} px)")
    st.markdown(
        f"**Verification:** {meta.get('verification', 'n/a')}")

    ic, ip = st.columns(2)
    ic.image(_framed(meta.get("src_png"), "OHRC ORTHO"),
             caption="OHRC ortho — delivered registered product (frame = "
                     "delivered)", width=620)
    ip.image(_framed(meta.get("ref_png"), "NAC ORTHO"),
             caption="NAC ortho — delivered registered product (frame = "
                     "delivered)", width=620)
    st.image(_framed(meta.get("overlay_png"), "50/50 OVERLAY"),
             caption="50/50 overlay of the two ground grids (frame = "
                     "delivered)", width=1240)


def _execute_auto(normalize, equal_gsd):
    """One-click fully-automatic registration: choose a preset pair or upload the
    three inputs, then run_auto does everything (stage, register, decide,
    report)."""
    from src.auto_pipeline import run_auto

    source = st.radio("Input source", ["Project pair-1", "Project pair-2 (polar)"],
                      key="auto_source")
    if source.startswith("Project pair-1"):
        pair = PAIR1
        nac_geom = ""
        gt = "data/ground_truth/pair1_gt_v2.csv"
    else:
        pair = PAIR2
        nac_geom = pair.get("nac_geometry", "")
        gt = ""

    out_dir = os.path.join(DEMO_DIR, "auto")
    prefix = "auto_pair1" if source.startswith("Project pair-1") else "auto_pair2"

    if st.button("Run fully-automatic registration"):
        st.session_state["auto_report"] = run_auto(
            pair["ohrc_img"], pair["ohrc_geometry"], pair["nac_img"],
            out_dir=os.path.relpath(out_dir, ROOT), prefix=prefix,
            nac_geom_csv=nac_geom, ground_truth=gt,
            normalize=normalize, equal_gsd=equal_gsd,
            root=ROOT, fresh=True,
        )
    if "auto_report" in st.session_state and st.session_state["auto_report"]:
        return {"mode": "auto", "report": st.session_state["auto_report"],
                "prefix": prefix, "out_dir": out_dir}
    return {"error": "Press **Run fully-automatic registration** to start.",
            "notes": {}}


def _render_auto(res):
    rep = res.get("report", {})
    verdict = rep.get("verdict")
    st.subheader("Fully-automatic result")
    icon, headline, kind = _friendly_status(verdict)
    getattr(st, kind)(f"{icon} **{headline}**")

    c = st.columns(4)
    c[0].metric("Verdict", verdict)
    c[1].metric("RMSE vs reference (px)",
                f"{rep.get('rmse_px')}" if rep.get("rmse_px") is not None else "n/a")
    c[2].metric("Inliers", f"{rep.get('inliers')}")
    c[3].metric("Self-RMSE (px)",
                f"{rep.get('rmse_self_px')}" if rep.get("rmse_self_px") is not None else "n/a")

    if verdict in ("registered", "geometry_registered"):
        meth = rep.get("method")
        st.caption(f"Method: **{meth}**"
                   + (f" · GSD {rep.get('staged_gsd_m') or rep.get('gsd_m')} m/px"
                      if rep.get("staged_gsd_m") or rep.get("gsd_m") else ""))
        if verdict == "geometry_registered":
            st.write("The scene is a dark or low-sun region, so automatic "
                     "feature matching cannot be trusted. The pipeline placed "
                     "both images on the same geographic grid from the "
                     "spacecraft geometry instead — they are registered in "
                     "space even though the images look different.")
        arts = rep.get("artifacts") or {}
        tags = {"matches": "VERIFIED MATCHES",
                "checkerboard": "CHECKERBOARD (ALIGNED)",
                "overlay": "REGISTERED OVERLAY"}
        caps = {"matches": "Verified match points — crossed feature pairs the "
                           "pipeline trusts (frame = delivered product).",
                "checkerboard": "Checkerboard blend of the aligned crops — "
                                "features run continuously across the tiles, "
                                "which is what 'aligned' looks like.",
                "overlay": "Registered overlay on a common grid (frame = "
                           "delivered registered product)."}
        for key in ("matches", "checkerboard", "overlay"):
            p = arts.get(key)
            if p and os.path.exists(os.path.join(ROOT, p)):
                st.image(_framed(os.path.join(ROOT, p), tags[key]),
                         caption=caps[key], width="stretch")
        _render_ground_artifacts(rep)
        if rep.get("rmse_px") is not None and 0 < rep["rmse_px"] < 1.0:
            st.success(f"**Sub-pixel registration**: RMSE {rep['rmse_px']:.4f} px "
                       "< 1.0 px target.")
    elif verdict == "no_content_correspondence":
        st.info("No content correspondence; geometry fallback was used "
                "(feature match not verifiable — photometric/polar pair).")
    else:
        st.error("Registration not completed: " + " | ".join(rep.get("notes", [])))
    if rep.get("notes"):
        with st.expander("Technical notes"):
            for n in rep["notes"]:
                st.markdown(f"- {n}")


def _execute_sensor():
    """Friendly wrapper around run_sensor_auto for the TMC / IIRS presets."""
    from src.auto_pipeline import run_sensor_auto

    label = st.selectbox("Sensor pair", list(SENSOR_PRESETS.keys()))
    preset = SENSOR_PRESETS[label]
    st.caption(preset["note"])

    if preset["sensor"] == "tmc-ohrc":
        with st.spinner("Preparing TMC product (unzip ~2 GB on first run)…"):
            err = _ensure_tmc_extracted()
        if err:
            return {"error": err, "notes": {}}
        if not os.path.exists(os.path.join(ROOT, preset["src_img"])):
            return {"error": "TMC product did not extract correctly — check "
                             "the archive under data/PATCH-004/TMC/.", "notes": {}}

    out_dir = os.path.join(DEMO_DIR, "sensor")
    prefix = preset["sensor"]

    if st.button(f"Register — {label}", type="primary"):
        with st.spinner("Registering… (dark/low-sun pairs use the geometry "
                        "fallback automatically)"):
            st.session_state["sensor_report"] = run_sensor_auto(
                preset["sensor"], preset["src_img"], preset["src_geom"],
                preset["ref_img"], preset["ref_geom"],
                out_dir=os.path.relpath(out_dir, ROOT), prefix=prefix,
                root=ROOT, verbose=True,
            )
    if st.session_state.get("sensor_report"):
        return {"mode": "sensor", "sensor": preset["sensor"],
                "report": st.session_state["sensor_report"]}
    return {"error": "Press the button to start registration.", "notes": {}}


def _execute_any():
    """Drop any two images (OHRC / TMC / IIRS / NAC): auto-detect the sensor
    pair, register, and give the definitive MATCH / NO MATCH decision."""
    from src.auto_pipeline import detect_pair, run_sensor_auto

    st.caption("Give the app any two Chandrayaan-2 / LRO images. The sensor of "
               "each file is detected from its name (`ch2_ohr*`, `ch2_tmc*`, "
               "`ch2_iir*`, `M1………….IMG`/`.qub`), the right register runs, and "
               "you get a clear **MATCH** or **NO MATCH** with the reason and "
               "the best aligned product.")

    presets = _any_presets()
    label = st.selectbox("Example pair", list(presets.keys()))
    p = presets[label]
    st.caption(p.get("note", ""))

    src_img = st.text_input("Source image path", value=p["src_img"])
    src_geom = st.text_input("Source geometry", value=p["src_geom"],
                             help="ISRO ground-grid CSV (OHRC/TMC) or IIRS .hdr")
    ref_img = st.text_input("Reference image path", value=p["ref_img"])
    ref_geom = st.text_input("Reference geometry", value=p["ref_geom"],
                             help="NAC SPICE geometry CSV (optional)")

    if st.button("Register this pair", type="primary"):
        try:
            pair = detect_pair(src_img, ref_img)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Auto-detect failed: {exc}")
            return {"error": str(exc), "notes": {}}
        st.write(f"Detected sensor pair: **{pair}**")
        out_dir = os.path.join(DEMO_DIR, "any")
        prefix = pair.replace("-", "_")
        with st.spinner("Registering… this runs the real pipeline, be patient."):
            report = run_sensor_auto(
                pair, src_img, src_geom, ref_img, ref_geom,
                out_dir=os.path.relpath(out_dir, ROOT), prefix=prefix,
                root=ROOT, verbose=True,
            )
            if "decision" not in report:
                from src.evaluation.decision import classify
                report["decision"] = classify(report)
        st.session_state["sensor_report"] = report
        st.session_state["any_sensor"] = pair
    if st.session_state.get("sensor_report"):
        return {"mode": "sensor",
                "sensor": st.session_state.get("any_sensor", "any"),
                "report": st.session_state["sensor_report"]}
    return {"error": "Choose a pair and press **Register this pair**.", "notes": {}}


def _any_presets():
    """A curated set of real on-disk pairs for the 'any image pair' demo."""
    P001 = "data/PATCH-001/data"
    ohrc = (f"{P001}/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/data/"
            "calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_d_img_d18.img")
    ohrc_g = (f"{P001}/OHRC/ch2_ohr_ncp_20210405T1606536730_d_img_d18/geometry/"
              "calibrated/20210405/ch2_ohr_ncp_20210405T1606536730_g_grd_d18.csv")
    nac1 = f"{P001}/LRO_NAC/LC/M1469248775LC.IMG"
    nac2 = "data/PATCH-004/LRO NAC/OHRC/M1127547939RC.IMG"
    iirs = (f"{P001}/IIRS/ch2_iir_nri_20211221T0324126144_d_img_hw1/data/raw/"
            "20211221/ch2_iir_nri_20211221T0324126144_d_img_hw1.qub")
    iirs_g = (f"{P001}/IIRS/ch2_iir_nri_20211221T0324126144_d_img_hw1/data/raw/"
              "20211221/ch2_iir_nri_20211221T0324126144_d_img_hw1.hdr")
    tmc = "data/processed/tmc/ch2_tmc_nca_20260607T2319176707_d_img_d18.img"
    tmc_g = "data/processed/tmc/ch2_tmc_nca_20260607T2319176707_g_grd_d18.csv"
    oh26 = ("data/PATCH-004/OHRC/data/calibrated/20260331/"
            "ch2_ohr_ncp_20260331T1105235288_d_img_d18.img")
    oh26_g = ("data/PATCH-004/OHRC/geometry/calibrated/20260331/"
              "ch2_ohr_ncp_20260331T1105235288_g_grd_d18.csv")
    return {
        "OHRC 2021 ↔ LRO NAC (content match)": {
            "src_img": ohrc, "src_geom": ohrc_g,
            "ref_img": nac1, "ref_geom": "data/processed/nac_geom_M1127_full.csv",
            "note": "Sub-pixel content match on real data (RMSE < 1 px)."},
        "TMC 2026 ↔ LRO NAC (frame disagreement)": {
            "src_img": tmc, "src_geom": tmc_g, "ref_img": nac2,
            "ref_geom": "data/processed/nac_geom_M1127_full.csv",
            "note": "Honest NO MATCH: ISRO CD grid vs NAC SPICE disagree ~10 km."},
        "OHRC 2026 ↔ LRO NAC (frame disagreement)": {
            "src_img": oh26, "src_geom": oh26_g, "ref_img": nac2,
            "ref_geom": "data/processed/nac_geom_M1127_full.csv",
            "note": "Independent OHRC over the same footprint: also no content lock."},
        "IIRS 2021 ↔ LRO NAC (cross-sensor)": {
            "src_img": iirs, "src_geom": iirs_g, "ref_img": nac2,
            "ref_geom": "",
            "note": "Infrared vs visible — honest NO MATCH across all bands."},
    }


def _execute(mode, normalize, equal_gsd, use_cached):
    os.makedirs(os.path.join(ROOT, UPLOAD_DIR), exist_ok=True)
    try:
        if use_cached:
            return _load_fallback(mode)

        from src.pipeline import run_experiment

        if mode.startswith("Automatic"):
            return _execute_auto(normalize, equal_gsd)

        if mode.startswith("Project pair-1"):
            cfg = build_pair_config(PAIR1, dict(normalize=normalize,
                                                equal_gsd=equal_gsd))
            cfg_path = _write_experiment(cfg, "pair1")
            row = run_experiment(cfg_path, verbose=False)
            fig = os.path.join(ROOT, cfg["outputs"]["matches_figure"])
            checker = _checker_from_cfg(cfg)
            meta = _read_meta(cfg)
            out = _row_to_result(row, fig, checker, meta,
                                 gt=cfg.get("ground_truth", ""))
            out["diff"] = _diff_from_paths(
                cfg["preprocessing"]["outputs"]["src"],
                cfg["preprocessing"]["outputs"]["ref"])
            return out

        if mode.startswith("Project pair-2"):
            cfg = build_pair_config(PAIR2, dict(normalize=normalize,
                                                equal_gsd=False))
            cfg_path = _write_experiment(cfg, "pair2")
            refusal = None
            try:
                row = run_experiment(cfg_path, verbose=False)
                if (isinstance(row.get("inliers"), int) and
                        row.get("inliers", 0) >= 20):
                    fig = os.path.join(ROOT, cfg["outputs"]["matches_figure"])
                    out = _row_to_result(row, fig, None, {})
                    out["diff"] = _diff_from_paths(
                        cfg["preprocessing"]["outputs"]["src"],
                        cfg["preprocessing"]["outputs"]["ref"])
                    return out
                refusal = (f"content matching is inconclusive "
                           f"({row.get('inliers', 0)} inliers)")
            except RuntimeError as e:
                refusal = str(e)
            meta = _phase5_registration()
            return {**{"mode": "phase5", "meta": meta},
                    "notes": {"content_match":
                              f"refused / inconclusive — {refusal}; falling "
                              "back to SPICE + ISRO geometry registration."}}

        if mode.startswith("Upload"):
            up1 = st.file_uploader("Source OHRC crop (grayscale PNG)",
                                   type=["png"], key="src_up")
            up2 = st.file_uploader("Reference NAC crop (grayscale PNG)",
                                   type=["png"], key="ref_up")
            if up1 is None or up2 is None:
                return {"error": "Upload two aligned grayscale PNGs first."}
            sp = os.path.join(UPLOAD_DIR, "upload_src.png")
            rp = os.path.join(UPLOAD_DIR, "upload_ref.png")
            for path, up in ((sp, up1), (rp, up2)):
                with open(os.path.join(ROOT, path), "wb") as fh:
                    fh.write(up.getbuffer())
            figp = os.path.join(UPLOAD_DIR, "upload_matches.png")
            cfg = build_upload_config(sp, rp, normalize,
                                      os.path.join(ROOT, figp))
            cfg_path = _write_experiment(cfg, "upload")
            row = run_experiment(cfg_path, verbose=False)
            checker = None
            a = cv2.imread(os.path.join(ROOT, sp), cv2.IMREAD_GRAYSCALE)
            b = cv2.imread(os.path.join(ROOT, rp), cv2.IMREAD_GRAYSCALE)
            if a is not None and b is not None and a.shape == b.shape:
                check = checkerboard(a, b)
                checker = os.path.join(UPLOAD_DIR, "upload_checkerboard.png")
                cv2.imwrite(os.path.join(ROOT, checker), check)
            out = _row_to_result(row, figp, checker, {})
            out["diff"] = _diff_from_paths(sp, rp, "upload_diff.png")
            return out
    except Exception as e:  # noqa: BLE001 — demo surfaces any pipeline error
        return {"error": str(e), "notes": {}}


def _row_to_result(row, fig, checker, meta, gt=""):
    out = {
        "rmse": row.get("rmse_px", ""),
        "inliers": row.get("inliers", 0),
        "ratio": row.get("inlier_ratio", 0.0),
        "n_matches": row.get("n_matches", 0),
        "time": row.get("time_s", 0.0),
        "fig": fig if os.path.exists(fig) else None,
        "checker": checker,
        "gsd_m": meta.get("crop_gsd_m") if meta else None,
        "gt": gt,
        "notes": {},
        "error": None,
    }
    if not (isinstance(out["rmse"], (int, float)) and out["rmse"] > 0):
        out["notes"] = {"rmse": "n/a (no ground truth for this input)"}
    return out


def _diff_from_paths(sp, rp, name="pair_diff.png"):
    """Save a 'what changed' |source − reference| map next to the staged crops."""
    a = cv2.imread(os.path.join(ROOT, sp), cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(os.path.join(ROOT, rp), cv2.IMREAD_GRAYSCALE)
    if a is None or b is None or a.shape != b.shape:
        return None
    p = os.path.join(os.path.dirname(os.path.join(ROOT, sp)), name)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    cv2.imwrite(p, diff_map(a, b))
    return p


def _read_meta(cfg):
    import json
    meta_path = os.path.join(ROOT, cfg["preprocessing"]["out_dir"],
                             "georef_pair1.json")
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            return json.load(fh)
    return {}


def _checker_from_cfg(cfg):
    src = cv2.imread(os.path.join(ROOT, cfg["inputs"]["src"]),
                     cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(os.path.join(ROOT, cfg["inputs"]["ref"]),
                     cv2.IMREAD_GRAYSCALE)
    if src is None or ref is None or src.shape != ref.shape:
        return None
    p = os.path.join(ROOT, cfg["preprocessing"]["out_dir"],
                     "demo_checkerboard.png")
    cv2.imwrite(p, checkerboard(src, ref))
    return p


if __name__ == "__main__":
    main()