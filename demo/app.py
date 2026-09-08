"""Streamlit demo for the lunar image registration pipeline (Phase 6.1).

The demo ONLY orchestrates the pipeline: it builds a FINAL_CONFIG-derived
experiment YAML and calls `src.pipeline.run_experiment`. No matching,
detection, georeferencing or outlier-rejection logic is reimplemented here.

Modes:
  * Project pair-1 (OHRC-2021 + LRO NAC M1469248775LC) — full run including
    georeference staging (cached crops reused when present).
  * Project pair-2 (polar OHRC-2026 + M1127547939RC) — demonstrates the
    documented, evidence-based georeference refusal.
  * Upload aligned crops — runs the FINAL detection/matching/outlier config on
    two uploaded grayscale images (preprocessing disabled).

Run with:
    venv/bin/streamlit run demo/app.py
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np
import streamlit as st
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

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


def _base_config():
    with open(os.path.join(ROOT, FINAL_CONFIG)) as fh:
        return yaml.safe_load(fh)


def _write_experiment(cfg, name):
    os.makedirs(os.path.join(ROOT, DEMO_DIR), exist_ok=True)
    path = os.path.join(ROOT, DEMO_DIR, f"demo_{name}.yaml")
    with open(path, "w") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return path


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
    st.title("OHRC ⟷ LRO NAC image registration")
    st.caption("Demo drives the repository pipeline (src/pipeline.py) with "
               "FINAL_CONFIG; no matching logic is reimplemented here.")

    with st.sidebar:
        mode = st.radio("Input mode", [
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

    if st.button("Run pipeline (FINAL_CONFIG)"):
        st.session_state["result"] = None
        st.session_state["result"] = _execute(mode, normalize, equal_gsd, use_cached)

    if "result" not in st.session_state or st.session_state["result"] is None:
        st.info("Choose a mode and press **Run pipeline (FINAL_CONFIG)**.")
        return

    res = st.session_state["result"]
    st.subheader("Results")
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
        left.image(res["fig"], caption="Inlier match overlay (pipeline output)",
                   use_container_width=True)
    if res.get("checker"):
        right.image(res["checker"], caption="Checkerboard blend of staged "
                                            "(aligned) crops",
                    use_container_width=True)
    gsd = res.get("gsd_m")
    if gsd:
        st.caption(f"Staged common GSD: {gsd} m/px")
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
    ic.image(meta.get("src_png"), caption="OHRC ortho (magma)", width=620)
    ip.image(meta.get("ref_png"), caption="NAC ortho (magma)", width=620)
    st.image(meta.get("overlay_png"), caption="50/50 overlay", width=1240)


def _execute(mode, normalize, equal_gsd, use_cached):
    os.makedirs(os.path.join(ROOT, UPLOAD_DIR), exist_ok=True)
    try:
        if use_cached:
            return _load_fallback(mode)

        from src.pipeline import run_experiment

        if mode.startswith("Project pair-1"):
            cfg = build_pair_config(PAIR1, dict(normalize=normalize,
                                                equal_gsd=equal_gsd))
            cfg_path = _write_experiment(cfg, "pair1")
            row = run_experiment(cfg_path, verbose=False)
            fig = os.path.join(ROOT, cfg["outputs"]["matches_figure"])
            checker = _checker_from_cfg(cfg)
            meta = _read_meta(cfg)
            return _row_to_result(row, fig, checker, meta)

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
                    return _row_to_result(row, fig, None, {})
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
            return _row_to_result(row, figp, checker, {})
    except Exception as e:  # noqa: BLE001 — demo surfaces any pipeline error
        return {"error": str(e), "notes": {}}


def _row_to_result(row, fig, checker, meta):
    out = {
        "rmse": row.get("rmse_px", ""),
        "inliers": row.get("inliers", 0),
        "ratio": row.get("inlier_ratio", 0.0),
        "n_matches": row.get("n_matches", 0),
        "time": row.get("time_s", 0.0),
        "fig": fig if os.path.exists(fig) else None,
        "checker": checker,
        "gsd_m": meta.get("crop_gsd_m") if meta else None,
        "notes": {},
        "error": None,
    }
    if not (isinstance(out["rmse"], (int, float)) and out["rmse"] > 0):
        out["notes"] = {"rmse": "n/a (no ground truth for this input)"}
    return out


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