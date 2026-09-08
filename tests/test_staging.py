"""Unit tests for equal-GSD staging + photometric preconditioning (Restart Phase 1).

Covers the pure helpers: H-based working-set scale, NAC-factor refinement,
NAC geometry CSV seed, and the Sobel `edges` preconditioner. End-to-end
staging effects are captured in the ablation runs, not here.
"""
import json

import cv2
import numpy as np
import pytest

from src.preprocessing.georeference import (
    ws_scale, refine_ws_factor, _nac_geom_rows, _nac_gsd_seed,
    NAC_WS_FACTOR,
)
from src.preprocessing.normalize import apply_edges
from src.preprocessing.staging import stage_pair, read_pair_meta, apply_precondition
from src.preprocessing.georeference import _crop_gsd_metres


def test_ws_scale_identity():
    H = np.eye(3)
    assert ws_scale(H) == pytest.approx(1.0)


def test_ws_scale_anisotropic():
    H = np.zeros((3, 3))
    H[0, 0], H[1, 1], H[2, 2] = 1.5, 2.0, 1.0
    assert ws_scale(H) == pytest.approx(0.5 * (1.5 + 2.0))


def test_refine_ws_factor_converges_to_locked_gdsd():
    fac = 8
    scale = 12.5 / 8.0  # nac ws gsd should be 12.5 -> equal at fac 12.5
    assert refine_ws_factor(fac, scale) == 12
    assert refine_ws_factor(12, 1.0) == 12  # locked


def test_refine_ws_factor_bounds():
    assert refine_ws_factor(1, 0.01) == 1
    assert refine_ws_factor(200, 10.0) == 256


def test_apply_edges_shape_dtype():
    rng = np.random.default_rng(5)
    img = (rng.random((64, 48)) * 255).astype(np.uint8)
    out = apply_edges(img)
    assert out.shape == img.shape and out.dtype == np.uint8
    assert out.min() == 0 and out.max() == 255


def test_apply_edges_responds_to_gradient():
    flat = np.full((40, 40), 128, np.uint8)
    out = apply_edges(flat)
    assert out.min() == 0 and out.max() == 0
    edge = np.zeros((40, 40), np.uint8)
    edge[:, 20:] = 255
    out2 = apply_edges(edge)
    col = out2[:, 20].astype(np.int16).astype(np.float32)
    assert col.max() > 0


def test_nac_geom_rows_and_seed(tmp_path):
    p = tmp_path / "nac.txt"
    # 5 lines x 5 samples, 0.001 deg/sample ~ 0.9-1.0 m (anisotropic resampled grid)
    rows = []
    for ln in range(5):
        for sm in range(5):
            rows.append(f"{297.0 - 0.001*sm:.6f} {8.0 - 0.002*ln:.6f} 1737400.0 {ln} {sm}\n")
    p.write_text("".join(rows))
    lo, la, ln, sm = _nac_geom_rows(str(p))
    assert len(lo) == 25 and ln[0] == 0.0 and sm[-1] == 4.0
    seed = _nac_gsd_seed(lo, la, ln, sm, gsd_ohrc_ws=10.0)
    assert seed >= 1 and seed <= 256


def test_seed_falls_back_without_geometry():
    assert _nac_gsd_seed([], [], [], [], 10.0) == NAC_WS_FACTOR
    assert _nac_gsd_seed([297.0], [8.0], [0.0], [0.0], 10.0) == NAC_WS_FACTOR


def _write_staged(tmp_path, out_dir, prefix="pair1"):
    od = tmp_path / out_dir
    od.mkdir(parents=True, exist_ok=True)
    src = np.full((64, 64), 100, np.uint8)
    ref = np.full((64, 64), 140, np.uint8)
    sp = od / f"{prefix}_src.png"
    rp = od / f"{prefix}_ref.png"
    cv2.imwrite(str(sp), src)
    cv2.imwrite(str(rp), ref)
    meta = {"crop_gsd_m": 0.99, "prefix": prefix, "ws_factor_iters": None}
    with open(od / f"georef_{prefix}.json", "w") as fh:
        json.dump(meta, fh)
    return sp, rp


def test_read_pair_meta_cached(tmp_path):
    sp, _ = _write_staged(tmp_path, "proc")
    pre = {"out_dir": str(sp.parent), "prefix": "pair1"}
    meta = read_pair_meta(pre, root="")
    assert meta["crop_gsd_m"] == 0.99


def test_stage_pair_cached_and_precondition(tmp_path):
    sp, _ = _write_staged(tmp_path, "proc")
    pre = {
        "out_dir": str(sp.parent), "prefix": "pair1", "equal_gsd": False,
        "nac_geom_csv": "", "ohrc_img": "x", "ohrc_geometry": "x",
        "nac_img": "x", "crop_px": 1024,
        "outputs": {"src": str(sp.parent / "pair1_src.png"),
                    "ref": str(sp.parent / "pair1_ref.png")},
    }
    st = stage_pair(pre, {"enabled": True, "method": "clahe", "clip_limit": 4.0,
                          "tile_grid": 8}, root="")
    assert st["src"].shape == (64, 64) and st["src"].dtype == np.uint8
    assert st["gsd_m"] == 0.99
    assert st["norm_label"] == "clahe:4.0"
    st2 = stage_pair(pre, {}, root="")
    assert st2["norm_label"] == "raw"


def test_stage_pair_runs_georef_when_missing(tmp_path, monkeypatch):
    od = tmp_path / "proc"
    pre = {
        "out_dir": str(od), "prefix": "pair1", "equal_gsd": False,
        "nac_geom_csv": "", "ohrc_img": "ohrc.img", "ohrc_geometry": "g.csv",
        "nac_img": "nac.img", "crop_px": 40,
        "outputs": {"src": str(od / "pair1_src.png"),
                    "ref": str(od / "pair1_ref.png")},
    }

    def fake_georef(**kw):
        od = kw["out_dir"]
        if isinstance(od, str):
            import os as _os
            _os.makedirs(od, exist_ok=True)
            from src.preprocessing.georeference import OHRC_WIDTH
            cv2.imwrite(_os.path.join(od, "pair1_src.png"), np.full((40, 40), 90, np.uint8))
            cv2.imwrite(_os.path.join(od, "pair1_ref.png"), np.full((40, 40), 90, np.uint8))
        return {"crop_gsd_m": 0.97, "prefix": kw["prefix"]}

    monkeypatch.setattr("src.preprocessing.georeference.georeference_pair", fake_georef)
    st = stage_pair(pre, {}, root="")
    assert st["gsd_m"] == 0.97
    assert st["ref"].shape == (40, 40)


def test_apply_precondition_unknown_raises():
    a = np.zeros((8, 8), np.uint8)
    with pytest.raises(ValueError):
        apply_precondition(a, a, {"enabled": True, "method": "nope"})


def test_crop_gsd_metres_from_grid():
    scan = np.array([0.0, 100.0, 200.0])
    px = np.array([0.0, 50.0, 100.0])
    lon = np.zeros((3, 3)) + 296.0 - np.add.outer(np.zeros(3), np.arange(3) * 0.002)
    lat = np.zeros((3, 3)) + 7.0 + np.add.outer(np.arange(3) * 0.001, np.zeros(3))
    g = _crop_gsd_metres(px, scan, lon, lat)
    assert g == pytest.approx(0.5 * (1.11 + 4.406895468662732), rel=1e-3)