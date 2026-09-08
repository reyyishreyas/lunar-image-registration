"""Unit tests for equal-GSD staging + photometric preconditioning (Restart Phase 1).

Covers the pure helpers: H-based working-set scale, NAC-factor refinement,
NAC geometry CSV seed, and the Sobel `edges` preconditioner. End-to-end
staging effects are captured in the ablation runs, not here.
"""
import numpy as np
import pytest

from src.preprocessing.georeference import (
    ws_scale, refine_ws_factor, _nac_geom_rows, _nac_gsd_seed,
    NAC_WS_FACTOR,
)
from src.preprocessing.normalize import apply_edges


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