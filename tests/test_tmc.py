"""Unit tests for Phase 9 Chandrayaan-2 TMC support.

Covers the generic pushbroom reader (width inference from the geometry CSV,
memmap), the ground grid reader, and GSD estimation on a synthetic ISRO-style
geometry CSV (so the TMC path is verified without needing the 2.2 GB products).
"""
import os

import numpy as np
import pytest

from src.preprocessing.tmc import (
    infer_across_track_width,
    read_ch2_raw,
    read_ground_grid,
    _gsd_m,
    _to_u8,
    TMC_WIDTH,
    OHRC_WIDTH,
)


def _write_geom_csv(tmp_path, width, n_scan, stride=100, lon0=295.0, lat0=18.0):
    """Write an ISRO-style geometry CSV sampled every `stride` native px/scan.

    The scene is a flat plane so ground spacing is linear and well-defined. The
    final pixel (width-1) is always included, matching the ISRO convention that
    the geometry grid spans the full across-track width.
    """
    path = tmp_path / "geom.csv"
    rows = []
    pix_grid = sorted(set(list(range(0, width, stride)) + [width - 1]))
    for s in range(0, n_scan, stride):
        for p in pix_grid:
            # each native px advances lon by 1e-4 deg; each scan by 1e-4 deg
            lon = lon0 + p * 1e-4
            lat = lat0 + s * 1e-4
            rows.append(f"{lon:.6f},{lat:.6f},{p},{s}\n")
    path.write_text("Longitude,Latitude,Pixel,Scan\n" + "".join(rows))
    return path


def test_infer_width_from_csv(tmp_path):
    g = _write_geom_csv(tmp_path, width=4000, n_scan=2000)
    assert infer_across_track_width(str(g)) == 4000


def test_read_ground_grid_shapes(tmp_path):
    g = _write_geom_csv(tmp_path, width=2000, n_scan=2500)
    pix, scan, lon, lat = read_ground_grid(str(g))
    assert pix[0] == 0 and scan[0] == 0
    assert lon.shape == lat.shape
    assert lon.shape[0] == len(scan) and lon.shape[1] == len(pix)


def test_gsd_m_reasonable(tmp_path):
    g = _write_geom_csv(tmp_path, width=4000, n_scan=1200)
    pix, scan, lon, lat = read_ground_grid(str(g))
    gsd = _gsd_m(pix, scan, lon, lat)
    # native stride is 100; plane spacing ~= 1e-4 deg ~ 11 m -> non-trivial
    assert np.isfinite(gsd) and gsd > 1.0 and gsd < 1000.0


def test_gsd_m_accounts_for_stride(tmp_path):
    # same plane but different sampling stride should give ~same per-pixel GSD
    paths = [_write_geom_csv(tmp_path, width=2000, n_scan=800, stride=50),
             _write_geom_csv(tmp_path, width=2000, n_scan=800, stride=200)]
    vals = []
    for p in paths:
        pix, scan, lon, lat = read_ground_grid(str(p))
        vals.append(_gsd_m(pix, scan, lon, lat))
    assert abs(vals[0] - vals[1]) / vals[0] < 0.3


def test_read_ch2_raw_infers_width(tmp_path):
    width, n_scan = 4000, 100
    raw = (np.arange(width * n_scan, dtype=np.int64) % 256
           ).astype(np.uint8).reshape(n_scan, width)
    img = tmp_path / "raw.img"
    img.write_bytes(raw.tobytes())
    g = _write_geom_csv(tmp_path, width=width, n_scan=n_scan * 2)
    arr = read_ch2_raw(str(img), str(g))
    assert arr.shape == (n_scan, width)
    assert arr.dtype == np.uint8


def test_read_ch2_raw_memmap(tmp_path):
    width, n_scan = 1000, 60
    raw = np.zeros((n_scan, width), np.uint8)
    img = tmp_path / "m.img"
    img.write_bytes(raw.tobytes())
    g = _write_geom_csv(tmp_path, width=width, n_scan=n_scan * 2)
    mm = read_ch2_raw(str(img), str(g), memmap=True)
    assert mm.shape == (n_scan, width)
    assert mm[0, 0] == 0


def test_to_u8_preserves_uint8():
    a = np.zeros((10, 10), np.uint8)
    assert _to_u8(a) is a or _to_u8(a).dtype == np.uint8
