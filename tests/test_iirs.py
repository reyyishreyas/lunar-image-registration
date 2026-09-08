"""Unit tests for Phase 9 Chandrayaan-2 IIRS (ENVI hyperspectral) support.

Verifies the ENVI header parser, the band-selective BSQ reader (memory-
bounded), representative-band selection, and uint8 normalisation — all on a
synthetic ENVI product matching the real IIRS header (256 bands, BSQ, int16).
"""
import os

import numpy as np
import pytest

from src.detection.iirs import (
    parse_envi_header,
    read_envi,
    choose_representative_bands,
    as_u8,
)


def _write_envi(tmp_path, lines=4, samples=5, nbands=3, dtype=2):
    hdr = tmp_path / "t.hdr"
    hdr.write_text(
        f"ENVI\nsamples = {samples}\nlines = {lines}\nbands = {nbands}\n"
        "header offset = 0\nfile type = ENVI Standard\n"
        f"data type = {dtype}\ninterleave = bsq\nbyte order = 0\n")
    arr = np.zeros((nbands, lines, samples), np.int16)
    for b in range(nbands):
        arr[b] = b * 100 + np.arange(lines * samples).reshape(lines, samples)
    qub = tmp_path / "t.qub"
    arr.tofile(qub)
    return str(qub), str(hdr)


def test_parse_envi_header(tmp_path):
    hdr = tmp_path / "h.hdr"
    hdr.write_text("ENVI\nsamples = 250\nlines = 12620\nbands = 256\n"
                   "data type = 2\ninterleave = bsq\n")
    meta = parse_envi_header(str(hdr))
    assert meta["samples"] == "250"
    assert meta["bands"] == "256"
    assert meta["interleave"] == "bsq"


def test_read_envi_single_band(tmp_path):
    qub, hdr = _write_envi(tmp_path)
    r = read_envi(qub, hdr, bands=[1])
    assert r["data"].shape == (1, 4, 5)
    assert (r["data"][0] == 100 + np.arange(20).reshape(4, 5)).all()


def test_read_envi_multi_band(tmp_path):
    qub, hdr = _write_envi(tmp_path, nbands=3)
    r = read_envi(qub, hdr, bands=[0, 2])
    assert r["data"].shape == (2, 4, 5)
    assert (r["data"][1] == 200 + np.arange(20).reshape(4, 5)).all()


def test_read_envi_memmap(tmp_path):
    qub, hdr = _write_envi(tmp_path, nbands=2)
    r = read_envi(qub, hdr, bands=[0, 1], memmap=True)
    assert r["data"].shape == (2, 4, 5)
    assert int(r["data"][1][0, 0]) == 100


def test_read_envi_missing_hdr_derived(tmp_path):
    qub, hdr = _write_envi(tmp_path, nbands=2)
    r = read_envi(qub, bands=[0])  # hdr derived from qub name
    assert r["data"].shape == (1, 4, 5)


def test_choose_representative_bands_spread():
    from src.detection.iirs import parse_envi_header
    meta = {"bands": "256"}
    bands = choose_representative_bands(meta, n=4)
    assert len(bands) == 4 and len(set(bands)) == 4
    assert 0 <= min(bands) and max(bands) < 256


def test_choose_representative_bands_small():
    bands = choose_representative_bands({"bands": "2"}, n=4)
    assert len(bands) == 2


def test_as_u8():
    a = np.array([[1000, 1000], [900, 950]], np.int16)
    out = as_u8(a)
    assert out.dtype == np.uint8
    assert out.max() > 0
