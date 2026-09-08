"""Unit tests for shared change-highlight visuals (src/visuals.py)."""
import numpy as np
import pytest
import cv2

from src.visuals import RED, frame_img, highlight_box, diff_map


def _img(h=200, w=300, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w), dtype=np.uint8)


def _red_count(bgr):
    return int(((bgr[:, :, 2].astype(int) > 150)
                & (bgr[:, :, 0].astype(int) < 60)
                & (bgr[:, :, 1].astype(int) < 60)).sum())


def test_frame_img_draws_red_border():
    bgr = frame_img(_img(), tag="DELIVERED")
    assert bgr.ndim == 3 and bgr.shape[2] == 3
    assert _red_count(bgr) > 0
    # border must reach the very edge (frame, not interior-only)
    assert tuple(bgr[1, 1]) == RED


def test_frame_img_preserves_color_figures():
    color = np.zeros((64, 64, 3), np.uint8)
    color[:, :, 2] = 200  # red-ish figure
    out = frame_img(color, tag="X")
    assert _red_count(out) > 0


def test_highlight_box_dims_outside():
    prev = _img()
    h, w = prev.shape
    box = (w // 4, h // 4, 3 * w // 4, 3 * h // 4)
    out = highlight_box(prev, box, tag="SWATH")
    assert _red_count(out) > 0
    # interior brightness stays high while outside is dimmed
    interior = int(out[h // 2, w // 2].mean())
    outside = int(out[2, 2].mean())
    assert interior > outside


def test_diff_map_identical_is_dark_different_is_bright():
    a = _img(seed=3)
    zero = np.zeros_like(a)
    same = diff_map(a, a)
    assert same.shape == (a.shape[0], a.shape[1], 3)
    assert same.mean() < 50, "identical frames must produce a near-dark diff"
    diff = diff_map(a, zero)
    assert diff.mean() > same.mean() + 50, "different frames must light up"


def test_diff_map_shape_mismatch_is_none():
    assert diff_map(_img(100, 100), _img(50, 50)) is None