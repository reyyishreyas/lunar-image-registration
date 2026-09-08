"""Shared change-highlight visuals used across every Streamlit surface.

PS 26166 wants the registered product + its match points delivered in a way
everyone can verify. These helpers give one consistent language everywhere:

  * ``frame_img``  — red frame + tag around any product panel ("REGISTERED",
    "MATCHES", …) so on-screen every delivered image is tagged at a glance.
  * ``diff_map``   — pixel-wise |source − reference| difference map (TURBO), the
    "what actually changed" proof for aligned pairs.
  * ``highlight_box`` — red rectangle around a region of interest (overlap
    swath etc.).

Everything is pure cv2/numpy and agnostic to sensor or mode, so the sensor,
automatic, pair-1/pair-2 and phase-5 views all share the same annotation look.
"""

from __future__ import annotations

import cv2
import numpy as np

RED = (0, 0, 255)
_WHITE = (255, 255, 255)
_BLACK = (0, 0, 0)


def _bgr(gray_like):
    """coerce a 2-D gray or 3-D BGR input into a writable BGR uint8 copy."""
    arr = gray_like
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(arr) if arr.ndim == 3 else arr


def _font_metrics(text, fs, th):
    (tw, thb), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
    return tw, thb


def frame_img(gray_like, tag="REGISTERED"):
    """Thick red frame + tag. Mirrors the sensor before/after annotations so the
    whole app reads the same way."""
    bgr = _bgr(gray_like)
    h, w = bgr.shape[:2]
    t = max(4, int(min(h, w) / 30))
    cv2.rectangle(bgr, (0, 0), (w - 1, h - 1), RED, t)
    glow_t = max(1, t // 3)
    cv2.rectangle(bgr, (t + glow_t, t + glow_t),
                  (w - 1 - t - glow_t, h - 1 - t - glow_t), RED, glow_t)
    fs = max(0.7, min(h, w) / 420)
    th = max(2, int(fs * 3.2))
    tw, thb = _font_metrics(tag, fs, th)
    pad = max(6, t // 2)
    cv2.rectangle(bgr, (pad - 2, pad - 2), (pad + tw + 10, pad + thb + 6),
                  _BLACK, -1)
    cv2.putText(bgr, tag, (pad + 8, pad + thb + 2), cv2.FONT_HERSHEY_SIMPLEX,
                fs, _WHITE, th, cv2.LINE_AA)
    cv2.putText(bgr, tag, (pad + 8 + max(1, th // 6), pad + thb + 2),
                cv2.FONT_HERSHEY_SIMPLEX, fs, RED, max(1, th // 5), cv2.LINE_AA)
    return bgr


def highlight_box(gray_like, box, tag=None):
    """Red rectangle around ``box=(x0,y0,x1,y1)``; optionally dims everything
    outside it (so the change pops) and adds a ``tag`` label."""
    bgr = _bgr(gray_like)
    if box is None:
        return bgr
    x0, y0, x1, y1 = box
    h, w = bgr.shape[:2]
    inside = np.zeros((h, w), bool)
    inside[max(0, y0):min(h, y1 + 1), max(0, x0):min(w, x1 + 1)] = True
    dimmed = bgr[~inside]
    bgr[~inside] = (dimmed * 0.35).astype(np.uint8)
    t = max(4, int(min(h, w) / 30))
    cv2.rectangle(bgr, (x0, y0), (x1, y1), RED, t)
    if tag:
        fs = max(0.7, min(h, w) / 420)
        th = max(2, int(fs * 3.2))
        tw, thb = _font_metrics(tag, fs, th)
        tx = max(2, min(x0, max(2, w - tw - 8)))
        ty = max(thb + 6, y0 - thb - 10)
        cv2.rectangle(bgr, (tx - 8, ty - thb - 8), (tx + tw + 8, ty + 6),
                      _BLACK, -1)
        cv2.putText(bgr, tag, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, fs, _WHITE,
                    th, cv2.LINE_AA)
        cv2.putText(bgr, tag, (tx + max(1, th // 6), ty),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, RED, max(1, th // 5),
                    cv2.LINE_AA)
    return bgr


def diff_map(a, b):
    """Normalised |a − b| difference (TURBO colormap) of two equal-shape uint8
    frames; ``None`` if the shapes differ."""
    if a.shape != b.shape:
        return None
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).astype(np.uint8)
    if d.max() > d.min():
        lo, hi = np.percentile(d, 2), np.percentile(d, 98)
        d = np.clip((d.astype(np.float32) - lo) / max(hi - lo, 1e-6) * 255, 0,
                    255).astype(np.uint8)
    return cv2.applyColorMap(d, cv2.COLORMAP_TURBO)