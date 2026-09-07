"""RIFT2 (phase congruency) detector wrapper.

Loads the vendored third-party implementation in third_party/RIFT2-python.
The third-party repo exposes its public API as the ``src`` package, which
collides with our own ``src`` package, so the import is performed inside a
contained loader that temporarily resolves ``src`` to the third-party tree,
captures the class references, and then restores the original ``sys.path``
and ``sys.modules`` state for ``src`` / ``src.*``.
"""

import sys
import types
from pathlib import Path

_RIFT2_LOADED = None


def _load_rift2():
    global _RIFT2_LOADED
    if _RIFT2_LOADED is not None:
        return _RIFT2_LOADED

    root = (
        Path(__file__).resolve().parents[2] / "third_party" / "RIFT2-python"
    ).resolve()
    src_dir = root / "src"
    preserved = {k: sys.modules[k] for k in list(sys.modules)
                 if k == "src" or k.startswith("src.")}
    fake_src = types.ModuleType("src")
    fake_src.__path__ = [str(src_dir)]
    fake_src.__package__ = "src"
    try:
        sys.modules["src"] = fake_src
        from src.RIFT2 import RIFT2 as _rift2_cls  # noqa: PLC0415
        from src.matcher_functions import match_keypoints_nn  # noqa: PLC0415

        _RIFT2_LOADED = (_rift2_cls, match_keypoints_nn)
    finally:
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        sys.modules.update(preserved)
    return _RIFT2_LOADED


def detect_rift2(image, verbose=False, **kwargs):
    """Run RIFT2 feature detection + description on a single image.

    Returns (keypoints, descriptors) with the same shape contract as the
    other pipeline detectors: a list of cv2.KeyPoint and a float32 descriptor
    matrix (N x 216).
    """
    import cv2
    import numpy as np

    cls, _ = _load_rift2()
    r2 = cls(**kwargs)
    img = r2._convert_to_grayscale(image)  # noqa: SLF001 (third-party API)
    key1, m1, eo1 = r2.feature_detection(img)
    kpts = r2.compute_orientation(key1, m1)
    des = r2.feature_description(img.shape, eo1, kpts).T
    if verbose:
        print(f"  RIFT2 keypoints detected: {kpts.shape[1]}")
    kp = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kpts.T]
    if des.dtype != np.float32:
        des = des.astype(np.float32)
    return kp, des


def match_rift2_nn(kp1, des1, kp2, des2, lowes_ratio=0.95, mutual=False,
                   verbose=False):
    """Match RIFT2 descriptors with the third-party NN + ratio-test matcher."""
    _, match_keypoints_nn = _load_rift2()
    _p1, _p2, matches = match_keypoints_nn(
        des1, des2, kp1, kp2, lowes_ratio=lowes_ratio, mutual=mutual)
    if verbose:
        print(f"  RIFT2 NN matches (ratio {lowes_ratio}, mutual={mutual}): {len(matches)}")
    return matches