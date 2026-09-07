"""SuperPoint detector adapter wrapping the official SuperGluePretrainedNetwork
repository cloned into third_party/.

SuperGlue needs more than keypoints/descriptors (it also consumes the images and
per-keypoint scores). To keep the pipeline's detector/matcher signature
unchanged, this module caches the last detected frame (image, keypoints, scores,
descriptors) per side so the matcher can reconstruct the exact tensors.
"""

import sys
from pathlib import Path

_REPO = (Path(__file__).resolve().parents[2]
         / "third_party" / "SuperGluePretrainedNetwork").resolve()
_IMPORTED = False

_cache = {"src": None, "ref": None}


def _ensure_imported():
    global _IMPORTED
    if _IMPORTED:
        return
    if str(_REPO) not in sys.path:
        sys.path.insert(0, str(_REPO))
    _IMPORTED = True


def _fetch_model(cfg):
    import torch  # noqa: PLC0415

    from models.superpoint import SuperPoint  # noqa: PLC0415

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = SuperPoint(cfg).eval().to(dev)
    return model, dev


def detect_superpoint(image, verbose=False, **kwargs):
    """SuperPoint detector. Returns (list[cv2.KeyPoint], float32 (N, 256) descriptors)."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    _ensure_imported()
    base = {
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": 1024,
        "remove_borders": 4,
    }
    base.update({k: v for k, v in kwargs.items() if k != "name"})
    model, dev = _fetch_model(base)

    frame = (np.asarray(image, np.float32) / 255.0)
    tensor = torch.from_numpy(frame)[None, None].to(dev)
    out = model({"image": tensor})
    kp_np = out["keypoints"][0].detach().cpu().numpy()  # (N, 2) x,y
    scores = out["scores"][0].detach().cpu().numpy()
    des = out["descriptors"][0].detach().cpu().numpy().transpose(1, 0)  # (N, 256)

    slot = "src" if _cache["src"] is None else "ref"
    _cache[slot] = {"image": image, "kp": kp_np, "scores": scores, "des": des}
    if verbose:
        print(f"  SuperPoint keypoints [{slot}]: {kp_np.shape[0]}")

    kp = [cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in kp_np]
    return kp, np.ascontiguousarray(des, np.float32)