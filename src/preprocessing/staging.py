"""Single ground-resolved staging entry point (Restart Phase 1).

All matching stages consume the pair exclusively from `stage_pair()`: in one
call the OHRC<->NAC pair is georeferenced (if not already staged), loaded at a
common ground resolution, and photometric preconditioning (`none | clahe |
edges | histmatch | gamma_shadow`) is applied. Downstream detection/matching
never re-derives GSD or re-reads rasters; the returned record carries the
staged GSD and georeference metadata so any stage can reason in ground units.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


def apply_precondition(src, ref, ncfg):
    """Apply the config's photometric preconditioning to the staged pair.

    Args:
        src, ref: uint8 grayscale crops.
        ncfg: config dict under top-level key "normalize" (empty if unset).

    Returns:
        (normalized_src, normalized_ref, label): label is a short string for the
        ablation row's preproc column.
    """
    if not ncfg.get("enabled", False):
        return src, ref, "raw"
    method = ncfg.get("method", "")
    if method == "histogram_match":
        from src.preprocessing.normalize import histogram_match
        return histogram_match(src, ref), ref, "histmatch"
    if method == "clahe":
        from src.preprocessing.normalize import apply_clahe
        clip = float(ncfg.get("clip_limit", 2.0))
        tile = int(ncfg.get("tile_grid", 8))
        return (apply_clahe(src, clip_limit=clip, tile_grid=tile),
                apply_clahe(ref, clip_limit=clip, tile_grid=tile),
                f"clahe:{clip}")
    if method == "edges":
        from src.preprocessing.normalize import apply_edges
        alpha = float(ncfg.get("alpha", 1.0))
        return (apply_edges(src, alpha=alpha),
                apply_edges(ref, alpha=alpha),
                "edges")
    if method == "gamma_shadow":
        from src.preprocessing.shadow_correct import gamma_shadow_correct
        gamma = float(ncfg.get("gamma", 0.5))
        return (gamma_shadow_correct(src, gamma=gamma),
                gamma_shadow_correct(ref, gamma=gamma),
                f"gamma_shadow:{gamma}")
    raise ValueError(f"unknown normalize method {method!r}")


def _join(root, p):
    return p if not root else os.path.join(root, p)


def read_pair_meta(pre, root=""):
    """Read the georeference JSON written alongside the staged crops.

    Falls back to the in-memory georeference return value first (stage_pair
    passes it), but cached runs (outputs already present, georeference skipped)
    recover the same metadata from disk.
    """
    rel = os.path.join(pre.get("out_dir", ""), f'georef_{pre.get("prefix", "pair1")}.json')
    p = _join(root, rel)
    if os.path.exists(p):
        with open(p) as fh:
            return json.load(fh)
    return {}


def stage_pair(pre, normalize=None, root=""):
    """Ground-resolved staging entry point shared by every matching stage.

    Args:
        pre: `preprocessing` config block (ohrc_img, ohrc_geometry, nac_img,
             out_dir, crop_px, prefix, equal_gsd, nac_geom_csv, outputs).
        normalize: photometric preconditioning block (`normalize` config).
        root: project root prefix for relative paths ("" == paths are absolute
             or already cwd-relative).

    Returns:
        dict(src=uint8, ref=uint8, gsd_m=float|None, norm_label=str,
             meta=dict)
    """
    from src.preprocessing.georeference import georeference_pair

    src_abs = _join(root, pre["outputs"]["src"])
    ref_abs = _join(root, pre["outputs"]["ref"])
    meta = {}
    if not (os.path.exists(src_abs) and os.path.exists(ref_abs)):
        meta = georeference_pair(
            ohrc_img_path=_join(root, pre["ohrc_img"]),
            ohrc_csv_path=_join(root, pre["ohrc_geometry"]),
            nac_img_path=_join(root, pre["nac_img"]),
            out_dir=_join(root, pre["out_dir"]),
            crop_px=pre.get("crop_px", 1024),
            prefix=pre.get("prefix", "pair1"),
            equal_gsd=bool(pre.get("equal_gsd", False)),
            nac_geom_csv=pre.get("nac_geom_csv", ""),
        )
    elif read_pair_meta(pre, root):
        meta = read_pair_meta(pre, root)

    src = cv2.imread(src_abs, cv2.IMREAD_GRAYSCALE)
    ref = cv2.imread(ref_abs, cv2.IMREAD_GRAYSCALE)
    if src is None or ref is None:
        raise FileNotFoundError(f"could not read staged crops {src_abs}, {ref_abs}")

    gsd_m = meta.get("crop_gsd_m") if meta else None
    src, ref, norm_label = apply_precondition(src, ref, normalize or {})
    return dict(src=src, ref=ref, gsd_m=gsd_m, norm_label=norm_label, meta=meta)