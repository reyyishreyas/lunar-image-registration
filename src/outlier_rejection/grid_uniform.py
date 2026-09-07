"""Phase 4 grid-based uniform match capping (AI_EXECUTION_PLAN.md Step 4.1).

Caps the number of matches per spatial grid cell over the source image so the
match set fed to RANSAC is spatially uniform instead of dominated by a few
dense clusters. Within a cell the lowest-distance matches are kept.
"""

from __future__ import annotations

import math


def cell_index(pt, image_shape, rows, cols):
    h, w = image_shape[:2]
    cell_h = math.ceil(h / rows)
    cell_w = math.ceil(w / cols)
    x, y = pt
    r = min(int(y // cell_h), rows - 1)
    c = min(int(x // cell_w), cols - 1)
    return r * cols + c


def cap_by_grid(matches, kp1, image_shape, rows=4, cols=4, max_per_cell=None,
                max_total=None):
    """Reduce matches to a spatially uniform subset.

    Args:
        matches: list of cv2.DMatch (already distance-sorted by the matcher).
        kp1: source keypoints (query side).
        image_shape: (h, w) of the source image.
        rows, cols: grid dimensions.
        max_per_cell: max matches kept per cell (best distances). If None,
            all matches in a cell are kept.
        max_total: optional global cap applied round-robin across cells so the
            surviving set stays spatially uniform. If None, no global cap.

    Returns:
        Filtered list of cv2.DMatch.
    """
    if not matches:
        return matches
    if not (max_per_cell or max_total):
        return matches
    cells = {}
    for m in matches:
        cidx = cell_index(kp1[m.queryIdx].pt, image_shape, rows, cols)
        cells.setdefault(cidx, []).append(m)

    kept = []
    for cidx in sorted(cells):
        bucket = cells[cidx]
        if max_per_cell is not None:
            bucket = bucket[:max_per_cell]
        kept.extend(bucket)

    if max_total is not None and len(kept) > max_total:
        if max_per_cell is None:
            per_cell = math.ceil(max_total / len(cells)) if cells else max_total
            per_cell = max(per_cell, 1)
            kept = []
            for cidx in sorted(cells):
                kept.extend(cells[cidx][:per_cell])
        cyclic = []
        by_cell = {}
        for m in kept:
            cidx = cell_index(kp1[m.queryIdx].pt, image_shape, rows, cols)
            by_cell.setdefault(cidx, []).append(m)
        for cidx in sorted(by_cell):
            by_cell[cidx] = iter(by_cell[cidx])
        while len(cyclic) < max_total:
            progressed = False
            for cidx in sorted(by_cell):
                try:
                    cyclic.append(next(by_cell[cidx]))
                    progressed = True
                except StopIteration:
                    pass
            if not progressed:
                break
        kept = cyclic[:max_total]

    return kept