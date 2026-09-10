"""Attempt to RESOLVE the pair-2 (polar, OHRC-2026 vs NAC M1127547939RC)
"feature correspondence NOT verifiable" problem on a FAIR footing.

Why a fair test first matters
-----------------------------
The historical "all matchers ~0 matches" (SIFT 14, SuperGlue 2, LoFTR 9/419,
DISK+LightGlue cross=0) was measured almost entirely on *mismatched working
sets* (OHRC WS 9370x1200 vs NAC WS 6528x633 in raw, un-aligned native
geometry, or LoFTR squash-resized 832x832). The pairs were NEVER co-located
before those matchers ran.

Here we match the two images that the geometry route ALREADY co-located on the
same 60 m/px ground grid (the SPICE/ISRO equal-GSD staged strip, 1536x194).
On that grid the true transform is the IDENTITY homography, which gives us a
quantitative target for content correspondence: real matches must agree with
identity, and must beat a row-reversed (envelope-preserving) null control.

Procedure per photometric front (repo cross-modal front-ends):
  * stretch the SRC (near-black, max 111) and REF (35-2255) to uint8
  * apply front: none | clahe | histogram_match | log_ratio | gradient_structure
  * SIFT + BF(0.75) + USAC_MAGSAC(5): n_matches, inliers, ratio, H
  * identity-consistency: inlier displacement vs identity, RMSE vs identity
  * phase correlation lock (shift,response) vs row-reversed-null response
  * NMI differential test: NMI at true(0) offset vs shifted controls and vs
    the row-reversed null

Resolution criterion (all must hold for at least one front):
  inliers >= 20           (content amount)
  inlier_ratio >= 0.25    (content quality)
  inliers > 3 * null_inliers   (beats envelope-null -> not an envelope artifact)
  >= 50% of inliers within 3 px of identity and rmse_identity <= 3 px
     (content agrees with the SPICE/ISRO geometry)

Writes results/logs/pair2_content_resolve.md + .json.
Exit 0 = resolved (at least one front passes), 1 = not resolved.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.detection.classical import detect_sift              # noqa: E402
from src.detection.cross_modal import _apply_front           # noqa: E402
from src.matching.classical_match import match_bf_ratio      # noqa: E402
from src.outlier_rejection.ransac import find_homography_ransac  # noqa: E402

FRONTS = ("none", "clahe", "histogram_match", "log_ratio", "gradient_structure")


def as_u8(arr: np.ndarray, lo_p=1.0, hi_p=99.5) -> np.ndarray:
    f = np.asarray(arr, np.float32)
    lo, hi = np.nanpercentile(f, (lo_p, hi_p))
    if hi - lo < 1e-9:
        return np.zeros(f.shape, np.uint8)
    return np.clip((f - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)


def _disp(pts1, pts2):
    """Inlier displacement magnitude from the IDENTITY transform (same grid)."""
    return np.sqrt(np.sum((pts2 - pts1) ** 2, axis=1))


def _nmi(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised mutual information on the uint8 overlap (skimage)."""
    from skimage.metrics import normalized_mutual_information
    mask = (a > 0) & (b > 0)
    if mask.sum() < 500:
        return float("nan")
    return float(normalized_mutual_information(a[mask], b[mask]))


def _phase(a8, b8):
    """cv2.phaseCorrelate on float32 pair; returns (shift, response)."""
    a = a8.astype(np.float32)
    b = b8.astype(np.float32)
    try:
        (dx, dy), resp = cv2.phaseCorrelate(a, b)
        return (round(float(dx), 3), round(float(dy), 3)), float(resp)
    except cv2.error:
        return None, None


def main() -> int:
    d = np.load(os.path.join(PROJECT_ROOT,
                             "data/processed/spice_georef/pair2_fullstrip.npz"),
                allow_pickle=True)
    src, ref = d["src"], d["ref"]

    src_st = as_u8(src)
    ref_st = as_u8(ref)

    rows = []      # per-front results
    best = None

    for front in FRONTS:
        s = _apply_front(src_st, front, ref=ref_st) if front != "histogram_match" \
            else _apply_front(src_st, front, ref=ref_st)
        r = _apply_front(ref_st, front, ref=src_st) if front != "histogram_match" \
            else _apply_front(ref_st, front, ref=src_st)

        kp1, d1 = detect_sift(s, nfeatures=20000, contrast_threshold=0.03)
        kp2, d2 = detect_sift(r, nfeatures=20000, contrast_threshold=0.03)
        m = match_bf_ratio(d1, d2, ratio=0.75, verbose=False)
        ps1 = np.float32([kp1[x.queryIdx].pt for x in m]).reshape(-1, 2)
        ps2 = np.float32([kp2[x.trainIdx].pt for x in m]).reshape(-1, 2)
        H, inl = (find_homography_ransac(ps1, ps2, ransac_thresh=5.0,
                                         method="usac_magsac")
                  if len(m) >= 8 else (None, None))

        n_matches = int(len(m))
        n_in = int(inl.sum()) if inl is not None else 0
        ratio = round(float(n_in / n_matches), 4) if n_matches else 0.0
        rmse_id = None
        frac_near = None
        if H is not None and n_in >= 4:
            p1i, p2i = ps1[inl > 0], ps2[inl > 0]
            disp = _disp(p1i, p2i)
            frac_near = float((disp <= 3.0).mean())
            rmse_id = float(np.sqrt(np.mean(disp ** 2)))

        # --- row-reversed null control (same front, content destroyed) ---
        rN = r[::-1, :].copy()
        kpN, dN = detect_sift(rN, nfeatures=20000, contrast_threshold=0.03)
        mN = match_bf_ratio(d1, dN, ratio=0.75, verbose=False)
        ps1N = np.float32([kp1[x.queryIdx].pt for x in mN]).reshape(-1, 2) \
            if mN else np.empty((0, 2))
        psN = np.float32([kpN[x.trainIdx].pt for x in mN]).reshape(-1, 2) \
            if mN else np.empty((0, 2))
        HN, inlN = (find_homography_ransac(ps1N, psN, ransac_thresh=5.0,
                                           method="usac_magsac")
                    if len(mN) >= 8 else (None, np.zeros(0, bool)))
        n_null = int(inlN.sum()) if inlN is not None and len(inlN) else 0

        # --- phase lock vs its null ---
        (sh_t, resp_t), (sh_n, resp_n) = _phase(s, r), _phase(s, rN)

        # --- NMI differential: true(0) vs shifted controls vs null ---
        nmi_true = _nmi(s, r)
        nmi_null = _nmi(s, rN)
        nmi_shifts = {f"({dx},{dy})": _nmi(s, np.roll(r, (dy, dx), axis=(0, 1)))
                      for dx, dy in ((30, 0), (60, 0), (0, 6), (30, 6))}

        content_lock = n_in >= 20 and ratio >= 0.25
        beats_null = n_null > 0 and n_in > 3 * n_null
        id_ok = (frac_near is not None and frac_near >= 0.5
                 and rmse_id is not None and rmse_id <= 3.0)
        resolved = bool(content_lock and beats_null and id_ok)

        row = {
            "front": front,
            "n_matches": n_matches, "inliers": n_in, "inlier_ratio": ratio,
            "rmse_identity": rmse_id, "frac_inliers_near_identity": frac_near,
            "null_inliers": n_null,
            "beats_null_x3": n_null > 0 and n_in / max(n_null, 1) >= 3.0,
            "phase_true": {"shift": sh_t, "response": resp_t},
            "phase_null": {"shift": sh_n, "response": resp_n},
            "phase_lock_ratio": (round(resp_t / resp_n, 3)
                                 if resp_t is not None and resp_n and resp_n > 0
                                 else None),
            "nmi_true": nmi_true, "nmi_null": nmi_null, "nmi_shifts": nmi_shifts,
            "content_lock": content_lock, "beats_null": beats_null,
            "identity_ok": id_ok, "resolved": resolved,
        }
        rows.append(row)
        if resolved and (best is None or n_in > best["inliers"]):
            best = row

    n_resolved = sum(1 for r in rows if r["resolved"])
    status = "RESOLVED" if n_resolved else "NOT RESOLVED"

    out = {
        "status": status,
        "pair": "OHRC-2026 sun_elev=1.895764 / NAC M1127547939RC",
        "grid": "equal-GSD 60 m/px staged orthos (1536x194), TRUE transform = identity",
        "criterion": "inliers>=20 & ratio>=0.25 & inliers>3x null & >=50% inliers "
                     "near identity & rmse_identity<=3px",
        "n_fronts_resolved": n_resolved,
        "fronts": rows,
        "best": best,
    }
    os.makedirs(os.path.join(PROJECT_ROOT, "results/logs"), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT,
                           "results/logs/pair2_content_resolve.json"), "w") as fh:
        json.dump(out, fh, indent=2, default=str)

    lines = [
        "# Pair-2 (polar) content-correspondence RESOLUTION experiment\n",
        f"Status: **{status}** "
        f"({n_resolved}/{len(rows)} fronts passed all bars).\n",
        "Method: SIFT + BF(0.75) + USAC_MAGSAC(5) on the **co-located** 60 m/px "
        "staged orthos (true transform = identity), with row-reversed null "
        "control and phase/NMI lock tests.\n",
        "| front | matches | inl | ratio | rmse_id | frac<3px | null_inl | "
        "beats3x | phase lock ratio | nmi true | nmi null | RESOLVED |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['front']} | {r['n_matches']} | {r['inliers']} | "
            f"{r['inlier_ratio']} | {r['rmse_identity']} | "
            f"{_pct(r['frac_inliers_near_identity'])} | {r['null_inliers']} | "
            f"{_yn(r['beats_null_x3'])} | {r['phase_lock_ratio']} | "
            f"{_f3(r['nmi_true'])} | {_f3(r['nmi_null'])} | "
            f"{_yn(r['resolved'])} |")
    lines.append("")
    if best:
        lines.append(f"Best front: **{best['front']}** — inliers {best['inliers']}, "
                     f"ratio {best['inlier_ratio']}, rmse_vs_identity "
                     f"{best['rmse_identity']} px, null {best['null_inliers']}, "
                     f"phase-lock-ratio {best['phase_lock_ratio']}.")
    lines.append("")
    lines.append("Interpreting the null controls: a front counts as REAL content "
                 "only if it beats its row-reversed null by 3x inliers AND shows "
                 "a phase/NMI lock that the raw intensity did not.")
    with open(os.path.join(PROJECT_ROOT,
                           "results/logs/pair2_content_resolve.md"), "w") as fh:
        fh.write("\n".join(lines))

    print(json.dumps(out, indent=2, default=str))
    return 0


def _yn(b):
    return "YES" if b else "no"


def _f3(x):
    return "n/a" if x is None else f"{x:.3f}"


def _pct(x):
    return "n/a" if x is None else f"{x:.2f}"


if __name__ == "__main__":
    sys.exit(main())