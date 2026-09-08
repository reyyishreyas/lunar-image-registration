"""Definitive match/no-match decision layer.

Every register_* in the pipeline already returns an honest verdict
(registered / geometry_registered / not_registered) with diagnostics. This
module turns that raw report into a single machine-readable decision a user can
act on: does source content actually correspond to reference content, and which
artefact is the best aligned product?

Classification rules (always evidence-based, never fabricated):

  * ``registered``        -> MATCH, cause=content_correspondence.
  * ``geometry_registered`` -> NO MATCH (content not verifiable), with an
    evidence-driven cause among:
      - dark_or_featureless    a staged crop has low contrast / near-zero mean
                               (illumination gap, not necessarily wrong geods)
      - cross_sensor_mismatch  IR (IIRS) vs visible (OHRC/TMC/NAC) pair
      - frame_disagreement     both crops are rich in texture yet stage to
                               near-zero overlap NCC -> the two geodetic frames
                               disagree (e.g. ISRO CD grid vs NAC SPICE)
    ``overlap_ncc`` and the low-contrast scores are always attached so the
    classification is checkable.
  * ``not_registered``    -> NO MATCH, cause=cross_sensor_mismatch (content and
    geometry both failed).
  * everything else       -> ERROR with the raw note (never a verdict).
"""

from __future__ import annotations

import os

import numpy as np


CONTENT_MIN = {"min_inliers": 8, "max_rmse": 25.0}


def classify(report: dict) -> dict:
    """Reduce a registration report to a definitive MATCH / NO MATCH decision.

    Returns a dict:
        {
          "matched": bool,
          "cause": str,
          "verdict": <original verdict>,
          "explanation": str,
          "evidence": {...checkable numbers...},
        }
    """
    verdict = report.get("verdict")
    evid = _evidence(report)

    if verdict == "registered":
        return _build(
            report, matched=True, cause="content_correspondence",
            headline="MATCH — the source content genuinely corresponds to the "
                     f"reference (inliers={report['inliers']}, "
                     f"self-RMSE={report.get('rmse_self_px')} px).",
            evid=evid)

    if verdict == "geometry_registered":
        cause, headline = _geometry_cause(report, evid)
        return _build(report, matched=False, cause=cause,
                      headline=headline, evid=evid)

    if verdict == "not_registered":
        return _build(
            report, matched=False, cause="cross_sensor_mismatch",
            headline="NO MATCH — content matching found no reliable correspond-"
                     "ence (IR/sensor mismatch or genuinely different scene); "
                     "geometry registration is also impossible for this pair.",
            evid=evid)

    return _build(
        report, matched=False, cause="error",
        headline=f"REGISTRATION ERROR — {verdict}: "
                 f"{report.get('notes', [])[-1] if report.get('notes') else 'no detail'}",
        evid=evid)


def _evidence(report: dict) -> dict:
    diag = report.get("diagnostics") or {}
    return {
        "verdict": report.get("verdict"),
        "inliers": report.get("inliers", 0),
        "inlier_ratio": report.get("inlier_ratio", 0.0),
        "n_matches": report.get("n_matches", 0),
        "rmse_px": report.get("rmse_px"),
        "rmse_self_px": report.get("rmse_self_px"),
        "overlap_ncc": diag.get("overlap_ncc"),
        "overlap_px": diag.get("overlap_px"),
        "src_low_contrast": diag.get("src_low_contrast"),
        "ref_low_contrast": diag.get("ref_low_contrast"),
        "src_mean": diag.get("src_mean"),
        "ref_mean": diag.get("ref_mean"),
    }


def _geometry_cause(report: dict, evid: dict):  # -> (cause, headline)
    ncc = evid.get("overlap_ncc")
    src_lc = evid.get("src_low_contrast")
    ref_lc = evid.get("ref_low_contrast")
    src_mean = evid.get("src_mean")
    ref_mean = evid.get("ref_mean")
    sensor = report.get("sensor_pair", "")
    cb = report.get("notes")
    joined = " ".join(cb or []) if cb else ""

    if "IIRS" in report.get("pair", str()) or "iirs" in sensor \
            or "IR" in joined:
        return ("cross_sensor_mismatch",
                "NO MATCH — IIRS (infrared) vs a visible reference: content "
                "cannot correspond; ground-grid registration is not defined "
                "for IIRS.")
    if src_lc is not None and ref_lc is not None \
            and (max(src_lc, ref_lc) > 0.5 or
                 min((src_mean if src_mean is not None else 255),
                     (ref_mean if ref_mean is not None else 255)) < 20 or
                 (ncc is not None and abs(ncc) > 0.05)):
        # a staged crop is featureless/dark or the staged pair shows a weak
        # genuine overlap — illumination gap is the plausible cause
        dark = (" source is very dark (mean=%.1f)" % src_mean) \
            if src_mean is not None and src_mean < 20 else ""
        return ("dark_or_featureless",
                "NO MATCH — content not verifiable on this pair; staged on a "
                f"common ground grid but overlap NCC {_fmt_ncc(ncc)}"
                f"{dark}; low-contrast src={src_lc:.3f} ref={ref_lc:.3f}. "
                "This is an illumination/dataset gap, not a claimed alignment.")
    return ("frame_disagreement",
            "NO MATCH — both staged crops are texture-rich, yet the common-grid "
            f"overlap NCC is {_fmt_ncc(ncc)}: the two geodetic frames (e.g. "
            "ISRO CD ground grid vs LRO NAC SPICE) genuinely disagree on this "
            "pair. Registered on geometry only; content correspondence is NOT "
            "claimed.")


def _fmt_ncc(ncc):
    if ncc is None:
        return "n/a"
    return f"{ncc:+.4f}"


def _build(report, *, matched, cause, headline, evid):
    return {
        "matched": matched,
        "matches": matched,
        "cause": cause,
        "verdict": report.get("verdict"),
        "explanation": headline,
        "evidence": evid,
        "best_product": best_product(report),
    }


def best_product(report: dict) -> str | None:
    """Path of the best aligned product for a report, or None."""
    arts = report.get("artifacts") or {}
    for key in ("best_aligned", "warped", "checkerboard", "montage",
                "overlay", "src", "matches"):
        p = arts.get(key)
        if p:
            return p
    return None


def find_best_aligned(report: dict) -> str | None:
    return best_product(report)


def best_aligned_name(report: dict) -> str | None:
    p = best_product(report)
    return os.path.basename(p) if p else None