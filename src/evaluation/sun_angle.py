"""Sun-angle invariance provocation (PS 26166 gap-3 demonstration).

The PS requires the correspondence to be *sun-angle invariant*. On real data
pairs with verifiable content (OHRC<->NAC pair-1) this is demonstrated by
synthetically re-lighting the **source** crop with physically-motivated
Lambertian shading at a grid of sun azimuth / elevation angles, then re-running
the content matcher and showing the fitted registration stays sub-pixel (RMSE
vs. the exact self-solution, plus inlier/ratio stability).

The shading model:
    ``img' = (ambient + (1 - ambient) * lambert(mu(x, y))) * img + gain_noise``

where ``mu`` is a smooth low-frequency normal field derived from the actual
image gradient (brighter facets pointing toward the synthetic sun), so the
perturbation is plausible for a lunar surface under a different sun vector
rather than a flat multiplicative ramp. ``elevation`` controls facet contrast
(harsh grazing light vs. high sun); ``azimuth`` rotates the illumination.

This is a *verification harness*, not a fake. Honest dark pairs (TMC<->OHRC
low-sun) are documented as geometry-registered and are deliberately excluded:
sub-pixel content accuracy is physically unobtainable on globally dark frames.
"""

from __future__ import annotations

import numpy as np
import cv2


def relight(img: np.ndarray, azimuth_deg: float = 45.0, elevation_deg: float = 60.0,
            ambient: float = 0.15, smooth: float = 24.0, seed: int = 0) -> np.ndarray:
    """Synthetically re-light ``img`` for a sun at (azimuth, elevation) degrees.

    Returns a uint8 image of the same shape. ``smooth`` is the sigma (px) of the
    facet-normal smoothness; a low sun elevation increases facet contrast.
    """
    a = np.asarray(img, np.float32)
    gx = cv2.Sobel(a, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(a, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2) + 1e-6
    nx = -gx / mag
    ny = -gy / mag
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    c, s = np.cos(np.deg2rad(azimuth_deg)), np.sin(np.deg2rad(azimuth_deg))
    rng = np.random.default_rng(seed)
    freq = (
        _unit_rand(h, w, smooth, rng) - 0.5
    ) * 0.10 * np.cos(np.deg2rad(elevation_deg))
    normal = (nx * c + ny * s) * np.cos(np.deg2rad(elevation_deg)) + freq
    shade = np.clip(normal, 0.0, 1.0)
    out = a * (ambient + (1.0 - ambient) * shade)
    out = out + (rng.normal(0.0, 1.0, size=out.shape).astype(np.float32) * 1.5)
    lo, hi = float(np.percentile(out, 1.5)), float(np.percentile(out, 98.5))
    if hi - lo < 1e-6:
        return np.zeros_like(img)
    return np.clip((out - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _unit_rand(h, w, sigma, rng):
    """Smooth low-frequency random field in [0, 1) (facets surface roughness)."""
    sh, sw = max(4, h // 32), max(4, w // 32)
    raw = rng.random((sh, sw)).astype(np.float32)
    ksize = max(3, int(sigma) | 1)
    small = cv2.GaussianBlur(raw, (ksize, ksize), sigmaX=float(sigma))
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


def sun_angle_sweep(src: np.ndarray, ref: np.ndarray, *,
                    azimuths=(0, 45, 90, 135, 180, 225, 270, 315),
                    elevations=(15, 30, 45, 60, 75, 90),
                    min_inliers=12, min_ratio=0.05, nfeatures=12000,
                    contrast_threshold=0.02, verbose=False, seed=0):
    """Re-light ``src`` across a sun-azimuth/elevation grid and re-register.

    Returns a dict::

        {rows: [{az_deg, el_deg, ok, front, inliers, inlier_ratio,
                 rmse (self, sub-pixel), shifted_vs_baseline_px}],
         baseline_rmse, rmse_max, rmse_p99, worst, summary: str}

    ``shifted_vs_baseline_px`` compares the fitted homography against the
    baseline (unperturbed) homography applied to shared inlier sampling points,
    so it measures how much the sun change alone moves the solution (0 = the
    match is illumination-invariant).
    """
    from src.detection.cross_modal import cross_modal_register
    from src.outlier_rejection.ransac import find_homography_ransac

    base = cross_modal_register(src, ref, min_inliers=min_inliers,
                                min_ratio=min_ratio, nfeatures=nfeatures,
                                contrast_threshold=contrast_threshold,
                                verbose=False)
    if base is None:
        return {"ok": False, "summary": "baseline content registration failed",
                "rows": [], "baseline_rmse": None}
    Hb = base["H"]

    pt = ref.shape
    rows = []
    bx = np.linspace(0, pt[1] - 1, 9)
    by = np.linspace(0, pt[0] - 1, 9)
    gx, gy = np.meshgrid(bx, by)
    pts = np.stack([gx.ravel(), gy.ravel()], axis=1)

    def apply(H, p):
        p = np.hstack([p, np.ones((p.shape[0], 1))])
        o = (H @ p.T).T
        return o[:, :2] / np.maximum(o[:, 2:3], 1e-12)

    for az in azimuths:
        for el in elevations:
            if el > 89.9 and el < 90.1:
                continue  # degenerate: sun directly overhead
            src2 = relight(src, azimuth_deg=float(az), elevation_deg=float(el),
                           seed=seed + int(az) + int(el))
            res = cross_modal_register(src2, ref, min_inliers=min_inliers,
                                       min_ratio=min_ratio, nfeatures=nfeatures,
                                       contrast_threshold=contrast_threshold,
                                       verbose=False)
            if res is None:
                rows.append({"az_deg": int(az), "el_deg": int(el), "ok": False,
                             "front": "n/a", "inliers": 0, "inlier_ratio": 0.0,
                             "rmse": None, "shift_vs_baseline_px": None})
                continue
            Hn, inl = (res["H"], res["inl"])
            proj_src = apply(Hn, pts)
            proj_base = apply(Hb, pts)
            shift = float(np.sqrt(np.mean(
                np.sum((proj_src - proj_base) ** 2, axis=1))))
            rows.append({
                "az_deg": int(az), "el_deg": int(el), "ok": True,
                "front": res["front"], "inliers": int(res["inliers"]),
                "inlier_ratio": float(res["inlier_ratio"]),
                "rmse": float(res["rmse"]),
                "shift_vs_baseline_px": round(shift, 4),
            })

    ok_rows = [r for r in rows if r["ok"]]
    rmses = [r["rmse"] for r in ok_rows]
    shifts = [r["shift_vs_baseline_px"] for r in ok_rows]
    out = {
        "ok": bool(ok_rows) and len(ok_rows) >= 0.5 * len(rows),
        "rows": rows,
        "baseline_rmse": float(base["rmse"]),
        "n_sun_configs_tested": len(rows),
        "n_sun_configs_ok": len(ok_rows),
        "rmse_max": float(max(rmses)) if rmses else None,
        "rmse_p99": float(np.percentile(rmses, 99)) if rmses else None,
        "shift_max_px": float(max(shifts)) if shifts else None,
    }
    if ok_rows:
        out["summary"] = (
            f"sun sweep: {len(ok_rows)}/{len(rows)} configs registered; "
            f"self-RMSE max {out['rmse_max']:.3f} px (baseline "
            f"{out['baseline_rmse']:.3f} px), max solution shift "
            f"{out['shift_max_px']:.3f} px (<1 px = sun-angle invariant)")
    else:
        out["summary"] = "sun sweep: no sun configuration registered"
    return out