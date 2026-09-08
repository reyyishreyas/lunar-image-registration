from src.evaluation.decision import classify


def _rep(verdict="registered", **kw):
    r = {"verdict": verdict, "inliers": 12, "inlier_ratio": 0.5,
         "n_matches": 200, "rmse_px": 0.7, "rmse_self_px": 0.9,
         "diagnostics": {"src_low_contrast": 0.1, "ref_low_contrast": 0.1,
                         "src_mean": 60.0, "ref_mean": 70.0,
                         "overlap_ncc": 0.5, "overlap_px": 10000},
         "sensor_pair": "ohrc-nac", "pair": {}, "notes": [],
         "artifacts": {"best_aligned": "x_aligned.png"}}
    r.update(kw)
    return r


def test_decision_registered_is_match():
    d = classify(_rep("registered"))
    assert d["matched"] is True
    assert d["cause"] == "content_correspondence"
    assert d["best_product"] == "x_aligned.png"


def test_decision_dark_crop():
    d = classify(_rep("geometry_registered",
                      diagnostics={"src_low_contrast": 0.05,
                                   "ref_low_contrast": 0.05,
                                   "src_mean": 4.0, "ref_mean": 90.0,
                                   "overlap_ncc": 0.01, "overlap_px": 9000}))
    assert d["matched"] is False
    assert d["cause"] == "dark_or_featureless"


def test_decision_frame_disagreement():
    d = classify(_rep("geometry_registered",
                      diagnostics={"src_low_contrast": 0.0,
                                   "ref_low_contrast": 0.0,
                                   "src_mean": 88.0, "ref_mean": 60.0,
                                   "overlap_ncc": -0.003, "overlap_px": 9000}))
    assert d["matched"] is False
    assert d["cause"] == "frame_disagreement"


def test_decision_iirs_mismatch():
    d = classify(_rep("not_registered", sensor_pair="iirs-nac"))
    assert d["matched"] is False
    assert d["cause"] == "cross_sensor_mismatch"