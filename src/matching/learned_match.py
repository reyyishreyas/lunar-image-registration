"""SuperGlue matcher adapter wrapping third_party/SuperGluePretrainedNetwork.

Consumes the keypoints/descriptors/scores/images cached by
``src.detection.learned.detect_superpoint`` and returns cv2.DMatch matches so the
pipeline's RANSAC and evaluation stages run unchanged.
"""


def _ensure_imported():
    import sys  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    repo = (Path(__file__).resolve().parents[2]
            / "third_party" / "SuperGluePretrainedNetwork").resolve()
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))


def _load_loftr():
    """Import the LoFTR repo under a synthetic ``src`` package (its package
    name collides with ours) and a ``kornia.utils.grid`` compatibility shim."""
    import sys  # noqa: PLC0415
    import types  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    root = (Path(__file__).resolve().parents[2] / "third_party" / "LoFTR").resolve()
    src_dir = root / "src"
    preserved_src = {k: sys.modules[k] for k in list(sys.modules)
                     if k == "src" or k.startswith("src.")}
    fake_src = types.ModuleType("src")
    fake_src.__path__ = [str(src_dir)]
    fake_src.__package__ = "src"

    grid_mod = types.ModuleType("kornia.utils.grid")
    import kornia.utils  # noqa: PLC0415

    grid_mod.create_meshgrid = kornia.utils.create_meshgrid
    preserved_kornia = sys.modules.get("kornia.utils.grid")
    try:
        sys.modules["src"] = fake_src
        sys.modules["kornia.utils.grid"] = grid_mod
        from src.loftr import LoFTR, default_cfg  # noqa: PLC0415 (third-party src)

        return LoFTR, default_cfg
    finally:
        for k in list(sys.modules):
            if k == "src" or k.startswith("src."):
                del sys.modules[k]
        sys.modules.update(preserved_src)
        if preserved_kornia is None:
            sys.modules.pop("kornia.utils.grid", None)
        else:
            sys.modules["kornia.utils.grid"] = preserved_kornia


def match_superglue(kp1, des1, kp2, des2, weights="outdoor",
                    match_threshold=0.2, sinkhorn_iterations=20,
                    verbose=False, **kwargs):
    """Match two SuperPoint feature sets with pretrained SuperGlue."""
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from src.detection.learned import _cache  # noqa: PLC0415

    _ensure_imported()
    from models.superglue import SuperGlue  # noqa: PLC0415

    c0, c1 = _cache["src"], _cache["ref"]
    if c0 is None or c1 is None:
        raise RuntimeError("match_superglue called before detect_superpoint cache was filled")

    def f2t(img):
        return torch.from_numpy(np.asarray(img, np.float32) / 255.0)[None, None].float()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    sg_cfg = {
        "weights": weights,
        "match_threshold": match_threshold,
        "sinkhorn_iterations": sinkhorn_iterations,
    }
    sg_cfg.update({k: v for k, v in kwargs.items() if k != "name"})
    model = SuperGlue(sg_cfg).eval().to(dev)

    data = {
        "image0": f2t(c0["image"]).to(dev),
        "image1": f2t(c1["image"]).to(dev),
        "keypoints0": torch.from_numpy(c0["kp"]).float()[None].to(dev),
        "scores0": torch.from_numpy(c0["scores"]).float()[None].to(dev),
        "descriptors0": torch.from_numpy(c0["des"]).float().transpose(0, 1)[None].to(dev),
        "keypoints1": torch.from_numpy(c1["kp"]).float()[None].to(dev),
        "scores1": torch.from_numpy(c1["scores"]).float()[None].to(dev),
        "descriptors1": torch.from_numpy(c1["des"]).float().transpose(0, 1)[None].to(dev),
    }
    pred = model(data)
    m0 = pred["matches0"][0].detach().cpu().numpy()
    conf = pred["matching_scores0"][0].detach().cpu().numpy()
    good = np.where(m0 >= 0)[0]
    matches = [cv2.DMatch(_queryIdx=int(i), _trainIdx=int(m0[i]),
                          _distance=float(1.0 - conf[i])) for i in good]
    if verbose:
        print(f"  SuperGlue matches: {len(matches)}")
    return matches


def match_loftr(kp1, des1, kp2, des2, weights="outdoor", verbose=False, **kwargs):
    """Match two images with pretrained LoFTR (dense matcher).

    LoFTR does not use the keypoint lists from the detector stage; it produces
    its own dense correspondences. To keep the pipeline's DMatch-based RANSAC
    unchanged, the adapter appends LoFTR's keypoints to the (mutable) kp lists
    passed in and returns one DMatch per correspondence.
    """
    import cv2  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from src.detection.learned import _cache  # noqa: PLC0415

    LoFTR, default_cfg = _load_loftr()

    c0, c1 = _cache["src"], _cache["ref"]
    if c0 is None or c1 is None:
        raise RuntimeError("match_loftr called before detect_superpoint cache was filled")

    ckpt = (Path(__file__).resolve().parents[2]
            / "third_party" / "LoFTR" / "weights"
            / f"outdoor_ds.ckpt" if weights == "outdoor"
            else Path(__file__).resolve().parents[2]
            / "third_party" / "LoFTR" / "weights" / f"{weights}_ds.ckpt")

    def make_tensor(img):
        a = np.asarray(img, np.float32) / 255.0
        h, w = a.shape
        pad_h, pad_w = (8 - h % 8) % 8, (8 - w % 8) % 8
        if pad_h or pad_w:
            a = np.pad(a, ((0, pad_h), (0, pad_w)), mode="edge")
        return torch.from_numpy(a)[None, None].float()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    matcher = LoFTR(config=default_cfg)
    try:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    except Exception:
        state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    matcher.load_state_dict(state["state_dict"])
    matcher = matcher.eval().to(dev)

    batch = {"image0": make_tensor(c0["image"]).to(dev),
             "image1": make_tensor(c1["image"]).to(dev)}
    with torch.no_grad():
        matcher(batch)
    mk0 = batch["mkpts0_f"].detach().cpu().numpy()
    mk1 = batch["mkpts1_f"].detach().cpu().numpy()
    if verbose:
        print(f"  LoFTR matches: {len(mk0)}")

    base0, base1 = len(kp1), len(kp2)
    kp1.extend(cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in mk0)
    kp2.extend(cv2.KeyPoint(x=float(p[0]), y=float(p[1]), size=1) for p in mk1)
    matches = [cv2.DMatch(_queryIdx=base0 + i, _trainIdx=base1 + i, _distance=0.0)
               for i in range(len(mk0))]
    return matches