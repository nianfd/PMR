"""Exemplar-correlation response: construction and calibration.

This mirrors CountingDINO's ``post_process_density_map`` step for step, but
returns the intermediate quantities the readout needs (the calibration
coefficient, the response cut, the map itself).  Dividing the averaged response
by the exemplar mass makes the count an integral of the calibrated map, so
responses of different images live on the same scale.

Nothing in this module is trained.
"""
from types import SimpleNamespace

import numpy as np
import torch
import torchvision.ops as ops

from .backend import ellipse_coverage, rescale_tensor, resize_conv_maps
from .pcm import build_kernels2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Cfg:
    """Response-calibration switches of the original pipeline."""

    divide_et_impera = True
    divide_et_impera_twice = True
    filter_background = True
    ellipse_normalization = True
    use_minmax_norm = True
    correct_bbox_resize = True
    remove_bbox_intersection = False
    scaling_coeff = 1.0


def post_process_instrumented(conv_maps, pooled_feats, bboxes, cfg, apply_cut=True):
    """Average, rescale, normalise by the exemplar mass, and cut the response."""
    output, resize_ratios = resize_conv_maps(conv_maps)
    output = output.mean(dim=0)
    output = rescale_tensor(output)
    pre_norm = output.clone()
    pooled_vals, sboxes = [], []
    for bbox, ratio in zip(bboxes, resize_ratios):
        sb = torch.tensor([bbox[0] * ratio[1], bbox[1] * ratio[0],
                           bbox[2] * ratio[1], bbox[3] * ratio[0]]).int()
        osz = (int(sb[3] - sb[1]), int(sb[2] - sb[0]))
        if osz[0] < 2 or osz[1] < 2:
            continue
        pv = ops.roi_align(output.unsqueeze(0).unsqueeze(0),
                           [sb.unsqueeze(0).float().to(DEVICE)], output_size=osz,
                           spatial_scale=1.0)
        pooled_vals.append(pv)
        sboxes.append([int(v) for v in sb])
    if not pooled_vals:
        return None
    region_mass = [(p[0, 0] * ellipse_coverage(p.shape[-2], p.shape[-1]).to(DEVICE)).sum()
                   for p in pooled_vals]
    norm_coeff = sum(region_mass) / (len(pooled_vals) * cfg.scaling_coeff)
    y = output / norm_coeff.clamp(min=1e-12)
    thresh = max([f.shape[-2] * f.shape[-1] for f in pooled_feats])
    cut = float(1.0 / thresh)
    if apply_cut:
        y = y.clone()
        y[y < cut] = 0
    return {
        "y": y, "pre_norm": pre_norm, "norm_coeff": float(norm_coeff),
        "per_exemplar_mass": [float(v) for v in region_mass],
        "n_exemplars": len(pooled_vals), "cut": cut, "thresh": int(thresh),
        "boxes_y": sboxes,
    }


def calibrated_response(feats, bboxes, mask_lists, cfg=None):
    """Full PCM response for one image, or ``None`` if no kernel survives.

    ``mask_lists`` comes from :func:`pmr.pcm.pcm_mask_lists`; the response is
    left uncut (``apply_cut=False``) exactly as in the reference runs, and the
    cut is reported so that the readout can apply it to the mass statistic.
    """
    cfg = cfg or Cfg()
    conv_maps, kernel_maps, rescaled = build_kernels2(feats, bboxes, mask_lists, use_ell=True)
    if not conv_maps:
        return None
    return post_process_instrumented(conv_maps, kernel_maps, rescaled, cfg, apply_cut=False)


def response_stats(res):
    """Scalar features of one calibrated response: mass, exemplar scale, cut."""
    y = np.clip(res["y"].detach().cpu().numpy().astype(np.float64), 0, None)
    H, W = y.shape
    ms, bs = [], []
    for sb in res.get("boxes_y") or []:
        y1, x1, y2, x2 = [int(v) for v in sb]
        y1, x1 = max(y1, 0), max(x1, 0)
        y2, x2 = min(y2, H), min(x2, W)
        if y2 - y1 < 2 or x2 - x1 < 2:
            continue
        ms.append(y[y1:y2, x1:x2].sum())
        bs.append((y2 - y1) * (x2 - x1))
    exmass = float(np.median(ms)) if ms else float("nan")
    boxarea = float(np.median(bs)) if bs else float("nan")
    box_r = float(0.5 * np.sqrt(boxarea)) if np.isfinite(boxarea) else float("nan")
    cut = float(res["cut"])
    return SimpleNamespace(
        y=y,
        mass=float(np.where(y >= cut, y, 0.0).sum()),
        box_r=box_r,
        norm_coeff=float(res["norm_coeff"]),
        cut=cut,
        shape=[int(H), int(W)],
        max=float(y.max()) if y.size else 0.0,
    )
