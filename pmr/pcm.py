"""Prototype--Contrastive Kernel Masking (PCM).

The exemplar kernel of CountingDINO is ``pooled * ellipse``.  PCM multiplies it
by a content mask

    M_j(x) = alpha + (1 - alpha) * sigmoid( (s_j(x) - med) / (1.4826 * MAD + eps) )

where ``s_j(x) = cos(f(x), mu_j^+) - cos(f(x), mu_j^-)`` is the central minus
peripheral prototype similarity, clipped to the exemplar box, and the kernel is

    K_j = E_j * M_j ** eta .

With (gamma, alpha, eta) = (0.6, 0.05, 2) this is the configuration reported in
the manuscript: no object-head mask is involved.
"""
import torch
import torch.nn.functional as F
import torchvision.ops as ops

from .backend import ellipse_coverage, rescale_bbox
from .geometry import build_regions

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def proto_masks(feats, tbboxes, grid, center_frac=0.6, alpha=0.05, eps=1e-6):
    """Content masks from the frozen features, one per exemplar box."""
    out = []
    f = feats[0]
    fn = F.normalize(f, dim=0)
    for bt in tbboxes:
        oh = max(1, int(bt[3] - bt[1]))
        ow = max(1, int(bt[2] - bt[0]))
        pooled = ops.roi_align(feats, [torch.tensor(bt).unsqueeze(0).float().to(DEVICE)],
                               output_size=(oh, ow), spatial_scale=1.0)[0]
        ys = torch.arange(oh, device=DEVICE, dtype=torch.float32) + 0.5
        xs = torch.arange(ow, device=DEVICE, dtype=torch.float32) + 0.5
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        px = (1 - center_frac) / 2 * ow
        py = (1 - center_frac) / 2 * oh
        pos = (xx >= px) & (xx < ow - px) & (yy >= py) & (yy < oh - py)
        neg = ~pos
        if pos.sum() < 1 or neg.sum() < 1:
            out.append(torch.ones(grid, grid, device=DEVICE))
            continue
        mu_p = pooled[:, pos].mean(1)
        mu_n = pooled[:, neg].mean(1)
        s = (fn * F.normalize(mu_p, dim=0)[:, None, None]).sum(0) - \
            (fn * F.normalize(mu_n, dim=0)[:, None, None]).sum(0)
        _, _, inside = build_regions(bt, grid, center_frac, DEVICE)
        vals = s[inside]
        if vals.numel() < 2:
            out.append(torch.ones(grid, grid, device=DEVICE))
            continue
        med = vals.median()
        mad = (vals - med).abs().median()
        m = torch.sigmoid((s - med) / (1.4826 * mad + eps))
        m = torch.where(inside, m, torch.zeros_like(m))
        out.append(alpha + (1 - alpha) * m)
    return out


def pcm_mask_lists(feats, tbboxes, grid, center_frac=0.6, alpha=0.05, eta=2):
    """Per-exemplar list of masks; the same content mask is applied ``eta`` times."""
    pms = proto_masks(feats, tbboxes, grid, center_frac, alpha)
    return [[m] * eta for m in pms], pms


def build_kernels2(feats, bboxes, mask_lists, use_ell=True):
    """Build the exemplar kernels and their correlation response maps.

    ``mask_lists[bi]`` is a list of full-grid masks, each applied once (ROI
    aligned and multiplied sequentially) on top of the elliptical prior.  This is
    the kernel path used for every number in the manuscript.
    """
    conv_maps, pooled_list, rescaled = [], [], []
    C = feats.shape[1]
    for bi, bbox in enumerate(bboxes):
        bt = torch.tensor(bbox)
        osz = (max(1, int(bt[3] - bt[1])), max(1, int(bt[2] - bt[0])))
        pooled = ops.roi_align(feats, [bt.unsqueeze(0).float().to(DEVICE)],
                               output_size=osz, spatial_scale=1.0)
        if use_ell:
            m = ellipse_coverage(osz[0], osz[1]).unsqueeze(0).unsqueeze(0).to(DEVICE)
        else:
            m = torch.ones(1, 1, *osz, device=DEVICE)
        if mask_lists is not None:
            for mk in (mask_lists[bi] if bi < len(mask_lists) else []):
                mm = mk[None, None].float().to(DEVICE)
                m = m * ops.roi_align(mm, [bt.unsqueeze(0).float().to(DEVICE)],
                                      output_size=osz, spatial_scale=1.0)
        kernel = pooled * m
        pooled_list.append(kernel)
        conv = torch.nn.Conv2d(C, C, kernel_size=osz, padding=0, groups=C, bias=False)
        conv.weight = torch.nn.Parameter(kernel.reshape(C, 1, *osz))
        with torch.no_grad():
            output = conv(feats[0])
        rescaled.append(rescale_bbox(bt, output, feats))
        conv_maps.append(output)
    return conv_maps, pooled_list, rescaled
