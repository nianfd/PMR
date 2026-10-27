"""Token-grid region helpers (central positive region / peripheral ring)."""
import torch


def build_regions(bbox, hw, center_frac=0.6, device="cuda"):
    """bbox = (x1, y1, x2, y2) in token coordinates.

    Returns three boolean [hw, hw] tensors: the central region, the peripheral
    ring inside the box, and the box support itself.
    """
    ys = torch.arange(hw, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(hw, device=device, dtype=torch.float32) + 0.5
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    x1, y1, x2, y2 = [float(v) for v in bbox]
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    inside = (xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)
    pad = (1.0 - center_frac) / 2.0
    pos = inside & (xx >= x1 + pad * bw) & (xx < x2 - pad * bw) & \
        (yy >= y1 + pad * bh) & (yy < y2 - pad * bh)
    neg = inside & (~pos)
    return pos, neg, inside
