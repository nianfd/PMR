#!/usr/bin/env python
"""Dump the PCM exemplar-correlation response for every image of a split (GPU).

For each image this writes

  * one record with the calibrated response statistics, including the mass
    statistic ``sum_cut`` and the exemplar scale ``box_r`` used by PMR, and
  * optionally the full response map, needed to compute the peak statistic.

The tiling level is selected with ``--divide``:

    --divide 0 -> 1x  ( 1 tile)
    --divide 1 -> 2x  ( 4 tiles)
    --divide 2 -> 4x  (16 tiles, reference resolution)

Example (FSC-147 test at the reference resolution):

    python tools/dump_responses.py \
        --split test --divide 2 \
        --out  outputs/fsc147/readout_pcm05_test.json \
        --maps-out outputs/fsc147/maps_pcm05_test.npz
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmr.backend import (  # noqa: E402
    VisualBackbone, bboxes_tointeger, convert_4corners_to_x1y1x2y2, get_features, load_json,
)
from pmr.config import PCM
from pmr.geometry import build_regions
from pmr.pcm import pcm_mask_lists
from pmr.response import Cfg, calibrated_response, response_stats


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="directory holding the images")
    ap.add_argument("--annotation", required=True, help="annotation json (points + exemplar boxes)")
    ap.add_argument("--splits", required=True, help="split json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--divide", type=int, default=2, choices=(0, 1, 2),
                    help="0=1x, 1=2x, 2=4x (default, reference resolution)")
    ap.add_argument("--limit", type=int, default=-1, help="number of images (default: all)")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-gt", type=int, default=3,
                    help="skip images with fewer annotated instances (use 1 for CARPK)")
    ap.add_argument("--gamma", type=float, default=PCM["gamma"])
    ap.add_argument("--pcm-alpha", type=float, default=PCM["alpha"])
    ap.add_argument("--eta", type=int, default=PCM["eta"])
    ap.add_argument("--density-dir", default="",
                    help="optional FSC-147 density maps (gt_density_map_adaptive_384_VarV2); "
                         "records N_dm so that the official density-map metric can be reported")
    ap.add_argument("--model-name", default="dinov2_vitl14_reg")
    ap.add_argument("--resize", type=int, default=840)
    ap.add_argument("--out", required=True)
    ap.add_argument("--maps-out", default="", help="npz file for the response maps")
    args = ap.parse_args()

    import torch
    import torchvision.transforms as T
    from PIL import Image
    from tqdm import tqdm

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[warn] running on CPU; the reference runs used a single GPU")

    model = VisualBackbone(args.model_name, img_size=args.resize).to(device).eval()
    transform = T.Compose([
        T.Resize((args.resize, args.resize), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])

    examples = load_json(args.annotation)
    splits = load_json(args.splits)
    keys = list(splits[args.split])
    random.Random(args.seed).shuffle(keys)
    keys = keys[args.start:] if args.limit < 0 else keys[args.start:args.start + args.limit]

    cfg = Cfg()
    cfg.divide_et_impera = args.divide >= 1
    cfg.divide_et_impera_twice = args.divide >= 2
    print("split=%s  images=%d  tiling=%dx (%d tiles)"
          % (args.split, len(keys), 2 ** args.divide, 4 ** args.divide))

    recs, maps = [], {}
    t0 = time.time()
    for i, fn in enumerate(tqdm(keys)):
        entry = examples[fn]
        img_path = os.path.join(args.data_root, fn)
        if not os.path.exists(img_path):
            continue
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        pts = np.asarray(entry["points"], dtype=np.float64)
        n_gt = int(pts.shape[0])
        if n_gt < args.min_gt:
            continue

        feats = get_features(model, img, transform, ["vit_out"],
                             divide_et_impera=cfg.divide_et_impera,
                             divide_et_impera_twice=cfg.divide_et_impera_twice)
        grid = feats.shape[-1]
        ex_bboxes = [convert_4corners_to_x1y1x2y2(b) for b in entry["box_examples_coordinates"]]
        bboxes = np.array([(x1 / w, y1 / h, x2 / w, y2 / h) for x1, y1, x2, y2 in ex_bboxes]) * grid
        bboxes = bboxes_tointeger(bboxes, cfg.remove_bbox_intersection)
        token_bboxes = [(x1 / w * grid, y1 / h * grid, x2 / w * grid, y2 / h * grid)
                        for x1, y1, x2, y2 in ex_bboxes]

        # An exemplar box that collapses at this grid cannot define a kernel.
        valid = False
        for tb in token_bboxes:
            pos, neg, _ = build_regions(tb, grid, args.gamma, device)
            if pos.sum() >= 1 and neg.sum() >= 1:
                valid = True
                break
        if not valid:
            continue

        mask_lists, _ = pcm_mask_lists(feats, token_bboxes, grid,
                                       args.gamma, args.pcm_alpha, args.eta)
        res = calibrated_response(feats, bboxes, mask_lists, cfg)
        if res is None:
            continue
        st = response_stats(res)

        n_dm = None
        if args.density_dir:
            dm_path = os.path.join(args.density_dir, os.path.splitext(fn)[0] + ".npy")
            if os.path.exists(dm_path):
                n_dm = float(np.load(dm_path).sum())

        slot = "a%04d" % i
        recs.append({
            "slot": slot, "img": fn, "N_gt": n_gt, "N_dm": n_dm,
            "w": w, "h": h,
            "cut": st.cut, "y_shape": st.shape, "norm_coeff": st.norm_coeff,
            "box_r": st.box_r, "mass": st.mass, "max": st.max,
            "divide": args.divide,
        })
        if args.maps_out:
            maps[slot] = st.y.astype(np.float16)
        if len(recs) % 100 == 0:
            print("  %d done, %.1fs" % (len(recs), time.time() - t0), flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"meta": vars(args), "records": recs}, f, indent=1)
    if args.maps_out:
        np.savez_compressed(args.maps_out, **maps)
        print("maps -> %s (%d)" % (args.maps_out, len(maps)))
    print("saved %s  n=%d  %.1fs" % (args.out, len(recs), time.time() - t0))


if __name__ == "__main__":
    main()
