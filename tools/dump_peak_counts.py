#!/usr/bin/env python
"""Compute the peak statistic of PMR over a (beta, t_peak) grid (CPU only).

Inputs are the outputs of ``tools/dump_responses.py`` for one tiling level:

    readout_pcm05_<split>.json + maps_pcm05_<split>.npz          (4x)
    readout_pcm05_<split>_1x.json + maps_pcm05_<split>_1x.npz    (1x)
    readout_pcm05_<split>_2x.json + maps_pcm05_<split>_2x.npz    (2x)

Output is an npz with the connected-peak count of every image for every
(beta, t_peak) pair.  The reported configuration is beta=0.25, t_peak=0.35.

Example:

    python tools/dump_peak_counts.py \
        --records outputs/fsc147/readout_pcm05_test.json \
        --maps    outputs/fsc147/maps_pcm05_test.npz \
        --out     outputs/fsc147/dar_counts_pcm_test.npz
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmr.readout import count_peaks, peak_radius  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", required=True)
    ap.add_argument("--maps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--betas", default="0.20,0.25,0.30,0.35,0.40,0.50,0.60")
    ap.add_argument("--t-peaks", default="0.20,0.25,0.30,0.35,0.40,0.45,0.50")
    args = ap.parse_args()

    betas = [float(x) for x in args.betas.split(",")]
    t_peaks = [float(x) for x in args.t_peaks.split(",")]

    with open(args.records) as f:
        recs = json.load(f)["records"]
    maps = np.load(args.maps)
    n = len(recs)
    cache = {}
    rows, keys = [], []
    for beta in betas:
        for t in t_peaks:
            out = np.empty(n)
            for i, r in enumerate(recs):
                slot = r["slot"]
                if slot not in cache:
                    cache[slot] = np.clip(maps[slot].astype(np.float32), 0, None)
                y = cache[slot]
                out[i] = float(count_peaks(y, peak_radius(r.get("box_r", np.nan), beta), t))
            rows.append(out)
            keys.append((beta, t))
            print("  beta=%.2f t=%.2f done" % (beta, t), flush=True)

    counts = np.stack(rows)
    np.savez_compressed(args.out, counts=counts,
                        rho=np.array([k[0] for k in keys]),
                        t=np.array([k[1] for k in keys]))
    print("[saved] %s  shape=%s" % (args.out, counts.shape))


if __name__ == "__main__":
    main()
