"""Readers for the dumped responses and peak counts.

The on-disk layout produced by the two dump tools is

    <dump>/readout_pcm05_<split>.json        + <dump>/dar_counts_pcm_<split>.npz          (4x)
    <dump>/readout_pcm05_<split>_1x.json     + <dump>/dar_counts_pcm_<split>_1x.npz       (1x)
    <dump>/readout_pcm05_<split>_2x.json     + <dump>/dar_counts_pcm_<split>_2x.npz       (2x)
"""
import json
import os

import numpy as np

from .config import PMR
from .metrics import mae

LEVELS = (1, 2, 4)


def level_paths(dump_dir, split, level):
    if level == 4:
        return (os.path.join(dump_dir, "readout_pcm05_%s.json" % split),
                os.path.join(dump_dir, "dar_counts_pcm_%s.npz" % split))
    return (os.path.join(dump_dir, "readout_pcm05_%s_%dx.json" % (split, level)),
            os.path.join(dump_dir, "dar_counts_pcm_%s_%dx.npz" % (split, level)))


def load_level(dump_dir, split, level, beta=PMR["beta"], t_peak=PMR["t_peak"]):
    """Per-image mass, peak count and ground truth for one tiling level."""
    rj, cj = level_paths(dump_dir, split, level)
    if not (os.path.exists(rj) and os.path.exists(cj)):
        return None
    with open(rj) as f:
        recs = json.load(f)["records"]
    z = np.load(cj)
    counts = z["counts"]
    if "tag" in z.files:                      # 4x grid also holds absolute radii
        val, tt = z["val"], z["t"]
        idx = {(round(float(val[k]), 2), round(float(tt[k]), 2)): k
               for k in range(counts.shape[0]) if str(z["tag"][k]) == "rho"}
    else:
        rho, tt = z["rho"], z["t"]
        idx = {(round(float(rho[k]), 2), round(float(tt[k]), 2)): k
               for k in range(counts.shape[0])}
    key = (round(beta, 2), round(t_peak, 2))
    if key not in idx:
        raise KeyError("configuration %s absent from %s" % (key, cj))
    kk = idx[key]
    out = {}
    for i, r in enumerate(recs):
        out[r["img"]] = {
            "N_gt": float(r["N_gt"]),
            "N_dm": (float(r["N_dm"]) if r.get("N_dm") is not None else None),
            "I": float(r.get("mass", r.get("sum_cut"))),
            "D": float(counts[kk, i]),
        }
    return out


def stack(dump_dir, split, beta=PMR["beta"], t_peak=PMR["t_peak"]):
    """Align the three levels on the 4x image set, falling back to 4x if needed."""
    L = {d: load_level(dump_dir, split, d, beta, t_peak) for d in LEVELS}
    if any(v is None for v in L.values()):
        missing = [d for d in LEVELS if L[d] is None]
        raise SystemExit("[missing dumps] split=%s levels=%s in %s\n"
                         "run tools/dump_responses.py and tools/dump_peak_counts.py first"
                         % (split, missing, dump_dir))
    keys = sorted(L[4])
    A = {}
    for d in LEVELS:
        g, gd, I, D = [], [], [], []
        for k in keys:
            src = L[d].get(k, L[4][k])
            g.append(src["N_gt"])
            gd.append(src["N_dm"] if src["N_dm"] is not None else np.nan)
            I.append(src["I"])
            D.append(src["D"])
        A[d] = {"gt": np.array(g, float), "gt_dm": np.array(gd, float),
                "I": np.array(I, float), "D": np.array(D, float)}
    return keys, A


def fused(A, d, guard=PMR["guard"]):
    """Geometric peak--mass fusion with the high-count guard, Eq. (9)--(10)."""
    peaks = np.clip(A[d]["D"], 0, None)
    mass = np.clip(A[d]["I"], 0, None)
    f = np.sqrt(peaks * mass)
    return np.where(A[d]["I"] < guard, f, np.maximum(f, A[d]["I"]))


def level_select(A, theta1, theta2):
    """0 -> 1x, 1 -> 2x, 2 -> 4x, from the 1x integral probe count."""
    return np.where(A[1]["I"] < theta1, 0, np.where(A[1]["I"] < theta2, 1, 2))


def apply_levels(values, idx):
    return np.where(idx == 0, values[1], np.where(idx == 1, values[2], values[4]))


def tune_theta(A, F, gt, tol=0.02, t1_range=range(5, 80), t2_range=range(8, 180)):
    """Validation search for (theta1, theta2) with the plateau tie-break."""
    best, surface = None, []
    for t1 in t1_range:
        for t2 in t2_range:
            if t2 <= t1 + 3:
                continue
            m = mae(apply_levels(F, level_select(A, t1, t2)), gt)
            surface.append((m, t1, t2))
            if best is None or m < best[0]:
                best = (m, t1, t2)
    plateau = [s for s in surface if s[0] <= best[0] + tol]
    pick = sorted(plateau, key=lambda s: (s[2], s[1]))[0]
    return pick, best, len(plateau)
