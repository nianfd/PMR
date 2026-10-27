#!/usr/bin/env python
"""Evaluate PCM + DAT + PMR on FSC-147 (CPU only).

Reads the responses and peak counts produced by ``tools/dump_responses.py`` and
``tools/dump_peak_counts.py``, selects the two DAT thresholds on the validation
split, freezes them, and reports the test split.

Protocol, as in the manuscript:
  * (theta1, theta2) is chosen on validation over theta1 in [5, 79],
    theta2 in [8, 179], with theta2 > theta1 + 3;
  * among all pairs within +0.02 MAE of the validation minimum, the one with the
    smallest theta2 (then smallest theta1) is adopted -- the most conservative
    downgrade policy on a flat validation surface;
  * the frozen pair is applied unchanged to test.

Reported row (Table 2): validation 17.93/47.61, test 18.09/61.28.

Example:

    python tools/eval_fsc147.py --dump-dir outputs/fsc147
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmr.config import DAT, PMR  # noqa: E402
from pmr.dat import tiles_used  # noqa: E402
from pmr.dumps import LEVELS, apply_levels, fused, level_select, stack, tune_theta  # noqa: E402
from pmr.metrics import mae, rmse  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--beta", type=float, default=PMR["beta"])
    ap.add_argument("--t-peak", type=float, default=PMR["t_peak"])
    ap.add_argument("--guard", type=float, default=PMR["guard"])
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--theta1", type=int, default=None, help="skip tuning, use this pair")
    ap.add_argument("--theta2", type=int, default=None, help="skip tuning, use this pair")
    args = ap.parse_args()

    print("PMR readout: beta=%.2f  t_peak=%.2f  guard=%.0f"
          % (args.beta, args.t_peak, args.guard))
    kv, V = stack(args.dump_dir, "val", args.beta, args.t_peak)
    kt, T = stack(args.dump_dir, "test", args.beta, args.t_peak)
    FV = {d: fused(V, d, args.guard) for d in LEVELS}
    FT = {d: fused(T, d, args.guard) for d in LEVELS}

    if args.theta1 is not None and args.theta2 is not None:
        t1, t2 = args.theta1, args.theta2
        print("thresholds supplied on the command line: (%d, %d)" % (t1, t2))
    else:
        pick, best, n_plateau = tune_theta(V, FV, V[4]["gt"], args.tol)
        t1, t2 = pick[1], pick[2]
        print("validation grid: minimum %.4f at (%d, %d); plateau(<=+%.2f) = %d pairs"
              % (best[0], best[1], best[2], args.tol, n_plateau))
        print("selected on validation: (%d, %d) -> MAE %.4f" % (t1, t2, pick[0]))

    for split, A, F in (("val", V, FV), ("test", T, FT)):
        idx = level_select(A, t1, t2)
        pred = apply_levels(F, idx)
        integral = apply_levels({d: A[d]["I"] for d in LEVELS}, idx)
        gt = A[4]["gt"]
        print("\n%s  n=%d  tiling 1x/2x/4x = %d/%d/%d  mean tiles %.2f"
              % (split.upper(), len(gt), int((idx == 0).sum()), int((idx == 1).sum()),
                 int((idx == 2).sum()), tiles_used(A[1]["I"], t1, t2, include_probe=True)))
        print("  DAT, integral readout   MAE %8.4f  RMSE %8.4f"
              % (mae(integral, gt), rmse(integral, gt)))
        print("  PCM + DAT + PMR         MAE %8.4f  RMSE %8.4f" % (mae(pred, gt), rmse(pred, gt)))
        if np.isfinite(A[4]["gt_dm"]).all():
            gd = A[4]["gt_dm"]
            print("  PCM + DAT + PMR, density-map GT   MAE %8.4f  RMSE %8.4f"
                  % (mae(pred, gd), rmse(pred, gd)))
        else:
            print("  (density-map GT unavailable: dump with --density-dir to report it)")

    print("\nreference: validation 17.9271/47.6082 (density-map GT),"
          " test 18.0851/61.2841, theta=(%d,%d)" % (DAT["theta1"], DAT["theta2"]))


if __name__ == "__main__":
    main()
