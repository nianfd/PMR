#!/usr/bin/env python
"""Evaluate PCM + DAT + PMR on CARPK test (CPU only).

Nothing is re-selected on CARPK: the PMR scalars and the DAT thresholds are the
ones frozen on FSC-147, and the same frozen DINOv2 backbone produces the
responses.  The script reports the readout at every fixed tiling level and the
routed PCM + DAT + PMR configuration.

Reported row (Table 6): 13.8546 MAE / 19.8040 RMSE with theta = (15, 31).
The same routing with theta = (19, 31) gives 13.8883 / 19.8149.

Example:

    python tools/eval_carpk.py --dump-dir outputs/carpk
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmr.config import DAT, PMR  # noqa: E402
from pmr.dat import tiles_used  # noqa: E402
from pmr.dumps import LEVELS, apply_levels, fused, level_select, stack  # noqa: E402
from pmr.metrics import mae, rmse  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--beta", type=float, default=PMR["beta"])
    ap.add_argument("--t-peak", type=float, default=PMR["t_peak"])
    ap.add_argument("--guard", type=float, default=PMR["guard"])
    ap.add_argument("--theta1", type=int, default=DAT["theta1"])
    ap.add_argument("--theta2", type=int, default=DAT["theta2"])
    args = ap.parse_args()

    print("PMR readout: beta=%.2f  t_peak=%.2f  guard=%.0f"
          % (args.beta, args.t_peak, args.guard))
    print("DAT thresholds frozen on FSC-147 validation: (%d, %d)" % (args.theta1, args.theta2))

    keys, A = stack(args.dump_dir, "test", args.beta, args.t_peak)
    F = {d: fused(A, d, args.guard) for d in LEVELS}
    gt = A[4]["gt"]
    print("\nCARPK test  n=%d  mean GT %.1f" % (len(gt), gt.mean()))

    print("\nfixed tiling, integral and PMR readout:")
    for d in LEVELS:
        print("  %2dx  integral MAE %7.4f RMSE %7.4f   PMR MAE %7.4f RMSE %7.4f"
              % (d, mae(A[d]["I"], gt), rmse(A[d]["I"], gt), mae(F[d], gt), rmse(F[d], gt)))

    idx = level_select(A, args.theta1, args.theta2)
    pred = apply_levels(F, idx)
    integral = apply_levels({d: A[d]["I"] for d in LEVELS}, idx)
    print("\nrouted (theta = %d, %d): 1x/2x/4x = %d/%d/%d  mean tiles %.2f"
          % (args.theta1, args.theta2, int((idx == 0).sum()), int((idx == 1).sum()),
             int((idx == 2).sum()), tiles_used(A[1]["I"], args.theta1, args.theta2,
                                               include_probe=True)))
    print("  DAT, integral readout   MAE %7.4f  RMSE %7.4f" % (mae(integral, gt), rmse(integral, gt)))
    print("  PCM + DAT + PMR         MAE %7.4f  RMSE %7.4f" % (mae(pred, gt), rmse(pred, gt)))

    print("\nreference (theta = 15, 31): 13.8546 / 19.8040 ;"
          " (theta = 19, 31): 13.8883 / 19.8149")


if __name__ == "__main__":
    main()
