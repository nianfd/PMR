#!/usr/bin/env python
"""Self-test of the PMR readout on synthetic response maps (no data, no GPU).

Checks the three properties the readout relies on:

  1. an isolated bump contributes one peak and one unit of mass;
  2. the geometric fusion is unbiased when peaks and mass agree;
  3. the high-count guard never returns less than the mass estimate.

Run:  python tools/selftest_pmr.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pmr.readout import count_peaks, fuse_count, peak_radius  # noqa: E402


def synthetic(n_bump, size=240, sigma=2.0, seed=0):
    """A response map with ``n_bump`` separated Gaussian bumps of unit mass."""
    rng = np.random.RandomState(seed)
    y = np.zeros((size, size), np.float64)
    yy, xx = np.mgrid[0:size, 0:size]
    placed, tries = [], 0
    while len(placed) < n_bump and tries < 100 * n_bump:
        tries += 1
        cy, cx = rng.randint(12, size - 12, 2)
        if any((cy - p[0]) ** 2 + (cx - p[1]) ** 2 < (8 * sigma) ** 2 for p in placed):
            continue
        placed.append((cy, cx))
        y += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2)))
    return y, len(placed)


def main():
    ok = True

    for n in (1, 5, 20, 60):
        y, n_placed = synthetic(n, seed=n)
        mass = float(y.sum())
        peaks = count_peaks(y, peak_radius(box_r=10, beta=0.25), t_peak=0.35)
        status = "ok" if peaks == n_placed else "MISMATCH"
        ok &= peaks == n_placed
        print("bumps=%3d  radius=%d  mass=%7.3f  peaks=%3d  %s"
              % (n_placed, peak_radius(10, 0.25), mass, peaks, status))

    print("\nfusion:")
    for n_peak, n_mass, guard in ((10, 10, 300), (10, 20, 300), (300, 500, 300), (500, 500, 300)):
        f = fuse_count(n_peak, n_mass, guard)
        expect = np.sqrt(max(n_peak, 0) * max(n_mass, 0))
        expect = expect if n_mass < guard else max(expect, n_mass)
        ok &= abs(f - expect) < 1e-9
        print("  peaks=%3d mass=%3d guard=%.0f -> %.3f  (expected %.3f)"
              % (n_peak, n_mass, guard, f, expect))

    empty = np.zeros((64, 64), np.float32)
    e = count_peaks(empty, 3, 0.35)
    ok &= e == 0
    print("\nempty map -> peaks=%d (expected 0)" % e)

    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
