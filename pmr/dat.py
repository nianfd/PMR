"""Density-Adaptive Tiling (DAT).

A single coarse (1x) integral pass is used as a density probe.  Two thresholds
map the probe count to a tiling level:

    n_probe <  theta1            -> 1x  (1 tile,  reuse the coarse response)
    theta1 <= n_probe < theta2   -> 2x  (4 tiles)
    n_probe >= theta2            -> 4x  (16 tiles)

The thresholds are selected on FSC-147 validation for the final readout and then
frozen; test and CARPK reuse them unchanged.
"""
import numpy as np

LEVELS = (1, 2, 4)
TILES = (1, 4, 16)


def route(n_probe, theta1, theta2):
    """Return the level index (0 -> 1x, 1 -> 2x, 2 -> 4x)."""
    n_probe = np.asarray(n_probe, dtype=np.float64)
    idx = np.where(n_probe < theta1, 0, np.where(n_probe < theta2, 1, 2))
    return idx if idx.ndim else int(idx)


def level_of(n_probe, theta1, theta2):
    idx = route(n_probe, theta1, theta2)
    return np.asarray(LEVELS, dtype=int)[idx]


def tiles_used(n_probe, theta1, theta2, include_probe=False):
    """Final-grid tile allocation, optionally including the coarse probe pass."""
    idx = np.atleast_1d(route(n_probe, theta1, theta2))
    t = np.asarray(TILES, dtype=float)[idx]
    if include_probe:
        t = t + (idx > 0)
    return float(t.mean())
