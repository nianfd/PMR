"""Complementary Peak--Mass Readout (PMR).

Two statistics are read from the *same* calibrated response map D:

    mass    n_mass = sum_x D(x)                      (Eq. 7)
    peaks   n_peak = #CC_8( L )                      (Eq. 8)

``L`` is the set of local maxima of D on an exemplar-scaled neighbourhood
(r = clip(round(beta * box_r), 2, 8)) whose value exceeds ``t_peak`` times the
maximum of D.  The two estimates are fused in the geometric mean

    n_fuse = sqrt(n_peak * n_mass)                   (Eq. 9)

and a high-count guard keeps the mass estimate as a lower bound when unresolved
crowding can only make the peak count too small:

    n_hat = n_fuse                 if n_mass <  C_g
            max(n_fuse, n_mass)    if n_mass >= C_g  (Eq. 10)

No parameter is fitted and no additional network is used.
"""
import numpy as np
from scipy import ndimage as ndi

STRUCT8 = np.ones((3, 3), bool)


def peak_radius(box_r, beta=0.25, r_min=2, r_max=8):
    """Exemplar-scaled neighbourhood radius, in response-map pixels."""
    if not np.isfinite(box_r):
        box_r = 5.0
    return int(min(r_max, max(r_min, round(beta * box_r))))


def count_peaks(y, radius, t_peak=0.35):
    """Eight-connected components of local maxima above a relative threshold."""
    y = np.clip(np.asarray(y, np.float32), 0, None)
    if y.size == 0:
        return 0
    mxf = ndi.maximum_filter(y, size=2 * int(radius) + 1, mode="nearest")
    sel = (y >= mxf) & (y > t_peak * max(float(y.max()), 1e-12))
    if not sel.any():
        return 0
    return int(ndi.label(sel, structure=STRUCT8)[1])


def fuse_count(n_peak, n_mass, guard=300.0):
    """Geometric peak--mass fusion with the high-count guard."""
    n_peak = float(np.clip(n_peak, 0, None))
    n_mass = float(np.clip(n_mass, 0, None))
    fused = float(np.sqrt(n_peak * n_mass))
    return fused if n_mass < guard else max(fused, n_mass)


def pmr_count(y, box_r, beta=0.25, t_peak=0.35, guard=300.0, mass=None):
    """Convenience wrapper: returns mass, peak and fused counts for one map."""
    y = np.clip(np.asarray(y, np.float64), 0, None)
    n_mass = float(y.sum()) if mass is None else float(mass)
    n_peak = count_peaks(y, peak_radius(box_r, beta), t_peak)
    return {
        "mass": n_mass,
        "peak": float(n_peak),
        "fused": fuse_count(n_peak, n_mass, guard),
        "radius": peak_radius(box_r, beta),
    }
