"""Reference implementation of Complementary Peak--Mass Readout (PMR).

The package contains the three components of the reported framework:

    PCM  (Prototype--Contrastive Kernel Masking)   -> pmr/pcm.py
    DAT  (Density--Adaptive Tiling)                -> pmr/dat.py
    PMR  (Complementary Peak--Mass Readout)        -> pmr/readout.py

`pmr/response.py` builds and calibrates the exemplar-correlation response that
PMR reads out.  The heavy feature extraction (DINOv2 backbone, tiled feature
stitching) is taken from the CountingDINO code base, which this package does not
redistribute; see README.md for the environment setup.
"""

from .config import PCM, PMR, DAT, DAT_INTEGRAL  # noqa: F401

__all__ = ["PCM", "PMR", "DAT", "DAT_INTEGRAL"]
