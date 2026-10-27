"""Thin adapter over the CountingDINO reference code.

The frozen DINOv2 backbone, the tiled feature extraction and the response
rescaling utilities are part of the CountingDINO release
(https://github.com/lorebianchi98/CountingDINO).  We import them instead of
copying them, so that the backbone behaves exactly as in the original code.

Set the environment variable COUNTINGDINO_ROOT to the root of a CountingDINO
checkout before importing this module, or place that checkout next to this
repository (../CountingDINO, ./third_party/CountingDINO).
"""
import os
import sys

_CANDIDATES = [
    os.environ.get("COUNTINGDINO_ROOT", ""),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "CountingDINO"),
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "CountingDINO"),
]


def _locate():
    for cand in _CANDIDATES:
        if not cand:
            continue
        cand = os.path.abspath(cand)
        if os.path.isfile(os.path.join(cand, "src", "utils.py")):
            return cand
    raise RuntimeError(
        "CountingDINO source tree not found.\n"
        "Clone https://github.com/lorebianchi98/CountingDINO and either\n"
        "  export COUNTINGDINO_ROOT=/path/to/CountingDINO\n"
        "or place the checkout at ./third_party/CountingDINO.\n"
        "Expected file: <root>/src/utils.py"
    )


COUNTINGDINO_ROOT = _locate()
if COUNTINGDINO_ROOT not in sys.path:
    sys.path.insert(0, COUNTINGDINO_ROOT)

from src.model import VisualBackbone  # noqa: E402
from src.utils import (  # noqa: E402
    bboxes_tointeger,
    convert_4corners_to_x1y1x2y2,
    ellipse_coverage,
    get_features,
    load_json,
    rescale_bbox,
    rescale_tensor,
    resize_conv_maps,
)

__all__ = [
    "COUNTINGDINO_ROOT",
    "VisualBackbone",
    "bboxes_tointeger",
    "convert_4corners_to_x1y1x2y2",
    "ellipse_coverage",
    "get_features",
    "load_json",
    "rescale_bbox",
    "rescale_tensor",
    "resize_conv_maps",
]
