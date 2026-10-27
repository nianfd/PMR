# Reproduction notes

This file records the provenance of the released code and the checks that let a
reviewer tell a correct run from a subtly wrong one.

## 1. What the released code computes

Three stages, exactly as in the paper:

1. **PCM** (`pmr/pcm.py`). For each exemplar box, RoI-aligned features give
   central and peripheral prototypes; their cosine-similarity difference defines
   a MAD-normalised sigmoid mask with floor α, and the kernel is
   `E_j ⊙ M_j^η`. No object-head/attention mask is used.
2. **Response** (`pmr/response.py`). The exemplar responses are averaged,
   rescaled and divided by the median exemplar mass — CountingDINO's calibration
   — so that the response integral is a count. The map is kept **uncut**; the cut
   is recorded and applied only to the mass statistic.
3. **PMR** (`pmr/readout.py`). Peak statistic from eight-connected maxima on an
   exemplar-scaled neighbourhood, geometric-mean fusion with the mass, high-count
   guard.

DAT (`pmr/dat.py`) inserts stage 1–2 at 1×, 2× or 4× according to the 1× integral
probe, which is itself computed with the PCM kernel.

## 2. Frozen settings

| Quantity | Value | Selected on |
|---|---|---|
| γ (central side ratio) | 0.6 | FSC-147 val |
| α (mask floor) | 0.05 | FSC-147 val |
| η (mask exponent) | 2 | FSC-147 val |
| β (peak radius factor) | 0.25 | FSC-147 val |
| t_p (relative peak threshold) | 0.35 | FSC-147 val |
| C_g (high-count guard) | 300 | FSC-147 val |
| θ₁, θ₂ (DAT) | 15, 31 | FSC-147 val, for the PMR readout |
| θ₁, θ₂ (DAT, integral readout) | 19, 31 | FSC-147 val, for the integral readout |

No value is re-selected on the test split or on CARPK.

The threshold pair is not the validation minimum: the minimum is (15, 32). The
adopted (15, 31) lies within +0.02 MAE of it and is chosen by the tie-break rule
"smallest θ₂, then smallest θ₁", i.e. the smallest deviation from the reference
4× resolution among statistically indistinguishable settings.

## 3. Expected intermediate values

These are deterministic given the dumps and are printed by the evaluation tools;
they are the fastest way to detect an incomplete or mismatched run.

| Quantity | Validation | Test |
|---|---|---|
| images | 1286 | 1190 |
| routing 1× / 2× / 4× | 257 / 514 / 515 | 184 / 446 / 560 |
| mean final-grid tiles (incl. probe) | 9.01 | 10.03 |
| DAT + integral readout, MAE/RMSE | 21.5431 / 51.8937 | 19.9109 / 63.3267 |
| PCM + DAT + PMR, MAE/RMSE | 17.6697 / 45.6530 | 18.0851 / 61.2841 |
| same, density-map GT | 17.9271 / 47.6082 | 18.0851 / 61.2841 |

| Quantity | CARPK test |
|---|---|
| images | 459 |
| mean GT | 103.5 |
| routing 1× / 2× / 4× | 15 / 19 / 425 |
| mean final-grid tiles (incl. probe) | 15.98 |
| DAT + integral readout, MAE/RMSE | 21.3321 / 28.4629 |
| PCM + DAT + PMR, MAE/RMSE | 13.8546 / 19.8040 |

## 4. The 3477.jpg annotation anomaly

In the released FSC-147 density maps, validation image `3477.jpg` integrates to
13, while its point annotation contains 344 instances. All configurations here
predict ≈533 for that image. Consequences:

* under the point-annotation convention the image contributes
  |533 − 344| ≈ 189 to the MAE sum;
* under the density-map convention it contributes |533 − 13| ≈ 520, i.e. +331
  over 1286 images = **+0.257 MAE** for every configuration, and a larger squared
  error.

This is a property of the released annotation file, not of the method: it shifts
all methods equally and leaves their ranking unchanged. The manuscript reports
the density-map convention, which is why the validation row is 17.93/47.61. The
test split has no such case, so the two conventions give the same numbers there.

## 5. Sources of small numerical drift

The pipeline is deterministic (frozen backbone, no sampling at inference), so a
correct rerun reproduces the reported numbers up to rounding. Residual
differences of ±0.01 can come from:

* GPU architecture and PyTorch/cuDNN version (different reduction orders in
  convolution and RoI-align);
* a different `torchvision` version changing the interpolation used when the
  response is rescaled;
* a different version of the DINOv2 hub repository.

Differences larger than ≈0.1 MAE indicate a protocol mismatch, not numerical
noise. The most common causes are a missing tiling level (the evaluation then
raises), the wrong `--min-gt` for CARPK, or a different exemplar annotation file.

## 6. Scope of the release

Reproduced here: the two rows of the paper that the code produces end to end
(FSC-147 validation/test and CARPK test, all with PCM + DAT + PMR).

Not reproduced here: the other rows of the comparison and ablation tables
(published methods and the component/baseline variants). Those require the
baseline pipelines of the respective papers and are reported from the manuscript.

## 7. Conventions worth knowing

* **Mass before/after the cut.** The response is stored uncut; the mass statistic
  sums the map above its own cut `1/(h·w)` (the largest exemplar kernel area),
  which is the readout used throughout.
* **Peak radius.** `r = clip(round(β · box_r), 2, 8)` where `box_r` is half the
  square root of the median exemplar box area *in the map's own coordinates*.
  Because the radius is recomputed per tiling level, the neighbourhood stays
  consistent in image space across 1×/2×/4×.
* **Fallbacks.** If an exemplar box collapses at a coarse grid (1× or 2×), that
  image falls back to its 4× prediction for the affected level, so the evaluated
  image set is never silently shrunk. The counts are 3 images at 1× and 0 at 2×
  on the test split, and 23 and 1 on validation.
