# Data preparation

The datasets are not redistributed with this repository. Both are public; download
them from their official sources and arrange them as shown below. The scripts only
need the paths, which are passed through environment variables (`FSC147_ROOT`,
`CARPK_ROOT`) or explicit command-line flags.

## FSC-147

Source: the FSC-147 release distributed with *Learning to Count Everything*
(CVPR 2021), <https://github.com/cvlab-stonybrook/LearningToCountEverything>.
The repository links the image archive, the annotations and the split file.

```
$FSC147_ROOT/
├── images_384_VarV2/                       # 6,135 images, 384 px short side
│   ├── 1.jpg
│   └── ...
├── annotation_FSC147_384.json              # per image: points + box_examples_coordinates
├── Train_Test_Val_FSC_147.json             # {"train": [...], "val": [1286], "test": [1190]}
└── gt_density_map_adaptive_384_VarV2/      # per image: <stem>.npy density map
    ├── 1.npy
    └── ...
```

* The density-map folder is part of the official release and is also what the
  CountingDINO evaluation reads. It is **required for the reported validation
  number** (17.93); without it the validation MAE becomes 17.67 (point
  annotations, see `README.md` §7).
* `annotation_FSC147_384.json` must contain, for every image, the key `points`
  (one entry per annotated instance) and `box_examples_coordinates` (three
  exemplar boxes, in corner format). No other field is used.

Quick check:

```bash
ls "$FSC147_ROOT/images_384_VarV2" | wc -l          # 6135
python - <<'PY'
import json, os
root = os.environ["FSC147_ROOT"]
sp = json.load(open(os.path.join(root, "Train_Test_Val_FSC_147.json")))
ann = json.load(open(os.path.join(root, "annotation_FSC147_384.json")))
print("splits:", {k: len(v) for k, v in sp.items()})   # val 1286, test 1190
print("annotations:", len(ann))
print("density maps:", len(os.listdir(os.path.join(root, "gt_density_map_adaptive_384_VarV2"))))
PY
```

## CARPK

Source: CARPK, <https://lzen.github.io/CARPK.html>. The version used in the
class-agnostic counting literature (and in CountingDINO) is the one with 1,448
images, split into 989 training and 459 test images.

```
$CARPK_ROOT/
├── images/                                                  # 1,448 images
│   ├── 1.jpg
│   └── ...
├── annotation_CARPK_3exemplars_11seed_5threshold.json        # FSC-147 JSON format
└── splits_carpk.json                                         # {"train": [...], "test": [...459...]}
```

* The annotation file uses the same schema as FSC-147: `points` holds one entry per
  annotated vehicle, `box_examples_coordinates` holds three exemplar boxes. The
  filename refers to the exemplar sampling protocol used when the file was built
  (three exemplars per image). **Use the file from the same release** — the count
  reported in the paper depends on the exemplar boxes, and a different sampling
  will change the numbers.
* `splits_carpk.json` only needs the keys `train` and `test`.
* Density maps are **not** needed for CARPK: the ground truth is the number of
  annotated points.

Quick check:

```bash
ls "$CARPK_ROOT/images" | wc -l                     # 1448
python - <<'PY'
import json, os
root = os.environ["CARPK_ROOT"]
sp = json.load(open(os.path.join(root, "splits_carpk.json")))
print({k: len(v) for k, v in sp.items()})           # train 989, test 459
PY
```

## Models

No checkpoint needs to be downloaded by hand. The backbone is
`dinov2_vitl14_reg`, fetched by `torch.hub` from
`facebookresearch/dinov2` on first use and cached in `~/.cache/torch/hub`.
For an offline machine, pre-populate that cache or set `TORCH_HOME` to a
directory that already contains it.

## Disk and memory

* FSC-147 images: a few GB, depending on the mirror you download.
* Density maps: ≈1 GB.
* CARPK images: under 1 GB.
* Dumps written by the scripts (three tiling levels, both splits): ≈0.5 GB for
  FSC-147, ≈0.1 GB for CARPK. They are regenerated, not versioned.
