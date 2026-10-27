#!/usr/bin/env bash
# Reproduce the FSC-147 row of Table 2: PCM + DAT + PMR.
#
#   validation  17.93 MAE / 47.61 RMSE
#   test        18.09 MAE / 61.28 RMSE
#
# GPU required for step 1 (three tiling levels per split); steps 2-3 are CPU only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export COUNTINGDINO_ROOT="${COUNTINGDINO_ROOT:-$ROOT/third_party/CountingDINO}"
export PYTHONPATH="$COUNTINGDINO_ROOT:$ROOT:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

: "${FSC147_ROOT:?set FSC147_ROOT to the directory holding the FSC-147 release}"

IMG="$FSC147_ROOT/images_384_VarV2"
ANN="$FSC147_ROOT/annotation_FSC147_384.json"
SPL="$FSC147_ROOT/Train_Test_Val_FSC_147.json"
DM="$FSC147_ROOT/gt_density_map_adaptive_384_VarV2"
OUT="${OUT_DIR:-$ROOT/outputs/fsc147}"
mkdir -p "$OUT"

DM_ARG=()
if [ -d "$DM" ]; then
  DM_ARG=(--density-dir "$DM")
else
  echo "[warn] $DM not found: the validation MAE will be 17.67 (point annotations)"
  echo "       instead of the reported 17.93 (official density maps)."
fi

for split in val test; do
  for divide in 2 1 0; do
    case $divide in
      2) tag="" ;;
      1) tag="_2x" ;;
      0) tag="_1x" ;;
    esac
    echo "== $split ${tag:-4x} =="
    "$PY" tools/dump_responses.py \
      --data-root "$IMG" --annotation "$ANN" --splits "$SPL" \
      --split "$split" --divide "$divide" ${DM_ARG[@]+"${DM_ARG[@]}"} \
      --out "$OUT/readout_pcm05_${split}${tag}.json" \
      --maps-out "$OUT/maps_pcm05_${split}${tag}.npz"

    "$PY" tools/dump_peak_counts.py \
      --records "$OUT/readout_pcm05_${split}${tag}.json" \
      --maps "$OUT/maps_pcm05_${split}${tag}.npz" \
      --out "$OUT/dar_counts_pcm_${split}${tag}.npz"
  done
done

echo "== evaluation (thresholds selected on validation, frozen for test) =="
"$PY" tools/eval_fsc147.py --dump-dir "$OUT"
