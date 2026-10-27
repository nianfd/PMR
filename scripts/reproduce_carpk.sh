#!/usr/bin/env bash
# Reproduce the CARPK row of Table 6: PCM + DAT + PMR.
#
#   test  13.8546 MAE / 19.8040 RMSE   with the FSC-147 thresholds (15, 31)
#
# Nothing is re-selected on CARPK: the PMR scalars and the DAT thresholds are the
# ones frozen on FSC-147 validation.  GPU required for step 1; steps 2-3 are CPU only.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export COUNTINGDINO_ROOT="${COUNTINGDINO_ROOT:-$ROOT/third_party/CountingDINO}"
export PYTHONPATH="$COUNTINGDINO_ROOT:$ROOT:${PYTHONPATH:-}"
PY="${PYTHON:-python}"

: "${CARPK_ROOT:?set CARPK_ROOT to the directory holding the CARPK release}"

IMG="$CARPK_ROOT/images"
ANN="${CARPK_ANNOTATION:-$CARPK_ROOT/annotation_CARPK_3exemplars_11seed_5threshold.json}"
SPL="${CARPK_SPLITS:-$CARPK_ROOT/splits_carpk.json}"
OUT="${OUT_DIR:-$ROOT/outputs/carpk}"
mkdir -p "$OUT"

echo "== CARPK test, three tiling levels =="
for divide in 2 1 0; do
  case $divide in
    2) tag="" ;;
    1) tag="_2x" ;;
    0) tag="_1x" ;;
  esac
  echo "-- ${tag:-4x} --"
  "$PY" tools/dump_responses.py \
    --data-root "$IMG" --annotation "$ANN" --splits "$SPL" \
    --split test --divide "$divide" --min-gt 1 \
    --out "$OUT/readout_pcm05_test${tag}.json" \
    --maps-out "$OUT/maps_pcm05_test${tag}.npz"

  "$PY" tools/dump_peak_counts.py \
    --records "$OUT/readout_pcm05_test${tag}.json" \
    --maps "$OUT/maps_pcm05_test${tag}.npz" \
    --out "$OUT/dar_counts_pcm_test${tag}.npz"
done

echo "== evaluation (FSC-147 thresholds, no CARPK tuning) =="
"$PY" tools/eval_carpk.py --dump-dir "$OUT"
