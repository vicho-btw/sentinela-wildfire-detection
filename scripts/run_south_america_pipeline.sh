#!/usr/bin/env bash
set -euo pipefail

REGION="${REGION:-south_america}"
DATA_ROOT="${SENTINELA_DATA_ROOT:-/workspace/data/goes_fire}"
SAMPLES_PER_CLASS="${SAMPLES_PER_CLASS:-10000}"
SMOKE_SAMPLES_PER_CLASS="${SMOKE_SAMPLES_PER_CLASS:-100}"
SMOKE_MAX_GOES_FILES="${SMOKE_MAX_GOES_FILES:-5}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-16}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_TRAIN="${RUN_TRAIN:-1}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-nominal}"

if [[ -z "${FIRMS_MAP_KEY:-}" ]]; then
  echo "Set FIRMS_MAP_KEY first." >&2
  exit 2
fi

mkdir -p "$DATA_ROOT"

echo "== Download FIRMS bootstrap labels =="
python3 scripts/download_firms_bootstrap_labels.py \
  --region "$REGION" \
  --data-root "$DATA_ROOT" \
  --min-confidence "$MIN_CONFIDENCE"

echo "== Ingest labels =="
python3 scripts/ingest_regional_labels.py \
  --region "$REGION" \
  --data-root "$DATA_ROOT" \
  --labels "$DATA_ROOT/$REGION/raw/labels/firms_bootstrap_events.csv"

if [[ "$RUN_SMOKE" == "1" ]]; then
  echo "== Streaming smoke sample build =="
  python3 scripts/build_regional_goes_samples_streaming.py \
    --region "$REGION" \
    --data-root "$DATA_ROOT" \
    --samples-per-class "$SMOKE_SAMPLES_PER_CLASS" \
    --max-goes-files "$SMOKE_MAX_GOES_FILES"

  echo "== Clear smoke samples before real build =="
  rm -rf "$DATA_ROOT/$REGION/samples"
  rm -f "$DATA_ROOT/$REGION/manifest.csv"
fi

echo "== Streaming real sample build =="
python3 scripts/build_regional_goes_samples_streaming.py \
  --region "$REGION" \
  --data-root "$DATA_ROOT" \
  --samples-per-class "$SAMPLES_PER_CLASS"

if [[ "$RUN_TRAIN" == "1" ]]; then
  echo "== Train regional model =="
  python3 scripts/train_regional_goes.py \
    --region "$REGION" \
    --data-root "$DATA_ROOT" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE"

  echo "== Evaluate regional model =="
  python3 scripts/evaluate_regional_goes.py \
    --region "$REGION" \
    --data-root "$DATA_ROOT" \
    --checkpoint "models/regional/$REGION/best.pt" \
    --split test \
    --out-json "$DATA_ROOT/$REGION/evaluation/best_test.json"

  echo "== Write benchmark report =="
  python3 scripts/write_regional_benchmark_report.py \
    --region "$REGION" \
    --data-root "$DATA_ROOT" \
    --eval-json "$DATA_ROOT/$REGION/evaluation/best_test.json" \
    --history-json "models/regional/$REGION/history.json"
fi

echo "done region=$REGION data_root=$DATA_ROOT"
