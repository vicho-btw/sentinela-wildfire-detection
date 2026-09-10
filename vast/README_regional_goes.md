# Vast.ai Regional GOES Training

This is the current path for the first regional Sentinela-ModelS wildfire model.
It uses raw GOES ABI channels and generic regional event labels. FIRMS/VIIRS is
not part of the core labeling pipeline; keep it only as optional external
reference/evaluation metadata.

## Setup

```bash
git clone <repo-url> /workspace/condor
cd /workspace/condor/Sentinela-ModelS

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

export SENTINELA_DATA_ROOT=/workspace/data/goes_fire
mkdir -p "$SENTINELA_DATA_ROOT"
```

## One-Command Pipeline

```bash
export FIRMS_MAP_KEY="<your NASA FIRMS map key>"
export SENTINELA_DATA_ROOT=/workspace/data/goes_fire
export SAMPLES_PER_CLASS=10000
export EPOCHS=20
export BATCH_SIZE=16

bash scripts/run_south_america_pipeline.sh
```

This runs label download, label ingestion, streaming smoke build, real streaming
sample build, training, evaluation, and a benchmark-style report. The report is
explicit that FIRMS is the bootstrap label source; it is not yet an independent
earlier-than-NASA lead-time claim.

## Ingest Labels

Input labels can be CSV or JSONL and must include:

```text
event_id,event_time_utc,lat,lon,label,label_name,source,confidence
```

Labels:

Training target:

```text
0 no_fire      = negative + hard_negative
1 fire_signal  = active_fire + early_fire_signal
```

Original diagnostic labels:

```text
0 negative
1 active_fire
2 early_fire_signal
3 hard_negative
4 uncertain
```

Command:

Bootstrap labels from NASA FIRMS hotspots:

```bash
export FIRMS_MAP_KEY="<your NASA FIRMS map key>"

python3 scripts/download_firms_bootstrap_labels.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --min-confidence nominal
```

Normalize those bootstrap labels:

```bash
python3 scripts/ingest_regional_labels.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --labels "$SENTINELA_DATA_ROOT/south_america/raw/labels/firms_bootstrap_events.csv"
```

## Smoke Sampled GOES Download

The downloader does not archive every GOES file. It builds a 50/50 sample plan:

- wildfire targets from the ingested regional labels
- non-fire targets sampled throughout the region bbox

Then it downloads only the GOES files needed for those targets.

Small smoke test:

```bash
python3 scripts/download_goes_region.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --samples-per-class 10 \
  --max-targets 20
```

## Two-Year South America Sample Download

The config defaults to the last two years. This command uses all positive labels
in that range and creates an equal number of non-fire targets:

```bash
python3 scripts/download_goes_region.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" 
```

For a fixed-size balanced run:

```bash
python3 scripts/download_goes_region.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --samples-per-class 50000
```

Outputs:

```text
data/goes_fire/south_america/raw/download_manifest.csv
data/goes_fire/south_america/processed/sample_targets.csv
```

## Build Samples

Preferred cheap-instance streaming path:

```bash
python3 scripts/build_regional_goes_samples_streaming.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --samples-per-class 10000
```

This downloads one GOES file, extracts all selected patches from that file,
writes temporal `.npz` samples, and deletes the raw NetCDF files before moving
to the next sequence. Use this for CPU/data-prep instances.

Default temporal sample contract:

```text
x shape: [4, 9, 64, 64]
offsets: [-30, -20, -10, 0] minutes
```

For legacy single-frame samples, pass `--temporal-offsets 0` to sample build,
train, and evaluation commands.

Raw 9-channel model:

```bash
python3 scripts/build_regional_goes_samples.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT"
```

Raw plus derived channels:

```bash
python3 scripts/build_regional_goes_samples.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --include-derived
```

Samples are written under `data/goes_fire/south_america/samples/` and indexed in
`data/goes_fire/south_america/manifest.csv`.

## Train

Raw 9-channel South America model:

```bash
python3 scripts/train_regional_goes.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --epochs 20 \
  --batch-size 16
```

The model is binary by default. It maps `active_fire` and `early_fire_signal` to
`fire_signal`; `negative` and `hard_negative` to `no_fire`; and gives
`uncertain` zero weight unless `--uncertain-weight` is overridden. Add
`--drop-uncertain` to remove uncertain rows entirely.

Best checkpoint:

```text
models/regional/south_america/best.pt
```

Raw plus derived channels:

```bash
python3 scripts/train_regional_goes.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --include-derived \
  --epochs 20 \
  --batch-size 16
```

## Evaluate

```bash
python3 scripts/evaluate_regional_goes.py \
  --region south_america \
  --data-root "$SENTINELA_DATA_ROOT" \
  --checkpoint models/regional/south_america/best.pt \
  --split test
```

The report includes binary precision, recall, F1, confusion matrix, class-wise
metrics, original five-label counts, false positives by `hard_negative_type`,
and delta-time summaries when event timestamps are available.
