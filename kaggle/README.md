# Kaggle GPU Runner

Kaggle cannot expose its GPU directly to a Python process running on your laptop.
The practical workflow is:

1. Upload this repo as a private Kaggle Dataset.
2. Push `sentinela_kaggle_runner.py` as a Kaggle Script/Notebook with GPU enabled.
3. Kaggle runs the training remotely.
4. Download checkpoints and reports from the Kaggle output.

## One-Time Kaggle CLI Setup

Install and authenticate locally:

```bash
python3 -m pip install kaggle
mkdir -p ~/.kaggle
# Download kaggle.json from Kaggle Account > API > Create New Token,
# then place it at ~/.kaggle/kaggle.json.
chmod 600 ~/.kaggle/kaggle.json
```

## Upload Sentinela-ModelS as a Dataset

From this `Sentinela-ModelS` folder:

```bash
mkdir -p /tmp/sentinela-kaggle-dataset
rsync -a \
  --exclude data \
  --exclude local_results \
  --exclude runs \
  --exclude models \
  --exclude .venv \
  ./ /tmp/sentinela-kaggle-dataset/Sentinela-ModelS/

cat > /tmp/sentinela-kaggle-dataset/dataset-metadata.json <<'JSON'
{
  "title": "Sentinela ModelS",
  "id": "YOUR_KAGGLE_USERNAME/sentinela-models",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

kaggle datasets create -p /tmp/sentinela-kaggle-dataset --private
```

For updates after code changes:

```bash
kaggle datasets version -p /tmp/sentinela-kaggle-dataset -m "Update Sentinela-ModelS"
```

## Push the Kaggle GPU Script

Copy the metadata template and edit:

- `id`
- `dataset_sources`
- optionally title

```bash
cp kaggle/kernel-metadata.json.template kaggle/kernel-metadata.json
```

Then push:

```bash
kaggle kernels push -p kaggle
```

Kaggle will run `sentinela_kaggle_runner.py` on a GPU instance when GPU is
enabled for the kernel.

## FIRMS Key

Set a Kaggle Secret named `FIRMS_MAP_KEY`, or set `FIRMS_MAP_KEY` as an
environment variable in the notebook/session. The runner checks both.

## Runner Defaults

The runner defaults to a smoke run:

```text
REGION=south_america
START_DATE=2024-01-01
END_DATE=2026-05-11
MAX_FIRMS_ROWS=2000
TEMPORAL_OFFSETS=-30,-20,-10,0
SAMPLES_PER_CLASS=1000
MAX_GOES_FILES=8
VARIANT=n
EPOCHS=1
```

For a real run, set:

```text
MAX_FIRMS_ROWS=0
SAMPLES_PER_CLASS=10000
MAX_GOES_FILES=0
VARIANT=s or m
EPOCHS=20
```

Kaggle output paths:

```text
/kaggle/working/models/regional/<region>/best.pt
/kaggle/working/models/regional/<region>/history.json
/kaggle/working/goes_fire/<region>/evaluation/best_test.json
/kaggle/working/goes_fire/<region>/reports/regional_benchmark.md
```
