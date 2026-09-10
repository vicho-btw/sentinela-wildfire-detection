# Source

Future training, dataset, and model code goes here.

Planned modules:

- `datasets/`: GeoTIFF loaders and patch datasets.
- `models/`: input adapters, backbones, segmentation heads.
- `training/`: losses, metrics, trainer entrypoints.
- `verification/`: postprocessing into Condor incident verification results.

## Live GOES scan

Run a one-shot AOI scan against the latest public NOAA GOES-East ABI full-disk file:

```bash
python scripts/run_goes_live_scan.py \
  --checkpoint runs/goes_multitask_balanced_all_gpu_maskfocus_e40/best_combined.pt \
  --bbox -92,14,-86,22 \
  --source goes_east_abi \
  --threshold 0.7 \
  --mask-threshold 0.5 \
  --out runs/live_trials/condor_live_alerts.csv \
  --evidence-dir ../condor-api/evidence \
  --evidence-base-url http://127.0.0.1:8000/evidence \
  --post-url http://127.0.0.1:8000/alert \
  --once
```

For a polling service:

```bash
python scripts/run_goes_live_scan.py \
  --checkpoint runs/goes_multitask_balanced_all_gpu_maskfocus_e40/best_combined.pt \
  --bbox -92,14,-86,22 \
  --source goes_east_abi \
  --threshold 0.7 \
  --mask-threshold 0.5 \
  --out runs/live_trials/condor_live_alerts.csv \
  --evidence-dir ../condor-api/evidence \
  --evidence-base-url http://127.0.0.1:8000/evidence \
  --post-url http://127.0.0.1:8000/alert \
  --watch \
  --interval-minutes 10
```

Use `--dry-run-grid` to validate the GOES projection and AOI window count without model inference.
