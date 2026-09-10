# Early Wildfire Detection Benchmark

Generated: 2026-05-02T05:58:12.294481Z

## Systems

| System | Evidence type | Lead over NASA FIRMS | Detection metric | Localization metric | Notes |
|---|---:|---:|---:|---:|---|
| NASA FIRMS | Public operational baseline | 0 min | Baseline | Hotspot point/footprint | Used as comparison clock, not as ground truth segmentation. |
| Satellites on Fire | Public claim | 35.0 min avg | Not independently measured here | Not independently measured here | Requires raw SoF alert timestamps for apples-to-apples scoring. |
| Condor | Measured model checkpoint | n/a min median | acc=0.9983, prec=0.9967, rec=1.0000 | mask_iou=0.6765 | early_hit_rate=n/a, matched_rate=n/a |

## Condor Checkpoint

- Path: `/workspace/condor/Sentinela-ModelS/runs/goes_multitask_balanced_all_gpu_maskfocus_e40/best_combined.pt`
- Epoch: `34`
- Source specs: `[{"in_channels": 6, "source_id": "goes_east_abi"}, {"in_channels": 6, "source_id": "goes_west_abi"}]`
- Metrics: `{"epoch": 34, "train_loss": 0.04437423473570526, "val_accuracy": 0.9983484723369116, "val_loss": 0.2772264985827549, "val_mask_bce": 0.012862826211743535, "val_mask_dice": 0.12815646958882704, "val_mask_iou": 0.6764928511354079, "val_precision": 0.9966869133075649, "val_recall": 1.0, "val_scene_loss": 0.008050731225636654}`

## Interpretation

- NASA FIRMS is the public baseline; its lead time is defined as 0 in this benchmark.
- Satellites on Fire is included as a public-claims reference, not an independently measured result.
- Condor model metrics are measured on our GOES balanced validation split.
- Condor early lead time requires live or replayed `condor_alerts.csv` scored against FIRMS with `benchmark_early_alerts.py`.

## Next Measurement

Generate Condor alerts from a replay/live GOES scan, then run:

```bash
python scripts/benchmark_early_alerts.py \
  --alerts-csv runs/live_trials/condor_alerts.csv \
  --firms-csv ../data/firms/mexico_viirs_snpp_2026-04-26.csv \
  --radius-km 0.375 \
  --target-lead-minutes 15.0 \
  --out-json runs/live_trials/early_alert_benchmark.json \
  --matches-csv runs/live_trials/early_alert_matches.csv
```

The winning system is the one with high precision/recall and the highest median positive lead time over FIRMS.
