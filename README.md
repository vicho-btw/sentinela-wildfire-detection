# Sentinela — Early Wildfire Detection from GOES Satellite Imagery

A multi-task model that reads GOES ABI satellite imagery and predicts both
*whether* a scene contains fire and *where* the fire is, from a short stack of
frames leading up to the present moment.

The goal is lead time. NASA FIRMS is the public operational baseline for
wildfire hotspots; the question this model asks is whether the fire signal is
already visible in GOES imagery earlier than FIRMS reports it. Detection is
framed temporally on purpose — the input is four frames at -30, -20, -10 and 0
minutes, so the model can see a signal developing rather than a single
snapshot.

Built as the modelling work for Condor, a wildfire detection startup. Published
here as the technical work; the product and business side are not part of this
repo.

## Results, stated carefully

Two different numbers appear in this repo and they are not comparable. Both are
included because the gap between them is the honest state of the project.

**The mature checkpoint** — `goes_multitask_balanced_all_gpu_maskfocus_e40`,
epoch 34, scored on a balanced GOES **validation** split:

| Metric | Value |
|---|---|
| Accuracy | 0.9983 |
| Precision | 0.9967 |
| Recall | 1.0000 |
| Mask IoU | 0.6765 |

Scene classification is close to saturated on that split; localisation
(IoU 0.68) is the part with real headroom left.

**The regional pipeline** — the `south_america` run in
`data/goes_fire/south_america/evaluation/best_test.json` — is early and its
test result is **not meaningful**: the test split holds two samples, the
confusion matrix is `[[0, 1], [0, 1]]`, and the model called both positive.
The reported 0.50 accuracy is sample-size noise, not a measurement. The
regional path is scaffolding that works end to end; it has not been trained at
a scale that would let it be scored.

**Lead time over FIRMS is unmeasured.** The benchmark in
`local_results/benchmarks/condor_vs_nasa_vs_sof.md` records it as `n/a`,
because measuring it requires replaying live GOES scans and scoring the
resulting alerts against FIRMS timestamps. The comparison against Satellites
on Fire in that table is *their published claim*, not something measured here,
and the file says so.

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env      # add a free NASA FIRMS map key
```

Training runs on a rented vast.ai GPU box; `vast/` holds the sync and setup
scripts and `vast/README_regional_goes.md` the commands. `kaggle/` is an
alternative runner for Kaggle's GPU kernels — copy
`kaggle/kernel-metadata.json.template` and fill in your own username.

Datasets are not committed: GOES ABI imagery is fetched by
`scripts/download_goes_region.py` and labels bootstrapped from FIRMS by
`scripts/download_firms_bootstrap_labels.py`.

## Architecture

The regional model keeps the original Sentinela-ModelS body/backbone in
`src/sentinela_models/model.py`. Regional runs change only:

- a temporal input adapter before the backbone
- the input channel count after flattening time into channels
- the output head for binary scene-level fire probability

Default raw GOES channels:

```text
CMI_C02, CMI_C03, CMI_C05, CMI_C06, CMI_C07,
CMI_C11, CMI_C13, CMI_C14, CMI_C15
```

Optional derived channels are configurable per run for ablations.

Default temporal setup:

```text
Input:  [B, 4, 9, 64, 64]
Steps:  [-30, -20, -10, 0] minutes
Model:  [B, 36, 64, 64] after temporal flattening
Output: scene_logits [B]
Mask:   mask_logits [B, 1, H, W] only when mask labels are available
```

## Labels

Training is binary:

```text
0 no_fire      = negative + hard_negative
1 fire_signal  = active_fire + early_fire_signal
```

Original regional labels are still stored for diagnostics:

```text
0 negative
1 active_fire
2 early_fire_signal
3 hard_negative
4 uncertain
```

`uncertain` samples default to zero training/evaluation weight. Use
`--drop-uncertain` during training to remove them from the dataset entirely.

Regional labels are ingested from generic CSV/JSONL event files. External hotspot
products may be used later only as optional reference/evaluation metadata, not
as the core training-label pipeline.

## Commands

Vast.ai setup and training commands are in:

```text
vast/README_regional_goes.md
```

Main scripts:

```text
scripts/download_firms_bootstrap_labels.py
scripts/build_regional_goes_samples_streaming.py
scripts/run_south_america_pipeline.sh
scripts/write_regional_benchmark_report.py
scripts/download_goes_region.py
scripts/ingest_regional_labels.py
scripts/build_regional_goes_samples.py
scripts/train_regional_goes.py
scripts/evaluate_regional_goes.py
```

