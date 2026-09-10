# Configs

Experiment configuration files will live here.

Planned config groups:

- dataset paths
- selected satellite bands
- patch size
- normalization stats
- model backbone
- output heads
- loss weights
- train/validation/test split rules

## Regional GOES

`regional_goes.yaml` defines the GOES regional training plan. The first active
region is `south_america`; other regions are scaffolded for the upward training
order. The config keeps channel selection, derived-channel flags, bbox, source
bucket, sector metadata, patch size, temporal offsets, and last-five-years date
behavior in one place so raw-channel and raw-plus-derived ablations can use the
same scripts.

The default regional contract is binary temporal GOES:

```text
raw channels: 9
temporal offsets: [-30, -20, -10, 0] minutes
sample x: [4, 9, 64, 64]
model input after adapter: [36, 64, 64]
scene labels: 0 no_fire, 1 fire_signal
```
