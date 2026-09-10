#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from contextlib import ExitStack
from datetime import date, datetime, time, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sentinela_models.regional import HARD_NEGATIVE_TYPES, LABEL_NAMES, get_region_config


GOES_START_RE = re.compile(r"_s(?P<stamp>\d{14})")
MANIFEST_COLUMNS = [
    "sample_id",
    "region",
    "source_id",
    "sector",
    "timestamp_utc",
    "center_lat",
    "center_lon",
    "bbox_w",
    "bbox_s",
    "bbox_e",
    "bbox_n",
    "patch_row0",
    "patch_col0",
    "patch_size",
    "label",
    "label_name",
    "label_source",
    "event_id",
    "event_time_utc",
    "delta_t_minutes",
    "hard_negative_type",
    "split",
    "path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream GOES downloads into regional .npz samples, deleting raw NetCDF files after extraction."
    )
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labels", default=None, help="Defaults to data/goes_fire/<region>/processed/labels.csv")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--samples-per-class", type=int, default=5000)
    parser.add_argument("--positive-labels", default="1,2")
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5)
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--temporal-offsets", default=None, help="Comma-separated minutes, default from regional config: -30,-20,-10,0")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-goes-files", type=int, default=0, help="Smoke cap after grouping targets by GOES temporal sequence.")
    parser.add_argument("--max-targets-per-file", type=int, default=0, help="Optional cap to prevent one file dominating.")
    parser.add_argument("--keep-raw", action="store_true", help="Keep raw .nc files instead of deleting after extraction.")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def parse_temporal_offsets(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(int(v) for v in default)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def load_existing_sample_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["sample_id"] for row in csv.DictReader(handle)}


def split_for_id(sample_id: str) -> str:
    n = int(hashlib.sha1(sample_id.encode("utf-8")).hexdigest()[:8], 16) % 100
    if n < 80:
        return "train"
    if n < 90:
        return "val"
    return "test"


def folder_for_label(label: int) -> str:
    return {
        0: "negative",
        1: "positive",
        2: "early_positive",
        3: "hard_negative",
        4: "uncertain",
    }[int(label)]


def sample_id_for(parts: list[str]) -> str:
    return "stream_" + hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def target_id(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


def goes_stamp_to_utc(key: str) -> datetime | None:
    match = GOES_START_RE.search(key)
    if not match:
        return None
    stamp = match.group("stamp")
    return datetime(int(stamp[0:4]), 1, 1, tzinfo=timezone.utc) + timedelta(
        days=int(stamp[4:7]) - 1,
        hours=int(stamp[7:9]),
        minutes=int(stamp[9:11]),
        seconds=int(stamp[11:13]),
        milliseconds=int(stamp[13]) * 100,
    )


def hour_prefix(product: str, dt: datetime) -> str:
    return f"{product}/{dt.year}/{dt:%j}/{dt:%H}/"


def list_keys(s3: Any, bucket: str, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for item in page.get("Contents", []):
            key = item["Key"]
            if key.endswith(".nc"):
                keys.append(key)
    return sorted(keys)


def nearest_key_for_time(s3: Any, buckets: list[str], product: str, target: datetime) -> tuple[str, str, datetime] | None:
    ranked: list[tuple[float, str, str, datetime]] = []
    for bucket in buckets:
        for offset in (0, -1, 1):
            hour = target.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
            for key in list_keys(s3, bucket, hour_prefix(product, hour)):
                stamp = goes_stamp_to_utc(key)
                if stamp is None:
                    continue
                ranked.append((abs((stamp - target).total_seconds()), bucket, key, stamp))
        if ranked:
            break
    if not ranked:
        return None
    _, bucket, key, stamp = sorted(ranked, key=lambda item: item[0])[0]
    return bucket, key, stamp


def nearest_sequence_for_time(
    s3: Any,
    buckets: list[str],
    product: str,
    anchor: datetime,
    offsets_minutes: tuple[int, ...],
) -> tuple[tuple[str, str, datetime], ...] | None:
    sequence: list[tuple[str, str, datetime]] = []
    for offset in offsets_minutes:
        match = nearest_key_for_time(s3, buckets, product, anchor + timedelta(minutes=int(offset)))
        if match is None:
            return None
        sequence.append(match)
    return tuple(sequence)


def download_one(s3: Any, bucket: str, key: str, out_path: Path, dry_run: bool) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and out_path.stat().st_size > 0:
        return "skipped"
    if dry_run:
        return "dry_run"
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    if tmp_path.exists() and tmp_path.stat().st_size == 0:
        tmp_path.unlink()
    s3.download_file(bucket, key, str(tmp_path))
    tmp_path.replace(out_path)
    return "downloaded"


def choose_background_center(region: str, bbox: tuple[float, float, float, float], hard_type: str, rng: random.Random) -> tuple[float, float]:
    west, south, east, north = bbox
    if region == "south_america":
        zones = {
            "ocean_glint": (-50.0, -34.0, -35.0, 5.0),
            "coastline": (-81.0, -35.0, -72.0, 10.0),
            "desert_hot_surface": (-72.0, -28.0, -66.0, -16.0),
            "urban_industrial_heat": (-47.5, -24.5, -43.0, -20.0),
            "volcano_geothermal": (-73.0, -42.0, -66.0, -15.0),
            "dust": (-70.0, -35.0, -58.0, -20.0),
            "fog_low_cloud": (-80.0, -40.0, -70.0, -10.0),
            "agricultural_burn": (-64.0, -25.0, -50.0, -8.0),
            "deep_convective_cloud": (-75.0, -12.0, -48.0, 8.0),
            "thin_cirrus": (-75.0, -30.0, -45.0, 10.0),
            "cloud_edge": (-78.0, -20.0, -45.0, 8.0),
            "sensor_edge_artifact": (-82.0, -55.0, -76.0, 12.0),
        }
        zone = zones.get(hard_type)
        if zone:
            zw, zs, ze, zn = zone
            return rng.uniform(max(south, zs), min(north, zn)), rng.uniform(max(west, zw), min(east, ze))
    return rng.uniform(south, north), rng.uniform(west, east)


def random_time(start: date, end: date, rng: random.Random) -> datetime:
    start_dt = datetime.combine(start, time.min, tzinfo=timezone.utc)
    end_dt = datetime.combine(end + timedelta(days=1), time.min, tzinfo=timezone.utc)
    seconds = int((end_dt - start_dt).total_seconds())
    return start_dt + timedelta(seconds=rng.randrange(max(1, seconds)))


def build_targets(
    labels: list[dict[str, str]],
    start: date,
    end: date,
    positive_labels: set[int],
    samples_per_class: int,
    hard_negative_ratio: float,
    cfg: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    positives = []
    for row in labels:
        label = int(row["label"])
        if label not in positive_labels:
            continue
        event_time = parse_utc(row["event_time_utc"])
        if start <= event_time.date() <= end:
            positives.append(row)
    rng.shuffle(positives)
    positives = positives[: int(samples_per_class)]
    targets: list[dict[str, Any]] = []
    for row in positives:
        label = int(row["label"])
        event_time = parse_utc(row["event_time_utc"])
        targets.append(
            {
                "target_id": target_id([cfg.name, row["event_id"], row["event_time_utc"], str(label)]),
                "target_kind": "positive",
                "label": label,
                "label_name": LABEL_NAMES[label],
                "event_id": row["event_id"],
                "event_time_utc": row["event_time_utc"],
                "target_time_utc": format_utc(event_time),
                "center_lat": row["lat"],
                "center_lon": row["lon"],
                "label_source": row.get("source", "regional_labels"),
                "hard_negative_type": "",
            }
        )

    hard_count = int(round(len(positives) * max(0.0, min(1.0, hard_negative_ratio))))
    regular_count = len(positives) - hard_count
    for idx in range(regular_count):
        target_time = random_time(start, end, rng)
        lat, lon = choose_background_center(cfg.name, cfg.bbox, "", rng)
        targets.append(
            {
                "target_id": target_id([cfg.name, "negative", str(idx), format_utc(target_time), f"{lat:.5f}", f"{lon:.5f}"]),
                "target_kind": "negative",
                "label": 0,
                "label_name": LABEL_NAMES[0],
                "event_id": "",
                "event_time_utc": "",
                "target_time_utc": format_utc(target_time),
                "center_lat": f"{lat:.6f}",
                "center_lon": f"{lon:.6f}",
                "label_source": "regional_background_sampler",
                "hard_negative_type": "",
            }
        )
    for idx in range(hard_count):
        target_time = random_time(start, end, rng)
        hard_type = rng.choice(HARD_NEGATIVE_TYPES)
        lat, lon = choose_background_center(cfg.name, cfg.bbox, hard_type, rng)
        targets.append(
            {
                "target_id": target_id([cfg.name, "hard_negative", str(idx), hard_type, format_utc(target_time), f"{lat:.5f}", f"{lon:.5f}"]),
                "target_kind": "hard_negative",
                "label": 3,
                "label_name": LABEL_NAMES[3],
                "event_id": "",
                "event_time_utc": "",
                "target_time_utc": format_utc(target_time),
                "center_lat": f"{lat:.6f}",
                "center_lon": f"{lon:.6f}",
                "label_source": "regional_background_sampler",
                "hard_negative_type": hard_type,
            }
        )
    rng.shuffle(targets)
    return targets


def projection_transformer(ds: Any) -> tuple[Any, float]:
    from pyproj import CRS, Transformer

    proj = ds["goes_imager_projection"]
    h = float(proj.attrs["perspective_point_height"])
    lon0 = float(proj.attrs["longitude_of_projection_origin"])
    sweep = str(proj.attrs.get("sweep_angle_axis", "x"))
    a = float(proj.attrs["semi_major_axis"])
    b = float(proj.attrs["semi_minor_axis"])
    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +a={a} +b={b} +units=m +no_defs")
    return Transformer.from_crs("EPSG:4326", crs, always_xy=True), h


def latlon_to_rowcol(ds: Any, lat: float, lon: float, variable: str) -> tuple[int, int]:
    transformer, h = projection_transformer(ds)
    px, py = transformer.transform(float(lon), float(lat))
    da = ds[variable]
    y_name, x_name = da.dims[-2], da.dims[-1]
    xs = np.asarray(ds[x_name].values, dtype=np.float64)
    ys = np.asarray(ds[y_name].values, dtype=np.float64)
    col = int(np.abs(xs - (px / h)).argmin())
    row = int(np.abs(ys - (py / h)).argmin())
    return row, col


def crop_2d(arr: np.ndarray, center_row: int, center_col: int, size: int) -> tuple[np.ndarray, int, int]:
    half = int(size) // 2
    row0 = int(center_row) - half
    col0 = int(center_col) - half
    out = np.zeros((int(size), int(size)), dtype=np.float32)
    src_r0 = max(0, row0)
    src_c0 = max(0, col0)
    src_r1 = min(arr.shape[0], row0 + int(size))
    src_c1 = min(arr.shape[1], col0 + int(size))
    dst_r0 = src_r0 - row0
    dst_c0 = src_c0 - col0
    if src_r1 > src_r0 and src_c1 > src_c0:
        out[dst_r0 : dst_r0 + (src_r1 - src_r0), dst_c0 : dst_c0 + (src_c1 - src_c0)] = arr[src_r0:src_r1, src_c0:src_c1]
    return out, row0, col0


def load_patch_from_open_dataset(ds: Any, lat: float, lon: float, channels: list[str], size: int) -> tuple[np.ndarray, int, int]:
    row, col = latlon_to_rowcol(ds, lat, lon, channels[0])
    bands = []
    row0 = col0 = 0
    for channel in channels:
        if channel not in ds:
            raise KeyError(f"{channel} missing from dataset")
        arr = np.asarray(ds[channel].values, dtype=np.float32)
        crop, row0, col0 = crop_2d(arr, row, col, size)
        bands.append(crop)
    return np.stack(bands, axis=0), row0, col0


def write_sample(
    path: Path,
    x: np.ndarray,
    target: dict[str, Any],
    source_id: str,
    sector: str,
    channels: list[str],
    timestamp_utc: str,
    temporal_offsets: tuple[int, ...],
    temporal_timestamps: list[str],
) -> bool:
    if path.exists() and path.stat().st_size > 0:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        x=x.astype(np.float32, copy=False),
        class_label=np.asarray(int(target["label"]), dtype=np.int64),
        center_lat=np.asarray(float(target["center_lat"]), dtype=np.float32),
        center_lon=np.asarray(float(target["center_lon"]), dtype=np.float32),
        timestamp_utc=np.asarray(timestamp_utc),
        temporal_offsets_minutes=np.asarray(temporal_offsets, dtype=np.int16),
        temporal_timestamps_utc=np.asarray(temporal_timestamps),
        source_id=np.asarray(source_id),
        sector=np.asarray(sector),
        channel_names=np.asarray(channels),
        derived_channel_names=np.asarray([]),
    )
    return True


def main() -> None:
    args = parse_args()
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing dependencies. Install requirements.txt first.") from exc

    rng = random.Random(int(args.seed))
    cfg = get_region_config(args.region, args.config)
    cfg_start, cfg_end = cfg.date_range
    start = date.fromisoformat(args.start_date) if args.start_date else cfg_start
    end = date.fromisoformat(args.end_date) if args.end_date else cfg_end
    positive_labels = {int(value) for value in args.positive_labels.split(",") if value.strip()}
    region_root = cfg.region_root(args.data_root)
    labels_path = Path(args.labels) if args.labels else region_root / "processed" / "labels.csv"
    labels = read_csv(labels_path)
    if not labels:
        raise SystemExit(f"No labels found at {labels_path}.")
    raw_channels = list(cfg.raw_channels)
    patch_size = int(args.patch_size or cfg.patch_size)
    temporal_offsets = parse_temporal_offsets(args.temporal_offsets, cfg.temporal_offsets_minutes)
    manifest_path = region_root / "manifest.csv"
    existing_sample_ids = load_existing_sample_ids(manifest_path)

    for folder in ["positive", "early_positive", "negative", "hard_negative", "uncertain"]:
        (region_root / "samples" / folder).mkdir(parents=True, exist_ok=True)

    targets = build_targets(labels, start, end, positive_labels, int(args.samples_per_class), float(args.hard_negative_ratio), cfg, rng)
    if not targets:
        raise SystemExit("No targets selected.")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    buckets = [cfg.bucket, *cfg.fallback_buckets]
    grouped: dict[tuple[tuple[str, str, str], ...], list[dict[str, Any]]] = defaultdict(list)
    missing = 0
    for idx, target in enumerate(targets, start=1):
        sequence = nearest_sequence_for_time(s3, buckets, cfg.product, parse_utc(str(target["target_time_utc"])), temporal_offsets)
        if sequence is None:
            missing += 1
            continue
        target["temporal_sequence"] = [(bucket, key, format_utc(stamp)) for bucket, key, stamp in sequence]
        target["timestamp_utc"] = format_utc(sequence[-1][2])
        grouped[tuple(target["temporal_sequence"])].append(target)
        if idx % 1000 == 0:
            print(f"planned_targets={idx}/{len(targets)} grouped_files={len(grouped)} missing={missing}", flush=True)

    file_items = list(grouped.items())
    if args.max_goes_files > 0:
        file_items = file_items[: int(args.max_goes_files)]

    counts: defaultdict[str, int] = defaultdict(int)
    for file_idx, (sequence, file_targets) in enumerate(file_items, start=1):
        if args.max_targets_per_file > 0:
            file_targets = file_targets[: int(args.max_targets_per_file)]
        sequence_paths: list[Path] = []
        for bucket, key, _timestamp_utc in sequence:
            local_path = region_root / "raw" / "streaming" / bucket / key
            status = download_one(s3, bucket, key, local_path, bool(args.dry_run))
            counts[status] += 1
            sequence_paths.append(local_path)
        if args.dry_run:
            continue
        rows: list[dict[str, Any]] = []
        try:
            with ExitStack() as stack:
                datasets = [stack.enter_context(xr.open_dataset(path, engine="netcdf4")) for path in sequence_paths]
                for target in file_targets:
                    label = int(target["label"])
                    timestamp_utc = str(target["timestamp_utc"])
                    temporal_timestamps = [str(row[2]) for row in target["temporal_sequence"]]
                    sample_id = sample_id_for([cfg.name, target["target_id"], "|".join(temporal_timestamps), str(label)])
                    if sample_id in existing_sample_ids:
                        counts["sample_skipped"] += 1
                        continue
                    try:
                        frames = []
                        row0 = col0 = 0
                        for ds in datasets:
                            frame, row0, col0 = load_patch_from_open_dataset(
                                ds,
                                float(target["center_lat"]),
                                float(target["center_lon"]),
                                raw_channels,
                                patch_size,
                            )
                            frames.append(frame)
                        x = np.stack(frames, axis=0)
                    except Exception as exc:  # noqa: BLE001
                        counts["sample_failed"] += 1
                        print(f"failed sample target_id={target['target_id']} error={exc}", flush=True)
                        continue
                    out_path = region_root / "samples" / folder_for_label(label) / f"{sample_id}.npz"
                    write_sample(out_path, x, target, cfg.goes_source, cfg.sector, raw_channels, timestamp_utc, temporal_offsets, temporal_timestamps)
                    existing_sample_ids.add(sample_id)
                    target_time = parse_utc(target.get("event_time_utc") or target["target_time_utc"])
                    delta_minutes = (parse_utc(timestamp_utc) - target_time).total_seconds() / 60.0
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "region": cfg.name,
                            "source_id": cfg.goes_source,
                            "sector": cfg.sector,
                            "timestamp_utc": timestamp_utc,
                            "center_lat": target["center_lat"],
                            "center_lon": target["center_lon"],
                            "bbox_w": cfg.bbox[0],
                            "bbox_s": cfg.bbox[1],
                            "bbox_e": cfg.bbox[2],
                            "bbox_n": cfg.bbox[3],
                            "patch_row0": row0,
                            "patch_col0": col0,
                            "patch_size": patch_size,
                            "label": label,
                            "label_name": LABEL_NAMES[label],
                            "label_source": target["label_source"],
                            "event_id": target.get("event_id", ""),
                            "event_time_utc": target.get("event_time_utc", ""),
                            "delta_t_minutes": f"{delta_minutes:.2f}",
                            "hard_negative_type": target.get("hard_negative_type", ""),
                            "split": split_for_id(sample_id),
                            "path": str(out_path.relative_to(region_root)),
                        }
                    )
                    counts["samples_written"] += 1
        finally:
            if not args.keep_raw:
                for path in set(sequence_paths):
                    path.unlink(missing_ok=True)
        append_csv(manifest_path, rows, MANIFEST_COLUMNS)
        if file_idx % 10 == 0 or file_idx == len(file_items):
            print(
                f"processed_goes_files={file_idx}/{len(file_items)} samples_written={counts['samples_written']} "
                f"raw_status={{downloaded:{counts['downloaded']}, skipped:{counts['skipped']}}}",
                flush=True,
            )

    print(
        json.dumps(
            {
                "region": cfg.name,
                "date_range": [str(start), str(end)],
                "targets_selected": len(targets),
                "missing_targets": missing,
                "goes_files": len(file_items),
                "manifest": str(manifest_path),
                "keep_raw": bool(args.keep_raw),
                "counts": dict(counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
