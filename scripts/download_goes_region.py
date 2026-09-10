#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
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

from sentinela_models.regional import HARD_NEGATIVE_TYPES, LABEL_NAMES, get_region_config


GOES_START_RE = re.compile(r"_s(?P<stamp>\d{14})")
TARGET_COLUMNS = [
    "target_id",
    "region",
    "target_kind",
    "label",
    "label_name",
    "event_id",
    "event_time_utc",
    "target_time_utc",
    "center_lat",
    "center_lon",
    "label_source",
    "hard_negative_type",
    "source_id",
    "sector",
    "timestamp_utc",
    "bucket",
    "key",
    "remote_uri",
    "local_path",
    "status",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download only sampled GOES ABI files needed for balanced regional wildfire training."
    )
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labels", default=None, help="Defaults to data/goes_fire/<region>/processed/labels.csv")
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Defaults to config last_years, currently 2.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive.")
    parser.add_argument("--samples-per-class", type=int, default=0, help="If 0, use all positive labels in the date range.")
    parser.add_argument("--positive-labels", default="1,2", help="Comma-separated labels treated as wildfire positives.")
    parser.add_argument("--hard-negative-ratio", type=float, default=0.5, help="Fraction of non-fire samples labeled hard_negative.")
    parser.add_argument("--temporal-offsets", default=None, help="Comma-separated minutes, default from regional config: -30,-20,-10,0")
    parser.add_argument("--max-targets", type=int, default=0, help="Smoke-test cap across positive+negative targets.")
    parser.add_argument("--seed", type=int, default=7)
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


def format_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_temporal_offsets(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(int(v) for v in default)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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
    candidate_hours = [
        target.replace(minute=0, second=0, microsecond=0) + timedelta(hours=offset)
        for offset in (0, -1, 1)
    ]
    ranked: list[tuple[float, str, str, datetime]] = []
    for bucket in buckets:
        for hour in candidate_hours:
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


def local_path(region_root: Path, source_id: str, bucket: str, key: str) -> Path:
    return region_root / "raw" / source_id / bucket / key


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


def target_id(parts: list[str]) -> str:
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:16]


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
        if not (start <= event_time.date() <= end):
            continue
        positives.append(row)
    rng.shuffle(positives)
    if samples_per_class > 0:
        positives = positives[:samples_per_class]

    targets: list[dict[str, Any]] = []
    for row in positives:
        label = int(row["label"])
        event_time = parse_utc(row["event_time_utc"])
        targets.append(
            {
                "target_id": target_id([cfg.name, row["event_id"], row["event_time_utc"], str(label)]),
                "region": cfg.name,
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

    negative_count = len(positives) if samples_per_class <= 0 else samples_per_class
    hard_count = int(round(negative_count * max(0.0, min(1.0, hard_negative_ratio))))
    regular_count = negative_count - hard_count
    for idx in range(regular_count):
        target_time = random_time(start, end, rng)
        lat, lon = choose_background_center(cfg.name, cfg.bbox, "", rng)
        targets.append(
            {
                "target_id": target_id([cfg.name, "negative", str(idx), format_utc(target_time), f"{lat:.5f}", f"{lon:.5f}"]),
                "region": cfg.name,
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
                "region": cfg.name,
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


def main() -> None:
    args = parse_args()
    try:
        import boto3
        from botocore import UNSIGNED
        from botocore.config import Config
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing boto3/botocore. Install requirements before downloading GOES data.") from exc

    rng = random.Random(int(args.seed))
    cfg = get_region_config(args.region, args.config)
    cfg_start, cfg_end = cfg.date_range
    start = date.fromisoformat(args.start_date) if args.start_date else cfg_start
    end = date.fromisoformat(args.end_date) if args.end_date else cfg_end
    positive_labels = {int(value) for value in args.positive_labels.split(",") if value.strip()}
    temporal_offsets = parse_temporal_offsets(args.temporal_offsets, cfg.temporal_offsets_minutes)
    region_root = cfg.region_root(args.data_root)
    labels_path = Path(args.labels) if args.labels else region_root / "processed" / "labels.csv"
    labels = read_csv(labels_path)
    if not labels:
        raise SystemExit(f"No labels found at {labels_path}. Run ingest_regional_labels.py first.")

    targets = build_targets(
        labels=labels,
        start=start,
        end=end,
        positive_labels=positive_labels,
        samples_per_class=int(args.samples_per_class),
        hard_negative_ratio=float(args.hard_negative_ratio),
        cfg=cfg,
        rng=rng,
    )
    if args.max_targets > 0:
        targets = targets[: int(args.max_targets)]
    if not targets:
        raise SystemExit("No positive labels matched the requested date range; cannot build a 50/50 sample plan.")

    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    buckets = [cfg.bucket, *cfg.fallback_buckets]
    counts: defaultdict[str, int] = defaultdict(int)
    download_manifest_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []

    for idx, target in enumerate(targets, start=1):
        target_time = parse_utc(str(target["target_time_utc"]))
        sequence = nearest_sequence_for_time(s3, buckets, cfg.product, target_time, temporal_offsets)
        if sequence is None:
            counts["missing"] += 1
            continue
        sequence_rows = []
        target_failed = False
        for bucket, key, file_time in sequence:
            out_path = local_path(region_root, cfg.goes_source, bucket, key)
            try:
                status = download_one(s3, bucket, key, out_path, bool(args.dry_run))
            except Exception as exc:  # noqa: BLE001
                counts["failed"] += 1
                target_failed = True
                print(f"failed s3://{bucket}/{key} error={exc}", flush=True)
                break
            counts[status] += 1
            sequence_row = {
                "region": cfg.name,
                "source_id": cfg.goes_source,
                "sector": cfg.sector,
                "timestamp_utc": format_utc(file_time),
                "bucket": bucket,
                "key": key,
                "remote_uri": f"s3://{bucket}/{key}",
                "local_path": str(out_path),
                "status": status,
            }
            sequence_rows.append(sequence_row)
            download_manifest_rows.append(sequence_row)
        if target_failed:
            continue
        anchor_row = sequence_rows[-1]
        row = {
            **target,
            "source_id": cfg.goes_source,
            "sector": cfg.sector,
            "timestamp_utc": anchor_row["timestamp_utc"],
            "bucket": anchor_row["bucket"],
            "key": anchor_row["key"],
            "remote_uri": anchor_row["remote_uri"],
            "local_path": anchor_row["local_path"],
            "status": anchor_row["status"],
        }
        target_rows.append(row)
        if idx % 25 == 0 or idx == len(targets):
            print(f"processed_targets={idx}/{len(targets)} counts={dict(counts)}", flush=True)

    # Keep only one manifest row per downloaded GOES file, but preserve all sample targets.
    unique_downloads = {row["local_path"]: row for row in download_manifest_rows}
    download_manifest = region_root / "raw" / "download_manifest.csv"
    sample_targets = region_root / "processed" / "sample_targets.csv"
    write_csv(download_manifest, list(unique_downloads.values()), list(download_manifest_rows[0].keys()) if download_manifest_rows else TARGET_COLUMNS[-9:])
    write_csv(sample_targets, target_rows, TARGET_COLUMNS)
    positives = sum(1 for row in target_rows if int(row["label"]) in positive_labels)
    non_fire = len(target_rows) - positives
    print(
        json.dumps(
            {
                "region": cfg.name,
                "date_range": [str(start), str(end)],
                "sample_targets": str(sample_targets),
                "download_manifest": str(download_manifest),
                "temporal_offsets_minutes": list(temporal_offsets),
                "targets": len(target_rows),
                "positive_targets": positives,
                "non_fire_targets": non_fire,
                "unique_goes_files": len(unique_downloads),
                "counts": dict(counts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
