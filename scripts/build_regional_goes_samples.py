#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from sentinela_models.regional import DERIVED_CHANNELS, HARD_NEGATIVE_TYPES, LABEL_NAMES, get_region_config


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
    parser = argparse.ArgumentParser(description="Build regional GOES wildfire .npz samples from downloaded ABI files.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--labels", default=None, help="Defaults to data/goes_fire/<region>/processed/labels.csv")
    parser.add_argument("--download-manifest", default=None, help="Defaults to raw/download_manifest.csv")
    parser.add_argument("--sample-targets", default=None, help="Defaults to processed/sample_targets.csv when present.")
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--patch-size", type=int, default=0)
    parser.add_argument("--temporal-offsets", default=None, help="Comma-separated minutes, default from regional config: -30,-20,-10,0")
    parser.add_argument("--max-events", type=int, default=0)
    parser.add_argument("--negative-count", type=int, default=0)
    parser.add_argument("--hard-negative-count", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_utc(value: str) -> datetime:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_temporal_offsets(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(int(v) for v in default)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


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


def projection_transformer(ds: Any) -> tuple[Any, float]:
    try:
        from pyproj import CRS, Transformer
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing pyproj. Install requirements before building regional GOES samples.") from exc

    proj = ds["goes_imager_projection"]
    h = float(proj.attrs["perspective_point_height"])
    lon0 = float(proj.attrs["longitude_of_projection_origin"])
    sweep = str(proj.attrs.get("sweep_angle_axis", "x"))
    a = float(proj.attrs["semi_major_axis"])
    b = float(proj.attrs["semi_minor_axis"])
    crs = CRS.from_proj4(f"+proj=geos +h={h} +lon_0={lon0} +sweep={sweep} +a={a} +b={b} +units=m +no_defs")
    transformer = Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    return transformer, h


def latlon_to_rowcol(ds: xr.Dataset, lat: float, lon: float, variable: str) -> tuple[int, int]:
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


def load_patch(path: Path, lat: float, lon: float, channels: list[str], size: int) -> tuple[np.ndarray, int, int]:
    try:
        import xarray as xr
    except ModuleNotFoundError as exc:
        raise SystemExit("Missing xarray. Install requirements before building regional GOES samples.") from exc

    with xr.open_dataset(path, engine="netcdf4") as ds:
        row, col = latlon_to_rowcol(ds, lat, lon, channels[0])
        bands = []
        row0 = col0 = 0
        for channel in channels:
            if channel not in ds:
                raise KeyError(f"{channel} missing from {path}")
            arr = np.asarray(ds[channel].values, dtype=np.float32)
            crop, row0, col0 = crop_2d(arr, row, col, size)
            bands.append(crop)
    return np.stack(bands, axis=0), row0, col0


def add_derived(raw: np.ndarray, raw_channels: list[str], previous_raw: np.ndarray | None, derived_names: list[str]) -> np.ndarray:
    index = {name: idx for idx, name in enumerate(raw_channels)}
    derived: list[np.ndarray] = []
    zeros = np.zeros_like(raw[0])
    for name in derived_names:
        if name == "C07_minus_C13":
            derived.append(raw[index["CMI_C07"]] - raw[index["CMI_C13"]])
        elif name == "C07_minus_C14":
            derived.append(raw[index["CMI_C07"]] - raw[index["CMI_C14"]])
        elif name == "C14_minus_C15":
            derived.append(raw[index["CMI_C14"]] - raw[index["CMI_C15"]])
        elif name == "C13_temporal_delta_10m":
            derived.append(raw[index["CMI_C13"]] - previous_raw[index["CMI_C13"]] if previous_raw is not None else zeros)
        elif name == "C07_temporal_delta_10m":
            derived.append(raw[index["CMI_C07"]] - previous_raw[index["CMI_C07"]] if previous_raw is not None else zeros)
        else:
            raise ValueError(f"Unsupported derived channel: {name}")
    return np.concatenate([raw, np.stack(derived, axis=0)], axis=0) if derived else raw


def nearest_file(rows: list[dict[str, str]], target: datetime, max_minutes: float | None = None, before_only: bool = False) -> dict[str, str] | None:
    ranked: list[tuple[float, dict[str, str]]] = []
    for row in rows:
        if row.get("status") not in ("downloaded", "skipped", "dry_run"):
            continue
        path = Path(row.get("local_path", ""))
        if not path.exists() or path.stat().st_size <= 0:
            continue
        ts = parse_utc(row["timestamp_utc"])
        delta = (ts - target).total_seconds() / 60.0
        if before_only and delta >= 0:
            continue
        if max_minutes is not None and abs(delta) > float(max_minutes):
            continue
        ranked.append((abs(delta), row))
    return sorted(ranked, key=lambda item: item[0])[0][1] if ranked else None


def nearest_temporal_rows(
    rows: list[dict[str, str]],
    anchor: datetime,
    offsets_minutes: tuple[int, ...],
    max_minutes: float | None,
) -> list[dict[str, str]] | None:
    sequence = []
    for offset in offsets_minutes:
        row = nearest_file(rows, anchor + timedelta(minutes=int(offset)), max_minutes=max_minutes)
        if row is None:
            return None
        sequence.append(row)
    return sequence


def file_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row.get("bucket", ""), row.get("key", "")): row for row in rows}


def load_temporal_patch(
    files: list[dict[str, str]],
    sequence_rows: list[dict[str, str]],
    lat: float,
    lon: float,
    raw_channels: list[str],
    patch_size: int,
    derived_names: list[str],
) -> tuple[np.ndarray, int, int]:
    frames = []
    row0 = col0 = 0
    for frame_row in sequence_rows:
        raw, row0, col0 = load_patch(Path(frame_row["local_path"]), lat, lon, raw_channels, patch_size)
        previous_raw = None
        if derived_names:
            file_time = parse_utc(frame_row["timestamp_utc"])
            previous_row = nearest_file(files, file_time, max_minutes=20, before_only=True)
            if previous_row:
                previous_raw, _, _ = load_patch(Path(previous_row["local_path"]), lat, lon, raw_channels, patch_size)
        frames.append(add_derived(raw, raw_channels, previous_raw, derived_names))
    return np.stack(frames, axis=0), row0, col0


def write_sample(out_path: Path, payload: dict[str, Any]) -> bool:
    if out_path.exists() and out_path.stat().st_size > 0:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **payload)
    return True


def sample_id_for(prefix: str, parts: list[str]) -> str:
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


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
            lon = rng.uniform(max(west, zw), min(east, ze))
            lat = rng.uniform(max(south, zs), min(north, zn))
            return lat, lon
    return rng.uniform(south, north), rng.uniform(west, east)


def main() -> None:
    args = parse_args()
    rng = random.Random(int(args.seed))
    cfg = get_region_config(args.region, args.config)
    region_root = cfg.region_root(args.data_root)
    for folder in ["positive", "early_positive", "negative", "hard_negative", "uncertain"]:
        (region_root / "samples" / folder).mkdir(parents=True, exist_ok=True)
    for folder in ["fire_temperature_rgb", "geocolor", "sandwich_rgb"]:
        (region_root / "qa_previews" / folder).mkdir(parents=True, exist_ok=True)

    labels_path = Path(args.labels) if args.labels else region_root / "processed" / "labels.csv"
    download_manifest = Path(args.download_manifest) if args.download_manifest else region_root / "raw" / "download_manifest.csv"
    sample_targets_path = Path(args.sample_targets) if args.sample_targets else region_root / "processed" / "sample_targets.csv"
    labels = read_csv(labels_path)
    files = read_csv(download_manifest)
    if not files:
        raise SystemExit(f"No GOES files found in {download_manifest}")

    if args.max_events > 0:
        labels = labels[: int(args.max_events)]
    patch_size = int(args.patch_size or cfg.patch_size)
    raw_channels = list(cfg.raw_channels)
    derived_names = list(cfg.derived_channels if args.include_derived else [])
    temporal_offsets = parse_temporal_offsets(args.temporal_offsets, cfg.temporal_offsets_minutes)
    manifest_rows: list[dict[str, Any]] = []
    written = skipped = failed = 0
    sample_targets = read_csv(sample_targets_path)
    files_by_key = file_by_key(files)

    if sample_targets:
        for target in sample_targets:
            label = int(target["label"])
            file_row = files_by_key.get((target.get("bucket", ""), target.get("key", "")), target)
            if file_row.get("status") not in ("downloaded", "skipped"):
                skipped += 1
                continue
            path = Path(file_row["local_path"])
            if not path.exists() or path.stat().st_size <= 0:
                skipped += 1
                continue
            file_time = parse_utc(file_row["timestamp_utc"])
            event_time_text = target.get("event_time_utc", "")
            target_time = parse_utc(event_time_text or target["target_time_utc"])
            delta_minutes = (file_time - target_time).total_seconds() / 60.0
            lat = float(target["center_lat"])
            lon = float(target["center_lon"])
            sequence_rows = nearest_temporal_rows(files, file_time, temporal_offsets, max_minutes=cfg.temporal_tolerance_minutes)
            if sequence_rows is None:
                skipped += 1
                continue
            try:
                x, row0, col0 = load_temporal_patch(files, sequence_rows, lat, lon, raw_channels, patch_size, derived_names)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"failed target_id={target.get('target_id')} path={path} error={exc}", flush=True)
                continue
            temporal_timestamps = [row["timestamp_utc"] for row in sequence_rows]
            sample_id = sample_id_for("target", [cfg.name, target["target_id"], "|".join(temporal_timestamps), str(label)])
            out_path = region_root / "samples" / folder_for_label(label) / f"{sample_id}.npz"
            did_write = write_sample(
                out_path,
                {
                    "x": x.astype(np.float32, copy=False),
                    "class_label": np.asarray(label, dtype=np.int64),
                    "center_lat": np.asarray(lat, dtype=np.float32),
                    "center_lon": np.asarray(lon, dtype=np.float32),
                    "timestamp_utc": np.asarray(file_row["timestamp_utc"]),
                    "temporal_offsets_minutes": np.asarray(temporal_offsets, dtype=np.int16),
                    "temporal_timestamps_utc": np.asarray(temporal_timestamps),
                    "source_id": np.asarray(file_row["source_id"]),
                    "sector": np.asarray(file_row["sector"]),
                    "channel_names": np.asarray(raw_channels + derived_names),
                    "derived_channel_names": np.asarray(derived_names),
                },
            )
            written += int(did_write)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "region": cfg.name,
                    "source_id": file_row["source_id"],
                    "sector": file_row["sector"],
                    "timestamp_utc": file_row["timestamp_utc"],
                    "center_lat": f"{lat:.6f}",
                    "center_lon": f"{lon:.6f}",
                    "bbox_w": cfg.bbox[0],
                    "bbox_s": cfg.bbox[1],
                    "bbox_e": cfg.bbox[2],
                    "bbox_n": cfg.bbox[3],
                    "patch_row0": row0,
                    "patch_col0": col0,
                    "patch_size": patch_size,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "label_source": target.get("label_source", ""),
                    "event_id": target.get("event_id", ""),
                    "event_time_utc": event_time_text,
                    "delta_t_minutes": f"{delta_minutes:.2f}",
                    "hard_negative_type": target.get("hard_negative_type", ""),
                    "split": split_for_id(sample_id),
                    "path": str(out_path.relative_to(region_root)),
                }
            )

        out_manifest = region_root / "manifest.csv"
        with out_manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
            writer.writeheader()
            writer.writerows(manifest_rows)
        print(
            json.dumps(
                {
                    "region": cfg.name,
                    "sample_targets": str(sample_targets_path),
                    "samples": len(manifest_rows),
                    "written": written,
                    "skipped": skipped,
                    "failed": failed,
                    "manifest": str(out_manifest),
                },
                indent=2,
            )
        )
        return

    for label_row in labels:
        event_time = parse_utc(label_row["event_time_utc"])
        label = int(label_row["label"])
        before_only = label == 2
        max_minutes = cfg.early_signal_minutes if before_only else cfg.temporal_tolerance_minutes
        file_row = nearest_file(files, event_time, max_minutes=max_minutes, before_only=before_only)
        if file_row is None:
            skipped += 1
            continue
        file_time = parse_utc(file_row["timestamp_utc"])
        delta_minutes = (file_time - event_time).total_seconds() / 60.0
        path = Path(file_row["local_path"])
        sequence_rows = nearest_temporal_rows(files, file_time, temporal_offsets, max_minutes=cfg.temporal_tolerance_minutes)
        if sequence_rows is None:
            skipped += 1
            continue
        try:
            x, row0, col0 = load_temporal_patch(
                files,
                sequence_rows,
                float(label_row["lat"]),
                float(label_row["lon"]),
                raw_channels,
                patch_size,
                derived_names,
            )
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"failed event_id={label_row.get('event_id')} path={path} error={exc}", flush=True)
            continue

        temporal_timestamps = [row["timestamp_utc"] for row in sequence_rows]
        sample_id = sample_id_for("event", [cfg.name, label_row["event_id"], "|".join(temporal_timestamps), str(label)])
        folder = folder_for_label(label)
        out_path = region_root / "samples" / folder / f"{sample_id}.npz"
        did_write = write_sample(
            out_path,
            {
                "x": x.astype(np.float32, copy=False),
                "class_label": np.asarray(label, dtype=np.int64),
                "center_lat": np.asarray(float(label_row["lat"]), dtype=np.float32),
                "center_lon": np.asarray(float(label_row["lon"]), dtype=np.float32),
                "timestamp_utc": np.asarray(file_row["timestamp_utc"]),
                "temporal_offsets_minutes": np.asarray(temporal_offsets, dtype=np.int16),
                "temporal_timestamps_utc": np.asarray(temporal_timestamps),
                "source_id": np.asarray(file_row["source_id"]),
                "sector": np.asarray(file_row["sector"]),
                "channel_names": np.asarray(raw_channels + derived_names),
                "derived_channel_names": np.asarray(derived_names),
            },
        )
        written += int(did_write)
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "region": cfg.name,
                "source_id": file_row["source_id"],
                "sector": file_row["sector"],
                "timestamp_utc": file_row["timestamp_utc"],
                "center_lat": label_row["lat"],
                "center_lon": label_row["lon"],
                "bbox_w": cfg.bbox[0],
                "bbox_s": cfg.bbox[1],
                "bbox_e": cfg.bbox[2],
                "bbox_n": cfg.bbox[3],
                "patch_row0": row0,
                "patch_col0": col0,
                "patch_size": patch_size,
                "label": label,
                "label_name": LABEL_NAMES[label],
                "label_source": label_row["source"],
                "event_id": label_row["event_id"],
                "event_time_utc": label_row["event_time_utc"],
                "delta_t_minutes": f"{delta_minutes:.2f}",
                "hard_negative_type": "",
                "split": split_for_id(sample_id),
                "path": str(out_path.relative_to(region_root)),
            }
        )

    west, south, east, north = cfg.bbox
    background_jobs = [(0, int(args.negative_count)), (3, int(args.hard_negative_count))]
    file_choices = [row for row in files if Path(row.get("local_path", "")).exists()]
    for label, count in background_jobs:
        for idx in range(count):
            file_row = rng.choice(file_choices)
            hard_type = rng.choice(HARD_NEGATIVE_TYPES) if label == 3 else ""
            lat, lon = choose_background_center(cfg.name, cfg.bbox, hard_type, rng)
            lat = _clamp(lat, south, north)
            lon = _clamp(lon, west, east)
            sample_id = sample_id_for("bg", [cfg.name, str(label), file_row["timestamp_utc"], f"{lat:.5f}", f"{lon:.5f}", hard_type, str(idx)])
            out_path = region_root / "samples" / folder_for_label(label) / f"{sample_id}.npz"
            try:
                file_time = parse_utc(file_row["timestamp_utc"])
                sequence_rows = nearest_temporal_rows(files, file_time, temporal_offsets, max_minutes=cfg.temporal_tolerance_minutes)
                if sequence_rows is None:
                    skipped += 1
                    continue
                x, row0, col0 = load_temporal_patch(files, sequence_rows, lat, lon, raw_channels, patch_size, derived_names)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"failed background sample={sample_id} error={exc}", flush=True)
                continue
            temporal_timestamps = [row["timestamp_utc"] for row in sequence_rows]
            did_write = write_sample(
                out_path,
                {
                    "x": x.astype(np.float32, copy=False),
                    "class_label": np.asarray(label, dtype=np.int64),
                    "center_lat": np.asarray(lat, dtype=np.float32),
                    "center_lon": np.asarray(lon, dtype=np.float32),
                    "timestamp_utc": np.asarray(file_row["timestamp_utc"]),
                    "temporal_offsets_minutes": np.asarray(temporal_offsets, dtype=np.int16),
                    "temporal_timestamps_utc": np.asarray(temporal_timestamps),
                    "source_id": np.asarray(file_row["source_id"]),
                    "sector": np.asarray(file_row["sector"]),
                    "channel_names": np.asarray(raw_channels + derived_names),
                    "derived_channel_names": np.asarray(derived_names),
                },
            )
            written += int(did_write)
            manifest_rows.append(
                {
                    "sample_id": sample_id,
                    "region": cfg.name,
                    "source_id": file_row["source_id"],
                    "sector": file_row["sector"],
                    "timestamp_utc": file_row["timestamp_utc"],
                    "center_lat": f"{lat:.6f}",
                    "center_lon": f"{lon:.6f}",
                    "bbox_w": cfg.bbox[0],
                    "bbox_s": cfg.bbox[1],
                    "bbox_e": cfg.bbox[2],
                    "bbox_n": cfg.bbox[3],
                    "patch_row0": row0,
                    "patch_col0": col0,
                    "patch_size": patch_size,
                    "label": label,
                    "label_name": LABEL_NAMES[label],
                    "label_source": "regional_background_sampler",
                    "event_id": "",
                    "event_time_utc": "",
                    "delta_t_minutes": "",
                    "hard_negative_type": hard_type,
                    "split": split_for_id(sample_id),
                    "path": str(out_path.relative_to(region_root)),
                }
            )

    out_manifest = region_root / "manifest.csv"
    with out_manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(json.dumps({"region": cfg.name, "samples": len(manifest_rows), "written": written, "skipped": skipped, "failed": failed, "manifest": str(out_manifest)}, indent=2))


if __name__ == "__main__":
    main()
