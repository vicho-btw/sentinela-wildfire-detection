#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinela_models.regional import LABEL_NAMES, get_region_config


REQUIRED = {"event_id", "event_time_utc", "lat", "lon", "label", "label_name", "source"}
NAME_TO_LABEL = {name: label for label, name in LABEL_NAMES.items()}
ALIASES = {
    "positive": "active_fire",
    "early_positive": "early_fire_signal",
    "fire": "active_fire",
    "non_fire": "negative",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize generic regional wildfire event labels.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--labels", required=True, help="Input CSV or JSONL labels/events file.")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--out", default=None, help="Defaults to data/goes_fire/<region>/processed/labels.csv")
    return parser.parse_args()


def parse_utc(value: str) -> str:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    missing = REQUIRED - set(row)
    if missing:
        raise ValueError(f"Missing required label fields: {sorted(missing)}")
    raw_name = str(row.get("label_name") or "").strip()
    name = ALIASES.get(raw_name, raw_name)
    label = int(row["label"]) if str(row.get("label", "")).strip() != "" else NAME_TO_LABEL[name]
    if label not in LABEL_NAMES:
        raise ValueError(f"Unsupported label id {label} for event_id={row.get('event_id')}")
    canonical_name = LABEL_NAMES[label]
    if name and name != canonical_name:
        raise ValueError(f"label={label} expects label_name={canonical_name}, got {raw_name}")
    out = {
        "event_id": str(row["event_id"]).strip(),
        "event_time_utc": parse_utc(str(row["event_time_utc"])),
        "lat": f"{float(row['lat']):.6f}",
        "lon": f"{float(row['lon']):.6f}",
        "label": str(label),
        "label_name": canonical_name,
        "source": str(row["source"]).strip(),
        "confidence": str(row.get("confidence", "")).strip(),
        "notes": str(row.get("notes", "")).strip(),
    }
    return out


def main() -> None:
    args = parse_args()
    cfg = get_region_config(args.region, args.config)
    region_root = cfg.region_root(args.data_root)
    raw_label_dir = region_root / "raw" / "labels"
    processed_dir = region_root / "processed"
    raw_label_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    labels_path = Path(args.labels)
    copied_raw = raw_label_dir / labels_path.name
    if labels_path.resolve() != copied_raw.resolve():
        shutil.copy2(labels_path, copied_raw)

    rows = [normalize_row(row) for row in read_rows(labels_path)]
    rows.sort(key=lambda row: (row["event_time_utc"], row["event_id"]))
    out_path = Path(args.out) if args.out else processed_dir / "labels.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["event_id", "event_time_utc", "lat", "lon", "label", "label_name", "source", "confidence", "notes"]
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps({"region": cfg.name, "rows": len(rows), "raw_copy": str(copied_raw), "out": str(out_path)}, indent=2))


if __name__ == "__main__":
    main()
