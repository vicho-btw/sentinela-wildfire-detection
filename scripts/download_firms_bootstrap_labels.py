#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import date, datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from time import sleep
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentinela_models.regional import get_region_config


DEFAULT_SOURCES = "VIIRS_SNPP_SP,VIIRS_NOAA20_SP,VIIRS_NOAA21_SP,MODIS_SP"
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download FIRMS hotspots as bootstrap regional event labels.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--map-key", default=None, help="Defaults to FIRMS_MAP_KEY env var.")
    parser.add_argument("--sources", default=DEFAULT_SOURCES)
    parser.add_argument("--start-date", default=None, help="YYYY-MM-DD. Defaults to regional config start date.")
    parser.add_argument("--end-date", default=None, help="YYYY-MM-DD inclusive. Defaults to regional config end date.")
    parser.add_argument("--chunk-days", type=int, default=5, help="FIRMS Area API supports 1..5 days per request.")
    parser.add_argument("--min-confidence", default="", help="Optional confidence filter: low, nominal, high, or numeric threshold.")
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)
    parser.add_argument("--out", default=None, help="Defaults to data/goes_fire/<region>/raw/labels/firms_bootstrap_events.csv")
    return parser.parse_args()


def normalize_confidence(value: str) -> str:
    text = str(value or "").strip().lower()
    return {"l": "low", "n": "nominal", "h": "high"}.get(text, text)


def confidence_ok(value: str, minimum: str) -> bool:
    if not minimum:
        return True
    conf = normalize_confidence(value)
    order = {"low": 0, "nominal": 1, "high": 2}
    if minimum.lower() in order:
        return order.get(conf, 0) >= order[minimum.lower()]
    try:
        return float(value) >= float(minimum)
    except ValueError:
        return True


def acq_datetime_utc(row: dict[str, str]) -> str:
    acq_date = row["acq_date"]
    acq_time = str(row["acq_time"]).strip().zfill(4)
    dt = datetime.strptime(f"{acq_date} {acq_time}", "%Y-%m-%d %H%M").replace(tzinfo=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def event_id_for(source: str, row: dict[str, str]) -> str:
    payload = "|".join(
        [
            source,
            row.get("latitude", ""),
            row.get("longitude", ""),
            row.get("acq_date", ""),
            str(row.get("acq_time", "")).zfill(4),
            row.get("satellite", ""),
            row.get("instrument", ""),
        ]
    )
    return "firms-" + hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def fetch_csv(map_key: str, source: str, bbox: tuple[float, float, float, float], start: date, day_range: int) -> list[dict[str, str]]:
    bbox_text = ",".join(f"{value:g}" for value in bbox)
    url = "/".join(
        [
            FIRMS_BASE_URL,
            quote(map_key),
            quote(source),
            quote(bbox_text, safe=","),
            str(day_range),
            start.isoformat(),
        ]
    )
    with urlopen(url, timeout=120) as response:
        text = response.read().decode("utf-8")
    if not text.strip() or text.lstrip().startswith("Invalid"):
        return []
    return list(csv.DictReader(io.StringIO(text)))


def iter_chunks(start: date, end: date, chunk_days: int) -> list[tuple[date, int]]:
    chunks: list[tuple[date, int]] = []
    current = start
    chunk = max(1, min(5, int(chunk_days)))
    while current <= end:
        days = min(chunk, (end - current).days + 1)
        chunks.append((current, days))
        current += timedelta(days=days)
    return chunks


def main() -> None:
    args = parse_args()
    import os

    map_key = args.map_key or os.getenv("FIRMS_MAP_KEY")
    if not map_key:
        raise SystemExit("Set FIRMS_MAP_KEY or pass --map-key. Get a free key from NASA FIRMS.")

    cfg = get_region_config(args.region, args.config)
    cfg_start, cfg_end = cfg.date_range
    start = date.fromisoformat(args.start_date) if args.start_date else cfg_start
    end = date.fromisoformat(args.end_date) if args.end_date else cfg_end
    region_root = cfg.region_root(args.data_root)
    out_path = Path(args.out) if args.out else region_root / "raw" / "labels" / "firms_bootstrap_events.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sources = [source.strip() for source in args.sources.split(",") if source.strip()]

    labels: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    chunks = iter_chunks(start, end, int(args.chunk_days))
    for source in sources:
        for idx, (chunk_start, days) in enumerate(chunks, start=1):
            try:
                rows = fetch_csv(map_key, source, cfg.bbox, chunk_start, days)
            except (HTTPError, URLError, TimeoutError) as exc:
                failures.append(f"{source} {chunk_start} days={days}: {exc}")
                continue
            for row in rows:
                if "latitude" not in row or "longitude" not in row or "acq_date" not in row or "acq_time" not in row:
                    continue
                if not confidence_ok(row.get("confidence", ""), str(args.min_confidence)):
                    continue
                event_id = event_id_for(source, row)
                labels[event_id] = {
                    "event_id": event_id,
                    "event_time_utc": acq_datetime_utc(row),
                    "lat": f"{float(row['latitude']):.6f}",
                    "lon": f"{float(row['longitude']):.6f}",
                    "label": "1",
                    "label_name": "active_fire",
                    "source": f"firms_bootstrap:{source}",
                    "confidence": normalize_confidence(row.get("confidence", "")),
                }
                if args.max_rows > 0 and len(labels) >= int(args.max_rows):
                    break
            if args.max_rows > 0 and len(labels) >= int(args.max_rows):
                break
            if idx % 20 == 0 or idx == len(chunks):
                print(f"source={source} chunks={idx}/{len(chunks)} labels={len(labels)}", flush=True)
            sleep(float(args.sleep_seconds))
        if args.max_rows > 0 and len(labels) >= int(args.max_rows):
            break

    rows_out = sorted(labels.values(), key=lambda row: (row["event_time_utc"], row["event_id"]))
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["event_id", "event_time_utc", "lat", "lon", "label", "label_name", "source", "confidence"])
        writer.writeheader()
        writer.writerows(rows_out)
    print(
        json.dumps(
            {
                "region": cfg.name,
                "date_range": [str(start), str(end)],
                "sources": sources,
                "labels": len(rows_out),
                "out": str(out_path),
                "failures": failures[:10],
                "failure_count": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
