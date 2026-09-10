from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import yaml


RAW_GOES_CHANNELS = [
    "CMI_C02",
    "CMI_C03",
    "CMI_C05",
    "CMI_C06",
    "CMI_C07",
    "CMI_C11",
    "CMI_C13",
    "CMI_C14",
    "CMI_C15",
]

DERIVED_CHANNELS = [
    "C07_minus_C13",
    "C07_minus_C14",
    "C14_minus_C15",
    "C13_temporal_delta_10m",
    "C07_temporal_delta_10m",
]

LABEL_NAMES = {
    0: "negative",
    1: "active_fire",
    2: "early_fire_signal",
    3: "hard_negative",
    4: "uncertain",
}

BINARY_LABEL_NAMES = {
    0: "no_fire",
    1: "fire_signal",
}

NO_FIRE_LABELS = frozenset({0, 3})
FIRE_SIGNAL_LABELS = frozenset({1, 2})
UNCERTAIN_LABELS = frozenset({4})
TEMPORAL_OFFSETS_MINUTES = (-30, -20, -10, 0)

HARD_NEGATIVE_TYPES = [
    "ocean_glint",
    "coastline",
    "cloud_edge",
    "deep_convective_cloud",
    "thin_cirrus",
    "desert_hot_surface",
    "urban_industrial_heat",
    "volcano_geothermal",
    "dust",
    "fog_low_cloud",
    "sensor_edge_artifact",
    "agricultural_burn",
    "unknown_hotspot",
]


@dataclass(frozen=True)
class RegionalConfig:
    name: str
    bbox: tuple[float, float, float, float]
    goes_source: str
    bucket: str
    fallback_buckets: tuple[str, ...]
    product: str
    sector: str
    raw_channels: tuple[str, ...] = tuple(RAW_GOES_CHANNELS)
    derived_channels: tuple[str, ...] = tuple(DERIVED_CHANNELS)
    derived_enabled_default: bool = False
    start_date: str | None = None
    end_date: str | None = None
    last_years: int = 5
    patch_size: int = 64
    temporal_offsets_minutes: tuple[int, ...] = TEMPORAL_OFFSETS_MINUTES
    temporal_tolerance_minutes: int = 15
    early_signal_minutes: int = 60
    data_root: str = "data/goes_fire"
    sector_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def date_range(self) -> tuple[date, date]:
        today = date.today()
        end = date.fromisoformat(self.end_date) if self.end_date else today
        start = date.fromisoformat(self.start_date) if self.start_date else end - timedelta(days=365 * int(self.last_years))
        return start, end

    def channels(self, include_derived: bool = False) -> list[str]:
        channels = list(self.raw_channels)
        if include_derived:
            channels.extend(self.derived_channels)
        return channels

    def region_root(self, data_root: str | Path | None = None) -> Path:
        return Path(data_root or self.data_root) / self.name


def _as_tuple(value: Any) -> tuple[Any, ...]:
    if value is None:
        return ()
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    return (value,)


def _region_from_mapping(name: str, row: dict[str, Any], defaults: dict[str, Any]) -> RegionalConfig:
    source = row.get("goes", {})
    return RegionalConfig(
        name=name,
        bbox=tuple(float(v) for v in row["bbox"]),
        goes_source=str(source.get("source_id", defaults["goes_source"])),
        bucket=str(source.get("bucket", defaults["bucket"])),
        fallback_buckets=tuple(str(v) for v in _as_tuple(source.get("fallback_buckets", defaults.get("fallback_buckets", [])))),
        product=str(source.get("product", defaults["product"])),
        sector=str(source.get("sector", defaults["sector"])),
        raw_channels=tuple(str(v) for v in row.get("raw_channels", defaults["raw_channels"])),
        derived_channels=tuple(str(v) for v in row.get("derived_channels", defaults["derived_channels"])),
        derived_enabled_default=bool(row.get("derived_enabled_default", defaults.get("derived_enabled_default", False))),
        start_date=row.get("start_date", defaults.get("start_date")),
        end_date=row.get("end_date", defaults.get("end_date")),
        last_years=int(row.get("last_years", defaults.get("last_years", 5))),
        patch_size=int(row.get("patch_size", defaults.get("patch_size", 64))),
        temporal_offsets_minutes=tuple(
            int(v) for v in _as_tuple(row.get("temporal_offsets_minutes", defaults.get("temporal_offsets_minutes", TEMPORAL_OFFSETS_MINUTES)))
        ),
        temporal_tolerance_minutes=int(row.get("temporal_tolerance_minutes", defaults.get("temporal_tolerance_minutes", 15))),
        early_signal_minutes=int(row.get("early_signal_minutes", defaults.get("early_signal_minutes", 60))),
        data_root=str(row.get("data_root", defaults.get("data_root", "data/goes_fire"))),
        sector_metadata=dict(source),
    )


def load_region_config(path: str | Path, region: str) -> RegionalConfig:
    with Path(path).open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    defaults = payload.get("defaults", {})
    defaults.setdefault("goes_source", "goes_east_abi")
    defaults.setdefault("bucket", "noaa-goes19")
    defaults.setdefault("fallback_buckets", ["noaa-goes16"])
    defaults.setdefault("product", "ABI-L2-MCMIPF")
    defaults.setdefault("sector", "full_disk")
    defaults.setdefault("raw_channels", RAW_GOES_CHANNELS)
    defaults.setdefault("derived_channels", DERIVED_CHANNELS)
    defaults.setdefault("temporal_offsets_minutes", TEMPORAL_OFFSETS_MINUTES)
    regions = payload.get("regions", {})
    if region not in regions:
        raise KeyError(f"Region '{region}' not found in {path}")
    return _region_from_mapping(region, regions[region], defaults)


def get_region_config(region: str, config_path: str | Path | None = None) -> RegionalConfig:
    path = Path(config_path) if config_path else Path(__file__).resolve().parents[3] / "configs" / "regional_goes.yaml"
    return load_region_config(path, region)
