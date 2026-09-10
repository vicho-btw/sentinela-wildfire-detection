from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import FIRE_SIGNAL_LABELS, NO_FIRE_LABELS, UNCERTAIN_LABELS


CLASS_LABEL_COLUMNS = ("class_label", "y_class", "label")


def _normalize_patch(x: np.ndarray) -> np.ndarray:
    if x.ndim == 4:
        return np.stack([_normalize_patch(frame) for frame in x], axis=0)
    x = x.astype(np.float32, copy=False)
    out = np.zeros_like(x, dtype=np.float32)
    for idx in range(x.shape[0]):
        band = x[idx]
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            continue
        lo, hi = np.percentile(valid, [1, 99])
        if hi <= lo:
            continue
        out[idx] = np.clip((band - lo) / (hi - lo), 0.0, 1.0)
    return np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)


def binary_label_for(raw_label: int) -> int:
    if raw_label in FIRE_SIGNAL_LABELS:
        return 1
    if raw_label in NO_FIRE_LABELS or raw_label in UNCERTAIN_LABELS:
        return 0
    raise ValueError(f"Unsupported regional label: {raw_label}")


class RegionalGoesFireDataset(Dataset[dict[str, Any]]):
    """Loads regional GOES wildfire `.npz` samples listed in manifest.csv."""

    def __init__(
        self,
        manifest: str | Path,
        split: str | None = None,
        normalize: bool = True,
        uncertain_weight: float = 0.0,
        drop_uncertain: bool = False,
        binary: bool = True,
        expected_temporal_steps: int | None = None,
    ):
        self.manifest = Path(manifest)
        self.root = self.manifest.parent
        self.normalize = bool(normalize)
        self.uncertain_weight = float(uncertain_weight)
        self.drop_uncertain = bool(drop_uncertain)
        self.binary = bool(binary)
        self.expected_temporal_steps = int(expected_temporal_steps) if expected_temporal_steps else None
        with self.manifest.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        if split:
            rows = [row for row in rows if str(row.get("split", "")).strip() == split]
        if self.drop_uncertain:
            rows = [row for row in rows if self._row_raw_label(row) not in UNCERTAIN_LABELS]
        self.rows = [row for row in rows if self._sample_path(row).exists()]
        if not self.rows:
            raise RuntimeError(f"No readable regional GOES samples found in {self.manifest}")

    def _row_raw_label(self, row: dict[str, str]) -> int:
        if "label" in row and str(row["label"]).strip():
            return int(row["label"])
        label_name = row.get("label_name", "")
        return {
            "negative": 0,
            "active_fire": 1,
            "early_fire_signal": 2,
            "hard_negative": 3,
            "uncertain": 4,
        }[label_name]

    def _sample_path(self, row: dict[str, str]) -> Path:
        path = row.get("path") or row.get("sample_path") or ""
        if path:
            p = Path(path)
            return p if p.is_absolute() else self.root / p
        label_name = row.get("label_name") or "unknown"
        sample_id = row["sample_id"]
        folder = {
            "active_fire": "positive",
            "early_fire_signal": "early_positive",
            "negative": "negative",
            "hard_negative": "hard_negative",
            "uncertain": "uncertain",
        }.get(label_name, label_name)
        return self.root / "samples" / folder / f"{sample_id}.npz"

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[int(idx)]
        path = self._sample_path(row)
        with np.load(path, allow_pickle=False) as npz:
            x = npz["x"].astype(np.float32, copy=False)
            if self.normalize:
                x = _normalize_patch(x)
            if x.ndim not in (3, 4):
                raise ValueError(f"Expected sample x with [C,H,W] or [T,C,H,W], got shape={tuple(x.shape)} in {path}")
            if self.expected_temporal_steps is not None:
                observed_steps = x.shape[0] if x.ndim == 4 else 1
                if observed_steps != self.expected_temporal_steps:
                    raise ValueError(
                        f"Expected {self.expected_temporal_steps} temporal steps, got {observed_steps} in {path}. "
                        "Use --temporal-offsets 0 for legacy single-frame samples."
                    )
            label = None
            for key in CLASS_LABEL_COLUMNS:
                if key in npz:
                    label = int(np.asarray(npz[key]).reshape(-1)[0])
                    break
            if label is None:
                label = int(row["label"])
            target_label = binary_label_for(label) if self.binary else label
            weight = self.uncertain_weight if label in UNCERTAIN_LABELS else 1.0
            sample: dict[str, Any] = {
                "image": torch.from_numpy(x),
                "class_label": torch.tensor(target_label, dtype=torch.long),
                "original_class_label": torch.tensor(label, dtype=torch.long),
                "sample_weight": torch.tensor(weight, dtype=torch.float32),
                "sample_id": row.get("sample_id", path.stem),
                "hard_negative_type": row.get("hard_negative_type", ""),
                "delta_t_minutes": row.get("delta_t_minutes", ""),
                "event_time_utc": row.get("event_time_utc", ""),
                "timestamp_utc": row.get("timestamp_utc", ""),
            }
            if "y_mask" in npz:
                mask = npz["y_mask"]
                if self.binary and mask.dtype.kind in {"i", "u"}:
                    mask = np.isin(mask, list(FIRE_SIGNAL_LABELS)).astype(np.float32, copy=False)
                sample["mask"] = torch.from_numpy(mask.astype(np.float32 if self.binary else np.int64 if mask.ndim == 2 else np.float32, copy=False))
            return sample


def regional_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([item["image"] for item in batch]).float()
    labels = torch.stack([item["class_label"] for item in batch]).long()
    original_labels = torch.stack([item["original_class_label"] for item in batch]).long()
    weights = torch.stack([item["sample_weight"] for item in batch]).float()
    out: dict[str, Any] = {
        "image": torch.nan_to_num(images, nan=0.0, posinf=1.0, neginf=0.0),
        "class_label": labels,
        "original_class_label": original_labels,
        "sample_weight": weights,
        "sample_id": [item["sample_id"] for item in batch],
        "hard_negative_type": [item.get("hard_negative_type", "") for item in batch],
        "delta_t_minutes": [item.get("delta_t_minutes", "") for item in batch],
    }
    if all("mask" in item for item in batch):
        out["mask"] = torch.stack([item["mask"] for item in batch])
    return out
