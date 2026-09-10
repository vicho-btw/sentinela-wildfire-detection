#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
from torch.utils.data import DataLoader

from sentinela_models import SentinelaConfig, SentinelaModel
from sentinela_models.regional import BINARY_LABEL_NAMES, LABEL_NAMES, RegionalGoesFireDataset, get_region_config, regional_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a regional GOES wildfire checkpoint.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--include-uncertain", action="store_true")
    parser.add_argument("--out-json", default=None)
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    cfg = get_region_config(args.region, args.config)
    region_root = cfg.region_root(args.data_root)
    manifest = Path(args.manifest) if args.manifest else region_root / "manifest.csv"
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else ROOT / "models" / "regional" / cfg.name / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_cfg = SentinelaConfig(**checkpoint["model_config"])
    model = SentinelaModel(model_cfg)
    model.load_state_dict(checkpoint["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    temporal_offsets = tuple(checkpoint.get("temporal_offsets_minutes", (0,)))
    dataset = RegionalGoesFireDataset(
        manifest,
        split=str(args.split),
        uncertain_weight=1.0 if args.include_uncertain else 0.0,
        expected_temporal_steps=len(temporal_offsets),
    )
    loader = DataLoader(dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=regional_collate)

    confusion = [[0 for _ in BINARY_LABEL_NAMES] for _ in BINARY_LABEL_NAMES]
    original_class_counts: Counter[str] = Counter()
    false_positive_hard_negative: Counter[str] = Counter()
    lead_times_by_class: defaultdict[str, list[float]] = defaultdict(list)
    fire_tp = fire_fp = fire_fn = 0
    correct = total = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["image"].to(device)
            labels = batch["class_label"].to(device)
            weights = batch["sample_weight"].to(device)
            logits = model(x)["scene_logits"].reshape(-1)
            pred = (torch.sigmoid(logits) >= float(args.threshold)).long().cpu()
            labels_cpu = labels.cpu()
            weights_cpu = weights.cpu()
            original_cpu = batch["original_class_label"].cpu()
            for idx, (target, predicted, weight, original) in enumerate(
                zip(labels_cpu.tolist(), pred.tolist(), weights_cpu.tolist(), original_cpu.tolist(), strict=True)
            ):
                original_class_counts[LABEL_NAMES[int(original)]] += 1
                if weight <= 0:
                    continue
                confusion[int(target)][int(predicted)] += 1
                correct += int(predicted == target)
                total += 1
                pred_fire = predicted == 1
                target_fire = target == 1
                fire_tp += int(pred_fire and target_fire)
                fire_fp += int(pred_fire and not target_fire)
                fire_fn += int((not pred_fire) and target_fire)
                if original == 3 and pred_fire:
                    false_positive_hard_negative[batch["hard_negative_type"][idx] or "unknown"] += 1
                delta = as_float(batch["delta_t_minutes"][idx])
                if delta is not None:
                    lead_times_by_class[LABEL_NAMES[int(original)]].append(delta)

    class_wise: dict[str, dict[str, float]] = {}
    for class_id, name in BINARY_LABEL_NAMES.items():
        tp = confusion[class_id][class_id]
        fp = sum(confusion[row][class_id] for row in BINARY_LABEL_NAMES if row != class_id)
        fn = sum(confusion[class_id][col] for col in BINARY_LABEL_NAMES if col != class_id)
        class_wise[name] = {
            "precision": tp / max(1, tp + fp),
            "recall": tp / max(1, tp + fn),
            "support": sum(confusion[class_id]),
        }

    result = {
        "region": cfg.name,
        "checkpoint": str(checkpoint_path),
        "split": str(args.split),
        "threshold": float(args.threshold),
        "temporal_offsets_minutes": list(temporal_offsets),
        "accuracy": correct / max(1, total),
        "precision": fire_tp / max(1, fire_tp + fire_fp),
        "recall": fire_tp / max(1, fire_tp + fire_fn),
        "f1": (2 * fire_tp) / max(1, 2 * fire_tp + fire_fp + fire_fn),
        "confusion_matrix": confusion,
        "labels": BINARY_LABEL_NAMES,
        "original_labels": LABEL_NAMES,
        "original_class_counts": dict(original_class_counts),
        "class_wise": class_wise,
        "false_positives_by_hard_negative_type": dict(false_positive_hard_negative),
        "delta_t_minutes": {
            name: {"count": len(values), "mean": sum(values) / max(1, len(values))}
            for name, values in lead_times_by_class.items()
        },
    }
    out_json = Path(args.out_json) if args.out_json else region_root / "evaluation" / f"{checkpoint_path.stem}_{args.split}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
