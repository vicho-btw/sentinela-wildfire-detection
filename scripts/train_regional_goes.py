#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
try:
    from tqdm import tqdm
except ModuleNotFoundError:
    def tqdm(iterable: Any, **_: Any) -> Any:
        return iterable

from sentinela_models import SentinelaConfig, SentinelaModel
from sentinela_models.regional import RegionalGoesFireDataset, get_region_config, regional_collate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a regional GOES wildfire model using the Sentinela-ModelS v1 body.")
    parser.add_argument("--region", default="south_america")
    parser.add_argument("--config", default=str(ROOT / "configs" / "regional_goes.yaml"))
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--include-derived", action="store_true")
    parser.add_argument("--variant", default="s")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--uncertain-weight", type=float, default=0.0)
    parser.add_argument("--drop-uncertain", action="store_true")
    parser.add_argument("--temporal-offsets", default=None, help="Comma-separated minutes, default from regional config: -30,-20,-10,0")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mask-loss-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def parse_temporal_offsets(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or not str(value).strip():
        return tuple(int(v) for v in default)
    return tuple(int(v.strip()) for v in str(value).split(",") if v.strip())


def weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    logits = logits.reshape(-1)
    targets = labels.to(dtype=logits.dtype).reshape(-1)
    weights = weights.to(dtype=logits.dtype).reshape(-1)
    per_sample = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    denom = weights.sum().clamp_min(1.0)
    return (per_sample * weights).sum() / denom


def optional_mask_loss(mask_logits: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor | None:
    if "mask" not in batch:
        return None
    mask = batch["mask"].to(mask_logits.device)
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    if mask.ndim == 4 and mask.shape[1] != 1:
        mask = mask[:, :1]
    return F.binary_cross_entropy_with_logits(mask_logits, mask.float())


@torch.no_grad()
def evaluate(model: SentinelaModel, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0
    fire_tp = fire_fp = fire_fn = 0
    for batch in loader:
        x = batch["image"].to(device)
        labels = batch["class_label"].to(device)
        weights = batch["sample_weight"].to(device)
        out = model(x)
        logits = out["scene_logits"]
        loss = weighted_bce(logits, labels, weights)
        pred = torch.sigmoid(logits.reshape(-1)) >= float(args.threshold)
        valid = weights > 0
        correct += int((pred[valid] == labels[valid]).sum().item())
        total += int(valid.sum().item())
        pred_fire = pred
        target_fire = labels.bool()
        fire_tp += int((pred_fire & target_fire & valid).sum().item())
        fire_fp += int((pred_fire & ~target_fire & valid).sum().item())
        fire_fn += int((~pred_fire & target_fire & valid).sum().item())
        total_loss += float(loss.item()) * int(x.shape[0])
    return {
        "loss": total_loss / max(1, len(loader.dataset)),
        "accuracy": correct / max(1, total),
        "precision": fire_tp / max(1, fire_tp + fire_fp),
        "recall": fire_tp / max(1, fire_tp + fire_fn),
        "f1": (2 * fire_tp) / max(1, 2 * fire_tp + fire_fp + fire_fn),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(int(args.seed))
    cfg = get_region_config(args.region, args.config)
    region_root = cfg.region_root(args.data_root)
    manifest = Path(args.manifest) if args.manifest else region_root / "manifest.csv"
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "models" / "regional" / cfg.name
    out_dir.mkdir(parents=True, exist_ok=True)

    temporal_offsets = parse_temporal_offsets(args.temporal_offsets, cfg.temporal_offsets_minutes)
    temporal_steps = len(temporal_offsets)
    train_ds = RegionalGoesFireDataset(
        manifest,
        split="train",
        uncertain_weight=float(args.uncertain_weight),
        drop_uncertain=bool(args.drop_uncertain),
        expected_temporal_steps=temporal_steps,
    )
    val_ds = RegionalGoesFireDataset(
        manifest,
        split="val",
        uncertain_weight=float(args.uncertain_weight),
        drop_uncertain=bool(args.drop_uncertain),
        expected_temporal_steps=temporal_steps,
    )
    train_loader = DataLoader(train_ds, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.num_workers), collate_fn=regional_collate)
    val_loader = DataLoader(val_ds, batch_size=int(args.batch_size), shuffle=False, num_workers=int(args.num_workers), collate_fn=regional_collate)

    channels = cfg.channels(include_derived=bool(args.include_derived))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = SentinelaConfig(
        in_channels=len(channels) * temporal_steps,
        mask_classes=1,
        scene_classes=1,
        temporal_steps=temporal_steps,
        input_channels_per_timestep=len(channels),
        variant=str(args.variant),
        input_size=int(cfg.patch_size),
        include_scene_head=True,
    )
    model = SentinelaModel(model_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    history: list[dict[str, Any]] = []
    best_score = -1.0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        total_loss = 0.0
        total = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            x = batch["image"].to(device)
            labels = batch["class_label"].to(device)
            weights = batch["sample_weight"].to(device)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            scene_loss = weighted_bce(out["scene_logits"], labels, weights)
            mask_loss = optional_mask_loss(out["mask_logits"], batch)
            loss = scene_loss
            if mask_loss is not None:
                loss = loss + float(args.mask_loss_weight) * mask_loss
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(x.shape[0])
            total += int(x.shape[0])

        val = evaluate(model, val_loader, device, args)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "val_loss": val["loss"],
            "val_accuracy": val["accuracy"],
            "val_precision": val["precision"],
            "val_recall": val["recall"],
            "val_f1": val["f1"],
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        checkpoint = {
            "model": model.state_dict(),
            "model_config": model_cfg.__dict__,
            "region_config": cfg.__dict__,
            "channels": channels,
            "temporal_offsets_minutes": temporal_offsets,
            "label_contract": {"0": "no_fire", "1": "fire_signal"},
            "history": history,
            "epoch": epoch,
            "metrics": row,
        }
        torch.save(checkpoint, out_dir / "latest.pt")
        score = 0.5 * float(val["precision"]) + 0.5 * float(val["recall"])
        if score > best_score:
            best_score = score
            torch.save(checkpoint, out_dir / "best.pt")

    (out_dir / "history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    print(f"done best_checkpoint={out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
