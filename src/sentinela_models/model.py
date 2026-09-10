from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import C3k2, ChannelSE, ConvBNAct, SPPF, UpBlock, scale_channels, scale_depth


@dataclass(frozen=True)
class SentinelaConfig:
    in_channels: int = 6
    mask_classes: int = 1
    scene_classes: int = 1
    temporal_steps: int = 1
    input_channels_per_timestep: int | None = None
    variant: str = "s"
    input_size: int = 256
    base_channels: int = 32
    width_mult: float = 1.0
    depth_mult: float = 1.0
    max_channels: int = 384
    dropout: float = 0.05
    include_scene_head: bool = True


VARIANTS: dict[str, dict[str, float | int]] = {
    "n": {"base_channels": 24, "width_mult": 0.75, "depth_mult": 0.67, "max_channels": 256},
    "s": {"base_channels": 32, "width_mult": 1.0, "depth_mult": 1.0, "max_channels": 384},
    "m": {"base_channels": 48, "width_mult": 1.15, "depth_mult": 1.35, "max_channels": 512},
    "l": {"base_channels": 64, "width_mult": 1.25, "depth_mult": 1.5, "max_channels": 1024},
}


class SpectralInputAdapter(nn.Module):
    """Maps multispectral bands into the model stem space."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.proj = ConvBNAct(self.in_channels, self.out_channels, k=3, s=1)
        self.mix = nn.Sequential(
            ConvBNAct(self.out_channels, self.out_channels, k=1, s=1),
            ChannelSE(self.out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} input bands, got {x.shape[1]}")
        return self.mix(self.proj(x))


class TemporalInputAdapter(nn.Module):
    """Flattens optional time-series rasters into the Sentinela channel axis."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = int(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 5:
            b, t, c, h, w = x.shape
            x = x.reshape(b, t * c, h, w)
        elif x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W] or [B,T,C,H,W] input, got shape={tuple(x.shape)}")
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} flattened input bands, got {x.shape[1]}")
        return x


class SentinelaEncoder(nn.Module):
    """YOLO/MVP-style multiscale encoder adapted for satellite rasters."""

    def __init__(self, cfg: SentinelaConfig):
        super().__init__()
        c1 = scale_channels(cfg.base_channels, cfg.width_mult, cfg.max_channels)
        c2 = scale_channels(cfg.base_channels * 2, cfg.width_mult, cfg.max_channels)
        c3 = scale_channels(cfg.base_channels * 4, cfg.width_mult, cfg.max_channels)
        c4 = scale_channels(cfg.base_channels * 8, cfg.width_mult, cfg.max_channels)
        c5 = scale_channels(cfg.base_channels * 10, cfg.width_mult, cfg.max_channels)

        d1 = scale_depth(1, cfg.depth_mult)
        d2 = scale_depth(2, cfg.depth_mult)
        d3 = scale_depth(2, cfg.depth_mult)

        self.channels = [c1, c2, c3, c4, c5]
        self.adapter = SpectralInputAdapter(cfg.in_channels, c1)
        self.stem = ConvBNAct(c1, c1, k=3, s=1)

        self.down1 = ConvBNAct(c1, c2, k=3, s=2)
        self.stage1 = C3k2(c2, c2, n=d1, c3k=False)

        self.down2 = ConvBNAct(c2, c3, k=3, s=2)
        self.stage2 = C3k2(c3, c3, n=d2, c3k=False)

        self.down3 = ConvBNAct(c3, c4, k=3, s=2)
        self.stage3 = C3k2(c4, c4, n=d3, c3k=True)

        self.down4 = ConvBNAct(c4, c5, k=3, s=2)
        self.stage4 = nn.Sequential(C3k2(c5, c5, n=d2, c3k=True), SPPF(c5, c5))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        p1 = self.stem(self.adapter(x))
        p2 = self.stage1(self.down1(p1))
        p3 = self.stage2(self.down2(p2))
        p4 = self.stage3(self.down3(p3))
        p5 = self.stage4(self.down4(p4))
        return [p1, p2, p3, p4, p5]


class SegmentationHead(nn.Module):
    def __init__(self, channels: list[int], mask_classes: int, dropout: float = 0.05):
        super().__init__()
        c1, c2, c3, c4, c5 = channels
        self.up4 = UpBlock(c5, c4, c4)
        self.up3 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up1 = UpBlock(c2, c1, c1)
        self.refine = nn.Sequential(
            ConvBNAct(c1, c1, k=3, s=1),
            nn.Dropout2d(float(dropout)),
            ConvBNAct(c1, c1, k=3, s=1),
        )
        self.mask_logits = nn.Conv2d(c1, int(mask_classes), kernel_size=1)

    def forward(self, feats: list[torch.Tensor], output_size: tuple[int, int]) -> torch.Tensor:
        p1, p2, p3, p4, p5 = feats
        x = self.up4(p5, p4)
        x = self.up3(x, p3)
        x = self.up2(x, p2)
        x = self.up1(x, p1)
        x = self.mask_logits(self.refine(x))
        if x.shape[-2:] != output_size:
            x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return x


class SceneHead(nn.Module):
    def __init__(self, channels: int, scene_classes: int = 1, dropout: float = 0.05):
        super().__init__()
        hidden = max(32, channels // 2)
        self.scene_classes = int(scene_classes)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden, self.scene_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits.squeeze(-1) if self.scene_classes == 1 else logits


class SentinelaModel(nn.Module):
    """
    Satellite fire segmentation model.

    Forward returns raw logits:
      - mask_logits: [B, mask_classes, H, W]
      - scene_logits: [B] when include_scene_head is enabled
    """

    def __init__(self, config: SentinelaConfig | None = None, **kwargs: Any):
        super().__init__()
        cfg = config or SentinelaConfig(**kwargs)
        variant = str(cfg.variant).lower().strip()
        if variant in VARIANTS:
            preset = VARIANTS[variant]
            cfg = SentinelaConfig(
                in_channels=cfg.in_channels,
                mask_classes=cfg.mask_classes,
                scene_classes=cfg.scene_classes,
                temporal_steps=cfg.temporal_steps,
                input_channels_per_timestep=cfg.input_channels_per_timestep,
                variant=variant,
                input_size=cfg.input_size,
                base_channels=int(preset["base_channels"]),
                width_mult=float(preset["width_mult"]),
                depth_mult=float(preset["depth_mult"]),
                max_channels=int(preset["max_channels"]),
                dropout=cfg.dropout,
                include_scene_head=cfg.include_scene_head,
            )
        elif variant not in ("custom", ""):
            raise ValueError(f"Unsupported Sentinela variant '{cfg.variant}'. Choose from {sorted(VARIANTS)} or custom.")

        self.config = cfg
        self.temporal_adapter = TemporalInputAdapter(cfg.in_channels)
        self.encoder = SentinelaEncoder(cfg)
        self.seg_head = SegmentationHead(self.encoder.channels, cfg.mask_classes, dropout=cfg.dropout)
        self.scene_head = (
            SceneHead(self.encoder.channels[-1], scene_classes=cfg.scene_classes, dropout=cfg.dropout)
            if cfg.include_scene_head
            else None
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        output_size = tuple(x.shape[-2:])
        x = self.temporal_adapter(x)
        feats = self.encoder(x)
        mask_logits = self.seg_head(feats, output_size=output_size)  # type: ignore[arg-type]
        out = {"mask_logits": mask_logits}
        if self.scene_head is not None:
            out["scene_logits"] = self.scene_head(feats[-1])
        return out

    @torch.no_grad()
    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> dict[str, torch.Tensor]:
        self.eval()
        out = self(x)
        if self.config.mask_classes == 1:
            mask_prob = torch.sigmoid(out["mask_logits"])
            mask = (mask_prob >= float(threshold)).to(mask_prob.dtype)
        else:
            mask_prob = torch.softmax(out["mask_logits"], dim=1)
            mask = torch.argmax(mask_prob, dim=1)
        pred = {
            **out,
            "mask_prob": mask_prob,
            "mask": mask,
        }
        if "scene_logits" in out:
            if self.config.scene_classes == 1:
                pred["scene_prob"] = torch.sigmoid(out["scene_logits"])
            else:
                pred["scene_prob"] = torch.softmax(out["scene_logits"], dim=-1)
        return pred

    @torch.no_grad()
    def infer_shapes(self, device: torch.device | str = "cpu") -> dict[str, tuple[int, ...]]:
        d = torch.device(device)
        if self.config.temporal_steps > 1:
            channels = self.config.input_channels_per_timestep or self.config.in_channels // self.config.temporal_steps
            x = torch.zeros(1, self.config.temporal_steps, channels, self.config.input_size, self.config.input_size, device=d)
        else:
            x = torch.zeros(1, self.config.in_channels, self.config.input_size, self.config.input_size, device=d)
        self.eval()
        out = self(x)
        return {k: tuple(v.shape) for k, v in out.items()}


def build_sentinela(**kwargs: Any) -> SentinelaModel:
    return SentinelaModel(SentinelaConfig(**kwargs))
