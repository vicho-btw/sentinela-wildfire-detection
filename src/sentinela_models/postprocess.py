from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


VerificationStatus = Literal["confirmed", "possible", "not_seen", "unavailable"]


@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    confidence: float
    fire_area_pixels: int
    centroid_xy: tuple[float, float] | None
    threshold: float

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "confidence": self.confidence,
            "fire_area_pixels": self.fire_area_pixels,
            "centroid_xy": self.centroid_xy,
            "threshold": self.threshold,
        }


def mask_to_verification(
    mask_prob: torch.Tensor,
    threshold: float = 0.5,
    min_pixels_confirmed: int = 25,
    min_pixels_possible: int = 5,
) -> VerificationResult:
    """
    Convert one fire probability mask into an incident-level verification result.

    Expects [H,W], [1,H,W], or [1,1,H,W]. Coordinates are returned as image-space
    (x, y) pixel centroids; geospatial conversion belongs in the dataset layer.
    """
    if mask_prob.ndim == 4:
        if mask_prob.shape[0] != 1:
            raise ValueError("mask_to_verification expects a single sample when given [B,C,H,W].")
        mask_prob = mask_prob[0]
    if mask_prob.ndim == 3:
        if mask_prob.shape[0] != 1:
            raise ValueError("mask_to_verification expects one mask channel.")
        mask_prob = mask_prob[0]
    if mask_prob.ndim != 2:
        raise ValueError(f"Expected 2D mask probability, got shape={tuple(mask_prob.shape)}")

    prob = mask_prob.detach().float().cpu().clamp(0, 1)
    mask = prob >= float(threshold)
    area = int(mask.sum().item())
    confidence = float(prob[mask].mean().item()) if area > 0 else float(prob.max().item())

    if area >= int(min_pixels_confirmed):
        status: VerificationStatus = "confirmed"
    elif area >= int(min_pixels_possible):
        status = "possible"
    else:
        status = "not_seen"

    centroid: tuple[float, float] | None = None
    if area > 0:
        ys, xs = torch.where(mask)
        centroid = (float(xs.float().mean().item()), float(ys.float().mean().item()))

    return VerificationResult(
        status=status,
        confidence=confidence,
        fire_area_pixels=area,
        centroid_xy=centroid,
        threshold=float(threshold),
    )
