from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return max(divisor, int(math.ceil(float(value) / float(divisor)) * divisor))


def scale_channels(base: int, width_mult: float, max_channels: int, divisor: int = 8) -> int:
    ch = make_divisible(int(base * float(width_mult)), divisor=divisor)
    return int(min(ch, int(max_channels)))


def scale_depth(base_repeats: int, depth_mult: float) -> int:
    return max(1, int(round(float(base_repeats) * float(depth_mult))))


class ConvBNAct(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        k: int = 1,
        s: int = 1,
        p: int | None = None,
        g: int = 1,
    ):
        super().__init__()
        pad = (int(k) - 1) // 2 if p is None else int(p)
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=int(k), stride=int(s), padding=pad, groups=int(g), bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, channels: int, shortcut: bool = True, e: float = 1.0, k: int = 3):
        super().__init__()
        hidden = max(8, int(round(int(channels) * float(e))))
        self.cv1 = ConvBNAct(channels, hidden, k=1, s=1)
        self.cv2 = ConvBNAct(hidden, channels, k=int(k), s=1)
        self.use_shortcut = bool(shortcut)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv2(self.cv1(x))
        if self.use_shortcut and y.shape == x.shape:
            y = y + x
        return y


class C3k2(nn.Module):
    """C2f-like block reused from the camera model family."""

    def __init__(self, c_in: int, c_out: int, n: int = 2, c3k: bool = False, e: float = 0.5):
        super().__init__()
        hidden = max(8, int(round(int(c_out) * float(e))))
        self.cv1 = ConvBNAct(c_in, 2 * hidden, k=1, s=1)
        k = 5 if bool(c3k) else 3
        self.blocks = nn.ModuleList([Bottleneck(hidden, shortcut=True, e=1.0, k=k) for _ in range(int(n))])
        self.cv2 = ConvBNAct((2 + int(n)) * hidden, c_out, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.cv1(x)
        a, b = y.chunk(2, dim=1)
        feats = [a, b]
        h = b
        for block in self.blocks:
            h = block(h)
            feats.append(h)
        return self.cv2(torch.cat(feats, dim=1))


class SPPF(nn.Module):
    def __init__(self, c_in: int, c_out: int, k: int = 5):
        super().__init__()
        hidden = max(8, int(c_in) // 2)
        self.cv1 = ConvBNAct(c_in, hidden, k=1, s=1)
        self.pool = nn.MaxPool2d(kernel_size=int(k), stride=1, padding=int(k) // 2)
        self.cv2 = ConvBNAct(hidden * 4, c_out, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        y1 = self.pool(x)
        y2 = self.pool(y1)
        y3 = self.pool(y2)
        return self.cv2(torch.cat([x, y1, y2, y3], dim=1))


class ChannelSE(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(8, int(channels) // int(reduction))
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, kernel_size=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.net(x)


class UpBlock(nn.Module):
    def __init__(self, c_in: int, c_skip: int, c_out: int, repeats: int = 2):
        super().__init__()
        self.reduce = ConvBNAct(c_in, c_out, k=1, s=1)
        self.fuse = C3k2(c_out + c_skip, c_out, n=int(repeats), c3k=True, e=0.5)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = self.reduce(x)
        return self.fuse(torch.cat([x, skip], dim=1))
