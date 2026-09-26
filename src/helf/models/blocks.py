"""Conditioning, deformable aggregation, and sparse attention blocks."""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
except Exception:  # pragma: no cover - installation dependent
    deform_conv2d = None


def sinusoidal_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequency = torch.exp(
        -math.log(10000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half - 1, 1)
    )
    angles = timestep.float()[:, None] * frequency[None]
    embedding = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class AdaGroupNorm(nn.Module):
    def __init__(self, channels: int, condition_dim: int, groups: int = 8) -> None:
        super().__init__()
        groups = min(groups, channels)
        while channels % groups != 0 and groups > 1:
            groups -= 1
        self.norm = nn.GroupNorm(groups, channels, affine=False)
        self.affine = nn.Linear(condition_dim, channels * 2)
        nn.init.zeros_(self.affine.weight)
        nn.init.zeros_(self.affine.bias)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        scale, shift = self.affine(condition).chunk(2, dim=-1)
        return self.norm(x) * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]


class ConditionedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        condition_dim: int,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = AdaGroupNorm(in_channels, condition_dim, groups)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = AdaGroupNorm(out_channels, condition_dim, groups)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
        self.skip = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(x, condition)))
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden, condition))))
        return self.skip(x) + hidden


class SaliencyDeformableConv(nn.Module):
    """Deformable convolution with a deterministic standard-convolution fallback."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        points = kernel_size * kernel_size
        self.offset_mask = nn.Conv2d(channels + 1, points * 3, 3, padding=1)
        nn.init.zeros_(self.offset_mask.weight)
        nn.init.zeros_(self.offset_mask.bias)
        self.weight = nn.Parameter(torch.empty(channels, channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    @property
    def uses_native_deformable_conv(self) -> bool:
        return deform_conv2d is not None

    def forward(self, x: torch.Tensor, saliency: torch.Tensor) -> torch.Tensor:
        saliency = F.interpolate(saliency, size=x.shape[-2:], mode="bilinear", align_corners=False)
        offset_mask = self.offset_mask(torch.cat([x, saliency], dim=1))
        points = self.kernel_size * self.kernel_size
        offset = offset_mask[:, : 2 * points]
        mask = offset_mask[:, 2 * points :].sigmoid()
        if deform_conv2d is not None:
            return deform_conv2d(
                x,
                offset,
                self.weight,
                self.bias,
                padding=(self.padding, self.padding),
                mask=mask,
            )
        # Fallback keeps the public pipeline functional where torchvision C++ ops are absent.
        return F.conv2d(x * (1.0 + saliency), self.weight, self.bias, padding=self.padding)


class SparseGlobalAttention(nn.Module):
    def __init__(self, channels: int, heads: int = 8) -> None:
        super().__init__()
        if channels % heads != 0:
            raise ValueError("channels must be divisible by heads")
        self.heads = heads
        self.head_dim = channels // heads
        self.scale = self.head_dim**-0.5
        self.to_q = nn.Linear(channels, channels)
        self.to_k = nn.Linear(channels, channels)
        self.to_v = nn.Linear(channels, channels)
        self.proj = nn.Linear(channels, channels)

    def forward(self, feature: torch.Tensor, saliency: torch.Tensor, fraction: float) -> torch.Tensor:
        batch, channels, height, width = feature.shape
        tokens = feature.flatten(2).transpose(1, 2)
        count = tokens.shape[1]
        selected_count = max(1, min(count, math.ceil(float(fraction) * count)))
        saliency_flat = F.interpolate(saliency, (height, width), mode="bilinear", align_corners=False).flatten(1)
        indices = saliency_flat.topk(selected_count, dim=1).indices

        q_all = self.to_q(tokens).view(batch, count, self.heads, self.head_dim).transpose(1, 2)
        k = self.to_k(tokens).view(batch, count, self.heads, self.head_dim).transpose(1, 2)
        v = self.to_v(tokens).view(batch, count, self.heads, self.head_dim).transpose(1, 2)
        gather_index = indices[:, None, :, None].expand(-1, self.heads, -1, self.head_dim)
        q = torch.gather(q_all, 2, gather_index)
        attention = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * self.scale, dim=-1)
        selected = torch.matmul(attention, v).transpose(1, 2).reshape(batch, selected_count, channels)
        selected = self.proj(selected)

        output = tokens.clone()
        scatter_index = indices[:, :, None].expand(-1, -1, channels)
        output.scatter_add_(1, scatter_index, selected)
        return output.transpose(1, 2).reshape(batch, channels, height, width)
