"""Saliency-gated conditional U-Net denoiser.

The predicted shock saliency map gates the latent data features in two forms: a
soft form that modulates deformable local aggregation, and a hard form (top-k
selection) that decides which positions enter sparse global attention. The same
map is reused by the latent fusion sampler to assign physical mechanisms to
shock and smooth regions.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import (
    ConditionedResidualBlock,
    SaliencyDeformableConv,
    SparseGlobalAttention,
    sinusoidal_embedding,
)


class SaliencyGatedDenoiser(nn.Module):
    def __init__(
        self,
        latent_channels: int,
        geometry_condition_dim: int,
        channels: Iterable[int] = (128, 256, 384, 512),
        time_dim: int = 256,
        groups: int = 8,
        attention_heads: int = 8,
        deformable_kernel: int = 3,
        salient_fraction: float = 0.05,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if len(channels) != 4:
            raise ValueError("denoiser channels must contain exactly four levels")
        c0, c1, c2, c3 = channels
        self.time_dim = time_dim
        self.salient_fraction = float(salient_fraction)
        condition_dim = geometry_condition_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.condition_fusion = nn.Sequential(
            nn.Linear(condition_dim * 2, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )

        self.input_conv = nn.Conv2d(latent_channels, c0, 3, padding=1)
        self.enc0 = ConditionedResidualBlock(c0, c0, condition_dim, groups, dropout)
        self.down1 = nn.Conv2d(c0, c1, 4, stride=2, padding=1)
        self.enc1 = ConditionedResidualBlock(c1, c1, condition_dim, groups, dropout)
        self.down2 = nn.Conv2d(c1, c2, 4, stride=2, padding=1)
        self.enc2 = ConditionedResidualBlock(c2, c2, condition_dim, groups, dropout)
        self.down3 = nn.Conv2d(c2, c3, 4, stride=2, padding=1)
        self.bottleneck = ConditionedResidualBlock(c3, c3, condition_dim, groups, dropout)

        self.saliency_head = nn.Sequential(
            nn.Conv2d(c3, c3 // 2, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(c3 // 2, 1, 1),
        )
        self.deformable = SaliencyDeformableConv(c3, deformable_kernel)
        self.sparse_attention = SparseGlobalAttention(c3, attention_heads)
        self.post_bottleneck = ConditionedResidualBlock(c3, c3, condition_dim, groups, dropout)

        self.up2 = nn.ConvTranspose2d(c3, c2, 4, stride=2, padding=1)
        self.dec2 = ConditionedResidualBlock(c2 * 2, c2, condition_dim, groups, dropout)
        self.up1 = nn.ConvTranspose2d(c2, c1, 4, stride=2, padding=1)
        self.dec1 = ConditionedResidualBlock(c1 * 2, c1, condition_dim, groups, dropout)
        self.up0 = nn.ConvTranspose2d(c1, c0, 4, stride=2, padding=1)
        self.dec0 = ConditionedResidualBlock(c0 * 2, c0, condition_dim, groups, dropout)
        self.output = nn.Conv2d(c0, latent_channels, 3, padding=1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _condition(self, timestep: torch.Tensor, geometry_code: torch.Tensor) -> torch.Tensor:
        time_code = self.time_mlp(sinusoidal_embedding(timestep, self.time_dim).to(geometry_code.dtype))
        return self.condition_fusion(torch.cat([time_code, geometry_code], dim=-1))

    @staticmethod
    def _resize_like(x: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != reference.shape[-2:]:
            x = F.interpolate(x, size=reference.shape[-2:], mode="bilinear", align_corners=False)
        return x

    def forward(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        geometry_code: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        condition = self._condition(timestep, geometry_code)
        x0 = self.enc0(self.input_conv(latent), condition)
        x1 = self.enc1(self.down1(x0), condition)
        x2 = self.enc2(self.down2(x1), condition)
        x3 = self.bottleneck(self.down3(x2), condition)

        saliency_logits = self.saliency_head(x3)
        saliency = torch.sigmoid(saliency_logits)
        local = self.deformable(x3, saliency)
        global_feature = self.sparse_attention(local, saliency, self.salient_fraction)
        x3 = self.post_bottleneck(x3 + local + global_feature, condition)

        y2 = self._resize_like(self.up2(x3), x2)
        y2 = self.dec2(torch.cat([y2, x2], dim=1), condition)
        y1 = self._resize_like(self.up1(y2), x1)
        y1 = self.dec1(torch.cat([y1, x1], dim=1), condition)
        y0 = self._resize_like(self.up0(y1), x0)
        y0 = self.dec0(torch.cat([y0, x0], dim=1), condition)
        return {
            "noise": self.output(y0),
            "saliency": saliency,
            "saliency_logits": saliency_logits,
            "features": x3,
        }
