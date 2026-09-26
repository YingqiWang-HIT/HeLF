"""Convolutional variational autoencoder for four-channel flow fields."""

from __future__ import annotations

from typing import Iterable, Tuple

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        )
        self.skip = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x) + self.skip(x)


class FlowVAE(nn.Module):
    """VAE with three downsampling stages by default (approximately 1/8 scale)."""

    def __init__(
        self,
        in_channels: int = 4,
        channels: Iterable[int] = (64, 128, 256),
        latent_channels: int = 4,
    ) -> None:
        super().__init__()
        channels = list(channels)
        if not channels:
            raise ValueError("channels must not be empty")
        encoder = []
        current = in_channels
        for channel in channels:
            encoder.append(ConvBlock(current, channel, stride=2))
            current = channel
        self.encoder = nn.Sequential(*encoder)
        self.to_moments = nn.Conv2d(current, 2 * latent_channels, 1)

        decoder = []
        current = latent_channels
        for channel in reversed(channels):
            decoder.extend(
                [
                    nn.ConvTranspose2d(current, channel, 4, stride=2, padding=1),
                    ConvBlock(channel, channel),
                ]
            )
            current = channel
        self.decoder = nn.Sequential(*decoder)
        self.to_flow = nn.Conv2d(current, in_channels, 3, padding=1)
        self.latent_channels = latent_channels
        self.downsample_factor = 2 ** len(channels)

    def encode(self, flow: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moments = self.to_moments(self.encoder(flow))
        mean, logvar = moments.chunk(2, dim=1)
        logvar = logvar.clamp(-20.0, 10.0)
        if sample:
            latent = mean + torch.exp(0.5 * logvar) * torch.randn_like(mean)
        else:
            latent = mean
        return latent, mean, logvar

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.to_flow(self.decoder(latent))

    def forward(self, flow: torch.Tensor, sample: bool = True) -> dict[str, torch.Tensor]:
        latent, mean, logvar = self.encode(flow, sample=sample)
        reconstruction = self.decode(latent)
        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
            "reconstruction": reconstruction,
        }

    @staticmethod
    def kl_loss(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1.0 + logvar - mean.square() - logvar.exp()).mean()
