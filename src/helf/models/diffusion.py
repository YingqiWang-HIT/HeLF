"""Variance-preserving diffusion schedules and DDIM transitions."""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn


def cosine_beta_schedule(steps: int, s: float = 0.008) -> torch.Tensor:
    x = torch.linspace(0, steps, steps + 1, dtype=torch.float64)
    alpha_bar = torch.cos(((x / steps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(1e-5, 0.999).float()


def linear_beta_schedule(steps: int) -> torch.Tensor:
    scale = 1000.0 / steps
    return torch.linspace(scale * 1e-4, scale * 2e-2, steps).clamp(max=0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self, train_steps: int = 1000, schedule: str = "cosine") -> None:
        super().__init__()
        if schedule == "cosine":
            betas = cosine_beta_schedule(train_steps)
        elif schedule == "linear":
            betas = linear_beta_schedule(train_steps)
        else:
            raise ValueError(f"Unsupported noise schedule: {schedule}")
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1), alpha_bar[:-1]], dim=0)
        self.train_steps = int(train_steps)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("alpha_bar_prev", alpha_bar_prev)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1.0 - alpha_bar).sqrt())

    @staticmethod
    def extract(values: torch.Tensor, timestep: torch.Tensor, shape: torch.Size) -> torch.Tensor:
        gathered = values.gather(0, timestep)
        return gathered.reshape(timestep.shape[0], *((1,) * (len(shape) - 1)))

    def q_sample(
        self,
        latent0: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(latent0)
        return (
            self.extract(self.sqrt_alpha_bar, timestep, latent0.shape) * latent0
            + self.extract(self.sqrt_one_minus_alpha_bar, timestep, latent0.shape) * noise,
            noise,
        )

    def predict_start_from_noise(
        self,
        latent_t: torch.Tensor,
        timestep: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        alpha_bar = self.extract(self.alpha_bar, timestep, latent_t.shape)
        return (latent_t - (1.0 - alpha_bar).sqrt() * noise) / alpha_bar.sqrt().clamp_min(1e-8)

    def diffusion_score(self, noise: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        denominator = self.extract(self.sqrt_one_minus_alpha_bar, timestep, noise.shape)
        return -noise / denominator.clamp_min(1e-8)

    def sampling_timesteps(self, sample_steps: int, device: torch.device) -> torch.Tensor:
        if sample_steps > self.train_steps:
            raise ValueError("sample_steps cannot exceed train_steps")
        return torch.linspace(self.train_steps - 1, 0, sample_steps, device=device).round().long()

    def ddim_step(
        self,
        latent_t: torch.Tensor,
        timestep: torch.Tensor,
        previous_timestep: torch.Tensor,
        predicted_noise: torch.Tensor,
        eta: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha_t = self.extract(self.alpha_bar, timestep, latent_t.shape)
        previous_safe = previous_timestep.clamp_min(0)
        alpha_prev = self.extract(self.alpha_bar, previous_safe, latent_t.shape)
        terminal = (previous_timestep < 0).reshape(previous_timestep.shape[0], *((1,) * (latent_t.ndim - 1)))
        alpha_prev = torch.where(terminal, torch.ones_like(alpha_prev), alpha_prev)
        latent0 = (latent_t - (1.0 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt().clamp_min(1e-8)
        sigma = eta * torch.sqrt(
            ((1.0 - alpha_prev) / (1.0 - alpha_t).clamp_min(1e-8))
            * (1.0 - alpha_t / alpha_prev.clamp_min(1e-8))
        ).clamp_min(0.0)
        direction = torch.sqrt((1.0 - alpha_prev - sigma.square()).clamp_min(0.0)) * predicted_noise
        noise = torch.randn_like(latent_t) if eta > 0 else torch.zeros_like(latent_t)
        previous = alpha_prev.sqrt() * latent0 + direction + sigma * noise
        return previous, latent0
