"""Top-level HeLF model assembly.

HeLF fuses three kinds of heterogeneous features in the latent space of a
diffusion model:

* data features: the latent variables of the variational autoencoder and the
  denoiser features, learned from high-fidelity flow fields;
* geometry and condition features: multiscale signed distance tokens modulated
  by the operating condition (a spatial field and two scalars);
* physical mechanism features: the gradient of a physical energy built from the
  conservation law, the wall and far-field boundary conditions, and the
  Rankine-Hugoniot jump relations, mapped through the decoder into the latent
  space.

A shock saliency map predicted from latent features gates the denoiser and
assigns each physical mechanism to the region where it holds.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
from torch import nn

from .denoiser import SaliencyGatedDenoiser
from .diffusion import GaussianDiffusion
from .geometry_encoder import MultiscaleGeometryEncoder
from .physics import PhysicalMechanismEnergy
from .vae import FlowVAE


class HeLF(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.config = config
        model_cfg = config["model"]
        vae_cfg = model_cfg["vae"]
        geometry_cfg = model_cfg["geometry"]
        denoiser_cfg = model_cfg["denoiser"]
        diffusion_cfg = model_cfg["diffusion"]

        self.vae = FlowVAE(**vae_cfg)
        self.geometry_encoder = MultiscaleGeometryEncoder(**geometry_cfg)
        self.denoiser = SaliencyGatedDenoiser(
            latent_channels=int(vae_cfg["latent_channels"]),
            geometry_condition_dim=int(geometry_cfg["condition_dim"]),
            **denoiser_cfg,
        )
        self.diffusion = GaussianDiffusion(
            train_steps=int(diffusion_cfg["train_steps"]),
            schedule=str(diffusion_cfg.get("schedule", "cosine")),
        )
        # The physical energy has no trainable parameters, so renaming this attribute
        # does not change checkpoint state_dict keys.
        self.physical_energy = PhysicalMechanismEnergy(model_cfg["physics"])

        flow_mean = torch.tensor(config["data"]["flow_mean"], dtype=torch.float32)[None, :, None, None]
        flow_std = torch.tensor(config["data"]["flow_std"], dtype=torch.float32)[None, :, None, None]
        self.register_buffer("flow_mean", flow_mean, persistent=True)
        self.register_buffer("flow_std", flow_std, persistent=True)

    def denormalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        return flow * self.flow_std.to(flow.dtype) + self.flow_mean.to(flow.dtype)

    def normalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        return (flow - self.flow_mean.to(flow.dtype)) / self.flow_std.to(flow.dtype).clamp_min(1e-8)

    def condition_code(self, sdf: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.geometry_encoder(sdf, condition)

    def predict_noise(
        self,
        latent_t: torch.Tensor,
        timestep: torch.Tensor,
        sdf: torch.Tensor,
        condition: torch.Tensor,
        geometry_code: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if geometry_code is None:
            geometry_code = self.condition_code(sdf, condition)
        return self.denoiser(latent_t, timestep, geometry_code)

    def latent_spatial_shape(self, height: int, width: int) -> tuple[int, int]:
        factor = self.vae.downsample_factor
        if height % factor != 0 or width % factor != 0:
            raise ValueError(
                f"Input size {(height, width)} must be divisible by VAE downsample factor {factor}"
            )
        return height // factor, width // factor
