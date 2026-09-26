"""Losses for the three-stage HeLF training strategy."""

from __future__ import annotations

from typing import Any, Dict, Mapping

import torch
import torch.nn.functional as F

from helf.models.physics import gradient_indicator


def saliency_target(flow: torch.Tensor) -> torch.Tensor:
    """Standardized gradient indicator used to supervise the shock saliency map."""
    indicator = gradient_indicator(flow)
    mean = indicator.mean(dim=(2, 3), keepdim=True)
    std = indicator.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return torch.sigmoid((indicator - mean) / std)


def _diffusion_terms(model: torch.nn.Module, batch: Mapping[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    with torch.set_grad_enabled(any(parameter.requires_grad for parameter in model.vae.parameters())):
        latent0, _, _ = model.vae.encode(batch["flow"], sample=True)
    timestep = torch.randint(
        0, model.diffusion.train_steps, (latent0.shape[0],), device=latent0.device, dtype=torch.long
    )
    latent_t, noise = model.diffusion.q_sample(latent0, timestep)
    geometry_code = model.condition_code(batch["sdf"], batch["condition"])
    prediction = model.denoiser(latent_t, timestep, geometry_code)
    predicted_latent0 = model.diffusion.predict_start_from_noise(latent_t, timestep, prediction["noise"])
    return {
        "latent0": latent0,
        "latent_t": latent_t,
        "timestep": timestep,
        "noise": noise,
        "predicted_noise": prediction["noise"],
        "predicted_latent0": predicted_latent0,
        "saliency": prediction["saliency"],
        "saliency_logits": prediction["saliency_logits"],
    }


def _condition_continuity(model: torch.nn.Module, sdf: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    delta = 0.08 * torch.randn_like(condition)
    condition1 = condition - delta
    condition2 = condition + delta
    lam = torch.rand(condition.shape[0], 1, device=condition.device, dtype=condition.dtype)
    interpolated = lam * condition1 + (1.0 - lam) * condition2
    h1 = model.condition_code(sdf, condition1)
    h2 = model.condition_code(sdf, condition2)
    h_mid = model.condition_code(sdf, interpolated)
    return F.mse_loss(h_mid, lam * h1 + (1.0 - lam) * h2)


def compute_stage_loss(
    model: torch.nn.Module,
    batch: Mapping[str, torch.Tensor],
    stage: str,
    epoch: int,
    config: Mapping[str, Any],
) -> tuple[torch.Tensor, Dict[str, float]]:
    stage = stage.upper()
    loss_cfg = config["training"]["loss"]

    if stage == "A":
        output = model.vae(batch["flow"], sample=True)
        reconstruction = F.mse_loss(output["reconstruction"], batch["flow"])
        kl = model.vae.kl_loss(output["mean"], output["logvar"])
        warmup = max(int(loss_cfg.get("kl_warmup_epochs", 1)), 1)
        beta = float(loss_cfg.get("beta_max", 1e-4)) * min(1.0, (epoch + 1) / warmup)
        total = reconstruction + beta * kl
        return total, {
            "loss": float(total.detach()),
            "reconstruction": float(reconstruction.detach()),
            "kl": float(kl.detach()),
            "beta": beta,
        }

    terms = _diffusion_terms(model, batch)
    diffusion = F.mse_loss(terms["predicted_noise"], terms["noise"])
    target = F.interpolate(
        saliency_target(batch["flow"]),
        size=terms["saliency_logits"].shape[-2:],
        mode="bilinear",
        align_corners=False,
    )
    pos_weight = torch.as_tensor(
        float(loss_cfg.get("saliency_pos_weight", 1.0)),
        device=target.device,
        dtype=target.dtype,
    )
    saliency = F.binary_cross_entropy_with_logits(
        terms["saliency_logits"], target, pos_weight=pos_weight
    )
    continuity = _condition_continuity(model, batch["sdf"], batch["condition"])
    total = (
        diffusion
        + float(loss_cfg.get("lambda_sal", 1.0)) * saliency
        + float(loss_cfg.get("lambda_cont", 1.0)) * continuity
    )
    metrics: Dict[str, float] = {
        "diffusion": float(diffusion.detach()),
        "saliency": float(saliency.detach()),
        "continuity": float(continuity.detach()),
    }

    if stage == "C":
        reconstruction_flow = model.vae.decode(terms["predicted_latent0"])
        reconstruction = F.l1_loss(reconstruction_flow, batch["flow"])
        physical_flow = model.denormalize_flow(reconstruction_flow)
        physics = model.physical_energy(
            physical_flow,
            batch["sdf"],
            batch["condition_physical"],
            batch["wall_mask"],
            batch["farfield_mask"],
            batch["spacing"],
            terms["saliency"],
        )
        total = (
            total
            + float(loss_cfg.get("lambda_phy", 1.0)) * physics["total"]
            + float(loss_cfg.get("lambda_rec", 1.0)) * reconstruction
        )
        metrics.update(
            {
                "physics": float(physics["total"].detach()),
                "physical_conservation": float(physics["conservation"].detach()),
                "physical_boundary": float(physics["boundary"].detach()),
                "physical_rh": float(physics["rh"].detach()),
                "reconstruction": float(reconstruction.detach()),
            }
        )
    metrics["loss"] = float(total.detach())
    return total, metrics
