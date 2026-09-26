"""DDIM sampling with latent fusion of physical mechanism and data features.

Physical mechanisms are evaluated as residuals of the decoded flow field, while
the diffusion score lives in the latent space. At each fusion step the sampler

1. decodes the clean estimate of the latent data features,
2. evaluates the physical energy, with the shock region chosen by the saliency
   gate (see :class:`helf.models.physics.PhysicalMechanismEnergy`),
3. back-propagates the energy through the decoder and the denoiser to the latent
   variable, which gives the latent physical score
   ``s_p = -(d z0_hat / d z_t)^T J_D^T grad_q E``,
4. calibrates ``s_p`` to the magnitude of the diffusion score ``s_d``, and
5. adds it to the DDIM proposal with the reliability-gated weight
   ``omega_t = omega_max * r_t * n_t``, where ``r_t = alpha_bar_t`` is the
   reliability of the clean estimate and ``n_t = E / (E + E_ref)`` is the
   correction need.

The notation follows Section 2.4 of the manuscript.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch


class LatentFusionSampler:
    def __init__(self, model: torch.nn.Module, config: Mapping[str, Any]) -> None:
        self.model = model
        self.config = config
        diffusion_cfg = config["model"]["diffusion"]
        physics_cfg = config["model"]["physics"]
        self.sample_steps = int(diffusion_cfg.get("sample_steps", 50))
        self.eta = float(diffusion_cfg.get("eta", 0.0))
        # Config keys are kept from the review-stage release for compatibility with
        # existing private overrides: ``correction_count`` is the number of fusion
        # steps |T_p| and ``correction_steps`` is the set T_p itself.
        self.omega_max = float(physics_cfg.get("omega_max", 0.1))
        self.correction_count = int(physics_cfg.get("correction_count", 8))
        self.correction_steps = physics_cfg.get("correction_steps", "auto")
        self.energy_ref = float(physics_cfg.get("energy_ref", 1.0))

    def _fusion_indices(self, count: int) -> set[int]:
        """Indices of the reverse steps in T_p at which physical mechanisms are fused."""
        if self.correction_count <= 0:
            return set()
        if isinstance(self.correction_steps, (list, tuple)):
            requested = {int(value) for value in self.correction_steps}
            return {index for index in requested if 0 <= index < count}
        # Public review default: distribute the fusion steps through the
        # structurally reliable later 80% of reverse diffusion. Exact paper indices
        # are withheld until the archival release.
        start = max(0, count // 5)
        positions = torch.linspace(start, count - 1, self.correction_count).round().long().tolist()
        return set(int(position) for position in positions)

    def _physical_mechanism_fusion(
        self,
        latent_t: torch.Tensor,
        timestep: torch.Tensor,
        geometry_code: torch.Tensor,
        batch: Mapping[str, torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        with torch.enable_grad():
            latent_variable = latent_t.detach().requires_grad_(True)
            output = self.model.denoiser(latent_variable, timestep, geometry_code.detach())
            latent0 = self.model.diffusion.predict_start_from_noise(
                latent_variable, timestep, output["noise"]
            )
            flow_normalized = self.model.vae.decode(latent0)
            flow_physical = self.model.denormalize_flow(flow_normalized)
            energy = self.model.physical_energy(
                flow_physical,
                batch["sdf"],
                batch["condition_physical"],
                batch["wall_mask"],
                batch["farfield_mask"],
                batch["spacing"],
                output["saliency"],
            )
            energy_gradient = torch.autograd.grad(energy["total"], latent_variable)[0]
            # s_p: latent physical score, i.e. the energy gradient mapped into the latent space.
            latent_physical_score = -energy_gradient
            # s_d: diffusion score of the learned latent distribution (data features).
            data_score = self.model.diffusion.diffusion_score(output["noise"], timestep)
            batch_dims = tuple(range(1, latent_t.ndim))
            data_norm = torch.linalg.vector_norm(data_score, dim=batch_dims, keepdim=True)
            physics_norm = torch.linalg.vector_norm(
                latent_physical_score, dim=batch_dims, keepdim=True
            )
            # Calibrated s_p = s_p / tau_t with tau_t = (||s_p|| + eps) / (||s_d|| + eps).
            # The product form below is the original review-stage computation.
            calibrated_gradient = (
                latent_physical_score * (data_norm + 1e-8) / (physics_norm + 1e-8)
            )
            temperature = (physics_norm + 1e-8) / (data_norm + 1e-8)

            # Reliability gate: r_t is the reliability of the clean estimate and n_t
            # the remaining need for correction.
            reliability = self.model.diffusion.extract(
                self.model.diffusion.alpha_bar, timestep, latent_t.shape
            )
            need = energy["total"].detach() / (energy["total"].detach() + self.energy_ref)
            omega = self.omega_max * reliability * need
            correction = omega * calibrated_gradient.detach()
            diagnostics = {
                "energy": float(energy["total"].detach().cpu()),
                "conservation": float(energy["conservation"].detach().cpu()),
                "boundary": float(energy["boundary"].detach().cpu()),
                "rh": float(energy["rh"].detach().cpu()),
                "shock_fraction": float(energy["shock_fraction"].detach().cpu()),
                "reliability": float(reliability.mean().detach().cpu()),
                "need": float(need.detach().cpu()),
                "omega": float(omega.mean().detach().cpu()),
                "temperature": float(temperature.mean().detach().cpu()),
            }
        return correction, diagnostics

    def sample(
        self,
        batch: Mapping[str, torch.Tensor],
        latent: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, Any]:
        sdf = batch["sdf"]
        condition = batch["condition"]
        batch_size, _, height, width = sdf.shape
        latent_h, latent_w = self.model.latent_spatial_shape(height, width)
        if latent is None:
            latent = torch.randn(
                batch_size,
                self.model.vae.latent_channels,
                latent_h,
                latent_w,
                device=sdf.device,
                dtype=sdf.dtype,
            )
        geometry_code = self.model.condition_code(sdf, condition)
        times = self.model.diffusion.sampling_timesteps(self.sample_steps, sdf.device)
        fusion_indices = self._fusion_indices(len(times))
        intermediates = []
        diagnostics = []
        final_saliency = None

        for index, scalar_t in enumerate(times):
            timestep = torch.full((batch_size,), int(scalar_t.item()), device=sdf.device, dtype=torch.long)
            previous_value = int(times[index + 1].item()) if index + 1 < len(times) else -1
            previous_timestep = torch.full(
                (batch_size,), previous_value, device=sdf.device, dtype=torch.long
            )
            with torch.no_grad():
                output = self.model.denoiser(latent, timestep, geometry_code)
                proposal, latent0 = self.model.diffusion.ddim_step(
                    latent, timestep, previous_timestep, output["noise"], eta=self.eta
                )
                final_saliency = output["saliency"]

            if index in fusion_indices:
                correction, diagnostic = self._physical_mechanism_fusion(
                    latent, timestep, geometry_code, batch
                )
                alpha_prev = (
                    torch.ones((batch_size, 1, 1, 1), device=sdf.device, dtype=sdf.dtype)
                    if previous_value < 0
                    else self.model.diffusion.extract(
                        self.model.diffusion.alpha_bar, previous_timestep, latent.shape
                    )
                )
                kappa = torch.sqrt((1.0 - alpha_prev).clamp_min(1e-8))
                proposal = proposal + kappa * correction
                diagnostic["sampling_index"] = index
                diagnostic["timestep"] = int(scalar_t.item())
                diagnostics.append(diagnostic)

            latent = proposal.detach()
            if return_intermediates:
                intermediates.append(latent0.detach().cpu())

        with torch.no_grad():
            flow_normalized = self.model.vae.decode(latent)
            flow_physical = self.model.denormalize_flow(flow_normalized)
        return {
            "flow": flow_physical,
            "flow_normalized": flow_normalized,
            "latent": latent,
            "saliency": final_saliency,
            "fusion_diagnostics": diagnostics,
            "intermediates": intermediates,
        }
