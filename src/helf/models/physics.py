"""Differentiable physical energy for compressible transonic flow fields.

The energy combines three physical mechanisms, each applied only where it holds:

* the differential conservation law of the inviscid compressible equations in the
  smooth region ``Omega_c``;
* the Rankine-Hugoniot jump relations in the shock region ``Omega_s``;
* no-penetration and freestream conditions on the airfoil wall and far field.

The shock region is chosen by the saliency gate (``shock_partition: saliency``).
Two alternative partitions are provided for the ablation reported in the
manuscript: ``gradient`` selects the region from the gradient indicator of the
decoded field, and ``none`` applies the conservation residual to the whole domain
without a Rankine-Hugoniot term.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn
import torch.nn.functional as F


def _gradient(field: torch.Tensor, spacing: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Central differences with replicated boundary padding.

    ``field`` has shape [B, C, H, W], spacing has shape [B, 2] as (dy, dx).
    """
    padded_x = F.pad(field, (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(field, (0, 0, 1, 1), mode="replicate")
    dx = spacing[:, 1].reshape(-1, 1, 1, 1).clamp_min(1e-8)
    dy = spacing[:, 0].reshape(-1, 1, 1, 1).clamp_min(1e-8)
    derivative_x = (padded_x[..., 2:] - padded_x[..., :-2]) / (2.0 * dx)
    derivative_y = (padded_y[..., 2:, :] - padded_y[..., :-2, :]) / (2.0 * dy)
    return derivative_x, derivative_y


def gradient_indicator(flow: torch.Tensor) -> torch.Tensor:
    """Joint gradient magnitude of density, pressure, and velocity on a unit grid.

    This is the indicator ``G`` used to supervise the saliency map. ``flow`` has
    shape [B, 4, H, W] ordered as rho, u, v, p; the result has shape [B, 1, H, W].
    """
    unit_spacing = torch.ones(flow.shape[0], 2, device=flow.device, dtype=flow.dtype)
    rho_x, rho_y = _gradient(flow[:, 0:1], unit_spacing)
    u_x, u_y = _gradient(flow[:, 1:3], unit_spacing)
    p_x, p_y = _gradient(flow[:, 3:4], unit_spacing)
    rho_grad = torch.sqrt(rho_x.square() + rho_y.square() + 1e-8)
    p_grad = torch.sqrt(p_x.square() + p_y.square() + 1e-8)
    u_grad = torch.sqrt(u_x.square() + u_y.square() + 1e-8).sum(dim=1, keepdim=True)
    return (rho_grad + p_grad + u_grad) / 3.0


def _top_fraction_mask(score: torch.Tensor, quantile: float) -> torch.Tensor:
    """Binary mask of the positions above the per-sample ``quantile`` of ``score``."""
    flat = score.flatten(1)
    selected = max(1, math.ceil((1.0 - quantile) * flat.shape[1]))
    indices = flat.topk(selected, dim=1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.reshape_as(score)


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.ndim < value.ndim:
        mask = mask.unsqueeze(1)
    weighted = value * mask
    return weighted.sum() / mask.expand_as(value).sum().clamp_min(1.0)


SHOCK_PARTITIONS = ("saliency", "gradient", "none")


class PhysicalMechanismEnergy(nn.Module):
    """Conservation, boundary, and Rankine-Hugoniot energies gated by a shock region.

    Configuration keys (``model.physics``):

    ``shock_partition``
        ``saliency`` (default) uses the predicted saliency map; if no saliency is
        passed, it falls back to the pressure-gradient magnitude, as in the
        original implementation. ``gradient`` always uses :func:`gradient_indicator`
        of the flow being evaluated. ``none`` disables the partition.
    ``shock_quantile``
        The shock region contains the positions above this per-sample quantile of
        the selection score (0.95 keeps the top 5%).
    ``shock_dilation``
        Radius ``r`` of a square morphological dilation applied to the selected
        region. The default of 0 keeps the original behavior (no dilation).
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.gamma = float(config.get("gamma", 1.4))
        self.lambda_cons = float(config.get("lambda_cons", 1.0))
        self.lambda_bc = float(config.get("lambda_bc", 1.0))
        self.lambda_rh = float(config.get("lambda_rh", 1.0))
        self.shock_quantile = float(config.get("shock_quantile", 0.95))
        self.shock_dilation = int(config.get("shock_dilation", 0))
        self.shock_partition = str(config.get("shock_partition", "saliency")).lower()
        if self.shock_partition not in SHOCK_PARTITIONS:
            raise ValueError(
                f"shock_partition must be one of {SHOCK_PARTITIONS}, got {self.shock_partition!r}"
            )
        if self.shock_dilation < 0:
            raise ValueError("shock_dilation must be non-negative")

    def shock_mask(self, flow: torch.Tensor, saliency: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return the shock region ``Omega_s`` as a {0, 1} mask of shape [B, 1, H, W]."""
        if self.shock_partition == "none":
            return torch.zeros_like(flow[:, :1])
        if self.shock_partition == "gradient":
            score = gradient_indicator(flow)
        elif saliency is not None:
            score = F.interpolate(
                saliency, size=flow.shape[-2:], mode="bilinear", align_corners=False
            )
        else:
            pressure = flow[:, 3:4]
            unit_spacing = torch.ones(flow.shape[0], 2, device=flow.device, dtype=flow.dtype)
            gx, gy = _gradient(pressure, unit_spacing)
            score = torch.sqrt(gx.square() + gy.square() + 1e-12)
        mask = _top_fraction_mask(score, self.shock_quantile)
        if self.shock_dilation > 0:
            radius = self.shock_dilation
            mask = F.max_pool2d(mask, kernel_size=2 * radius + 1, stride=1, padding=radius)
        return mask.to(flow.dtype)

    def _rh_energy(self, flow: torch.Tensor, shock: torch.Tensor, spacing: torch.Tensor) -> torch.Tensor:
        rho = flow[:, 0:1].clamp_min(1e-5)
        u = flow[:, 1:2]
        v = flow[:, 2:3]
        pressure = flow[:, 3:4].clamp_min(1e-5)
        gx, gy = _gradient(pressure, spacing)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-12)
        nx, ny = gx / magnitude, gy / magnitude

        batch, _, height, width = flow.shape
        y, x = torch.meshgrid(
            torch.linspace(-1.0, 1.0, height, device=flow.device, dtype=flow.dtype),
            torch.linspace(-1.0, 1.0, width, device=flow.device, dtype=flow.dtype),
            indexing="ij",
        )
        grid = torch.stack([x, y], dim=-1)[None].expand(batch, -1, -1, -1)
        offset = torch.stack(
            [2.0 * nx[:, 0] / max(width - 1, 1), 2.0 * ny[:, 0] / max(height - 1, 1)], dim=-1
        )
        plus = F.grid_sample(flow, grid + offset, align_corners=True, padding_mode="border")
        minus = F.grid_sample(flow, grid - offset, align_corners=True, padding_mode="border")

        def state(values: torch.Tensor) -> tuple[torch.Tensor, ...]:
            r = values[:, 0:1].clamp_min(1e-5)
            ux = values[:, 1:2]
            uy = values[:, 2:3]
            p = values[:, 3:4].clamp_min(1e-5)
            un = ux * nx + uy * ny
            h = self.gamma / (self.gamma - 1.0) * p / r
            return r, ux, uy, p, un, h

        rp, up, vp, pp, unp, hp = state(plus)
        rm, um, vm, pm, unm, hm = state(minus)
        mass = rp * unp - rm * unm
        momentum = (pp + rp * unp.square()) - (pm + rm * unm.square())
        energy = (hp + 0.5 * (up.square() + vp.square())) - (
            hm + 0.5 * (um.square() + vm.square())
        )
        mismatch = mass.square() + momentum.square() + energy.square()
        return _masked_mean(mismatch, shock)

    def forward(
        self,
        flow: torch.Tensor,
        sdf: torch.Tensor,
        condition_physical: torch.Tensor,
        wall_mask: torch.Tensor,
        farfield_mask: torch.Tensor,
        spacing: torch.Tensor,
        saliency: Optional[torch.Tensor] = None,
        shock: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Evaluate the physical energy of ``flow`` in physical units.

        ``shock`` optionally fixes the shock region (a [B, 1, H, W] mask). It is
        used for method-independent evaluation, where every method is scored on
        the same reference region. When it is omitted, the region comes from
        :meth:`shock_mask`.
        """
        rho = flow[:, 0:1].clamp_min(1e-5)
        u = flow[:, 1:2]
        v = flow[:, 2:3]
        pressure = flow[:, 3:4].clamp_min(1e-5)
        total_energy = pressure / (rho * (self.gamma - 1.0)) + 0.5 * (u.square() + v.square())
        conservative_energy = rho * total_energy

        flux_x = torch.cat(
            [rho * u, rho * u.square() + pressure, rho * u * v, u * (conservative_energy + pressure)],
            dim=1,
        )
        flux_y = torch.cat(
            [rho * v, rho * u * v, rho * v.square() + pressure, v * (conservative_energy + pressure)],
            dim=1,
        )
        dfx_dx, _ = _gradient(flux_x, spacing)
        _, dfy_dy = _gradient(flux_y, spacing)
        residual = dfx_dx + dfy_dy

        if shock is None:
            shock = self.shock_mask(flow, saliency)
        else:
            shock = shock.to(device=flow.device, dtype=flow.dtype)
        continuous = (1.0 - shock).clamp(0.0, 1.0)
        conservation = _masked_mean(residual.square().sum(dim=1, keepdim=True), continuous)

        sdf_x, sdf_y = _gradient(sdf, spacing)
        sdf_mag = torch.sqrt(sdf_x.square() + sdf_y.square() + 1e-12)
        wall_normal_x = sdf_x / sdf_mag
        wall_normal_y = sdf_y / sdf_mag
        no_penetration = (u * wall_normal_x + v * wall_normal_y).square()
        wall = _masked_mean(no_penetration, wall_mask)

        mach = condition_physical[:, 0].reshape(-1, 1, 1, 1)
        alpha = torch.deg2rad(condition_physical[:, 1]).reshape(-1, 1, 1, 1)
        freestream = torch.cat(
            [
                torch.ones_like(mach).expand(-1, -1, flow.shape[-2], flow.shape[-1]),
                (mach * torch.cos(alpha)).expand(-1, -1, flow.shape[-2], flow.shape[-1]),
                (mach * torch.sin(alpha)).expand(-1, -1, flow.shape[-2], flow.shape[-1]),
                torch.ones_like(mach).expand(-1, -1, flow.shape[-2], flow.shape[-1]),
            ],
            dim=1,
        )
        farfield = _masked_mean((flow - freestream).square(), farfield_mask)
        boundary = wall + farfield
        rh = self._rh_energy(flow, shock, spacing)
        total = self.lambda_cons * conservation + self.lambda_bc * boundary + self.lambda_rh * rh
        return {
            "total": total,
            "conservation": conservation,
            "boundary": boundary,
            "wall": wall,
            "farfield": farfield,
            "rh": rh,
            "shock_fraction": shock.mean(),
        }
