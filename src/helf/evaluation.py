"""Shared evaluation of reconstructed flow fields.

The same function scores HeLF and every baseline, so that reconstruction
metrics and physical residuals are computed with identical operators,
normalization, and shock region.
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional

import torch

from helf.metrics import metric_dict
from helf.models.physics import PhysicalMechanismEnergy, _top_fraction_mask, gradient_indicator

SHOCK_REGIONS = ("reference", "prediction")


def reference_shock_mask(
    target: torch.Tensor,
    energy: PhysicalMechanismEnergy,
    data_shock_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Method-independent shock region taken from the reference field.

    A non-empty ``shock_mask`` stored with the sample is used as is. Otherwise the
    region is the top ``1 - shock_quantile`` fraction of the gradient indicator of
    the reference flow, dilated by ``shock_dilation`` as configured.
    """
    mask = _top_fraction_mask(gradient_indicator(target), energy.shock_quantile)
    if energy.shock_dilation > 0:
        radius = energy.shock_dilation
        mask = torch.nn.functional.max_pool2d(
            mask, kernel_size=2 * radius + 1, stride=1, padding=radius
        )
    if data_shock_mask is not None:
        stored = data_shock_mask.to(device=target.device, dtype=target.dtype)
        has_stored = stored.flatten(1).sum(dim=1) > 0
        mask = torch.where(has_stored.reshape(-1, 1, 1, 1), stored, mask)
    return mask.to(target.dtype)


def evaluate_fields(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    energy: PhysicalMechanismEnergy,
    shock_region: str = "reference",
    saliency: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Score one batch of predictions in physical units.

    ``shock_region="reference"`` evaluates every method on the region derived from
    the reference field, which is the setting to use when comparing methods.
    ``shock_region="prediction"`` uses the region the method itself would select
    (the predicted saliency for HeLF), which is how the review-stage evaluation
    script behaved before this option was added.
    """
    if shock_region not in SHOCK_REGIONS:
        raise ValueError(f"shock_region must be one of {SHOCK_REGIONS}, got {shock_region!r}")
    target = batch["flow_physical"]
    data_range = float((target.max() - target.min()).clamp_min(1e-6).item())
    metrics = metric_dict(prediction, target, data_range=data_range)
    shock = (
        reference_shock_mask(target, energy, batch.get("shock_mask"))
        if shock_region == "reference"
        else None
    )
    with torch.no_grad():
        physical = energy(
            prediction,
            batch["sdf"],
            batch["condition_physical"],
            batch["wall_mask"],
            batch["farfield_mask"],
            batch["spacing"],
            saliency,
            shock=shock,
        )
    metrics.update({f"physical_{key}": float(value.detach().cpu()) for key, value in physical.items()})
    return metrics
