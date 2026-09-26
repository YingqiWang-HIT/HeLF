"""Tests for the saliency gate, partition ablations, and fusion diagnostics."""

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from helf import HeLF, load_config
from helf.models.physics import PhysicalMechanismEnergy, gradient_indicator
from helf.models.sampler import LatentFusionSampler
from helf.training.losses import saliency_target

ROOT = Path(__file__).resolve().parents[1]


def _flow(batch=2, size=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch, 4, size, size, generator=generator)
    return torch.stack([raw[:, 0].abs() + 0.5, raw[:, 1], raw[:, 2], raw[:, 3].abs() + 0.5], dim=1)


def _masks(flow):
    sdf = torch.randn(flow.shape[0], 1, *flow.shape[-2:])
    wall = (sdf.abs() < 0.1).float()
    far = torch.zeros_like(wall)
    far[..., :2, :] = 1
    far[..., -2:, :] = 1
    return sdf, wall, far


def _legacy_shock_mask(flow, saliency, quantile):
    """Shock selection of the review-stage release, kept to check compatibility."""
    score = F.interpolate(saliency, size=flow.shape[-2:], mode="bilinear", align_corners=False)
    flat = score.flatten(1)
    selected = max(1, math.ceil((1.0 - quantile) * flat.shape[1]))
    indices = flat.topk(selected, dim=1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, indices, 1.0)
    return mask.reshape_as(score).to(flow.dtype)


def _legacy_saliency_target(flow):
    def grad(field):
        px = F.pad(field, (1, 1, 0, 0), mode="replicate")
        py = F.pad(field, (0, 0, 1, 1), mode="replicate")
        return 0.5 * (px[..., 2:] - px[..., :-2]), 0.5 * (py[..., 2:, :] - py[..., :-2, :])

    rx, ry = grad(flow[:, 0:1])
    px, py = grad(flow[:, 3:4])
    ux, uy = grad(flow[:, 1:3])
    indicator = (
        torch.sqrt(rx.square() + ry.square() + 1e-8)
        + torch.sqrt(px.square() + py.square() + 1e-8)
        + torch.sqrt(ux.square() + uy.square() + 1e-8).sum(dim=1, keepdim=True)
    ) / 3.0
    mean = indicator.mean(dim=(2, 3), keepdim=True)
    std = indicator.std(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return torch.sigmoid((indicator - mean) / std)


def test_default_saliency_gate_matches_review_release():
    flow = _flow()
    saliency = torch.rand(2, 1, 8, 8)
    energy = PhysicalMechanismEnergy({"shock_quantile": 0.9})
    assert torch.equal(energy.shock_mask(flow, saliency), _legacy_shock_mask(flow, saliency, 0.9))


def test_saliency_target_is_unchanged():
    flow = _flow(seed=3)
    assert torch.equal(saliency_target(flow), _legacy_saliency_target(flow))
    assert gradient_indicator(flow).shape == (2, 1, 32, 32)


def test_partition_none_applies_conservation_everywhere():
    flow = _flow()
    sdf, wall, far = _masks(flow)
    condition = torch.tensor([[0.9, 5.0], [1.1, 10.0]])
    energy = PhysicalMechanismEnergy({"shock_partition": "none"})
    result = energy(flow, sdf, condition, wall, far, torch.ones(2, 2), torch.rand(2, 1, 8, 8))
    assert float(result["shock_fraction"]) == 0.0
    assert float(result["rh"]) == 0.0
    assert torch.isfinite(result["total"])


def test_partition_gradient_ignores_saliency():
    flow = _flow()
    energy = PhysicalMechanismEnergy({"shock_partition": "gradient", "shock_quantile": 0.9})
    first = energy.shock_mask(flow, torch.rand(2, 1, 8, 8))
    second = energy.shock_mask(flow, torch.rand(2, 1, 8, 8))
    assert torch.equal(first, second)
    assert abs(float(first.mean()) - 0.1) < 0.01


def test_dilation_grows_the_shock_region():
    flow = _flow()
    saliency = torch.rand(2, 1, 8, 8)
    plain = PhysicalMechanismEnergy({"shock_quantile": 0.95}).shock_mask(flow, saliency)
    dilated = PhysicalMechanismEnergy({"shock_quantile": 0.95, "shock_dilation": 1}).shock_mask(
        flow, saliency
    )
    assert torch.all(dilated >= plain)
    assert float(dilated.sum()) > float(plain.sum())


def test_invalid_partition_is_rejected():
    with pytest.raises(ValueError):
        PhysicalMechanismEnergy({"shock_partition": "threshold"})


def test_fixed_shock_region_overrides_the_gate():
    flow = _flow()
    sdf, wall, far = _masks(flow)
    condition = torch.tensor([[0.9, 5.0], [1.1, 10.0]])
    shock = torch.zeros(2, 1, 32, 32)
    shock[..., 10:12, :] = 1
    result = PhysicalMechanismEnergy({})(
        flow, sdf, condition, wall, far, torch.ones(2, 2), None, shock=shock
    )
    assert abs(float(result["shock_fraction"]) - float(shock.mean())) < 1e-6


def test_fusion_diagnostics_follow_the_reliability_gate():
    torch.manual_seed(0)
    config = load_config(ROOT / "configs/smoke.yaml")
    config["model"]["diffusion"]["sample_steps"] = 3
    config["model"]["physics"]["correction_count"] = 2
    model = HeLF(config).eval()
    sdf = torch.ones(1, 1, 64, 64)
    far = torch.zeros_like(sdf)
    far[:, :, 0] = 1
    batch = {
        "sdf": sdf,
        "condition": torch.zeros(1, 2),
        "condition_physical": torch.tensor([[0.9, 5.0]]),
        "wall_mask": torch.zeros_like(sdf),
        "farfield_mask": far,
        "spacing": torch.ones(1, 2),
    }
    result = LatentFusionSampler(model, config).sample(batch)
    diagnostics = result["fusion_diagnostics"]
    assert len(diagnostics) == 2
    omega_max = float(config["model"]["physics"]["omega_max"])
    for record in diagnostics:
        for key in ("reliability", "need", "omega", "temperature", "energy", "shock_fraction"):
            assert key in record
        assert record["omega"] == pytest.approx(
            omega_max * record["reliability"] * record["need"], rel=1e-5
        )
    assert torch.isfinite(result["flow"]).all()
