from pathlib import Path

import torch

from helf import HeLF, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_physical_energy_is_differentiable():
    config = load_config(ROOT / "configs/smoke.yaml")
    model = HeLF(config)
    flow = torch.randn(2, 4, 32, 32, requires_grad=True)
    flow = torch.stack(
        [flow[:, 0].abs() + 0.5, flow[:, 1], flow[:, 2], flow[:, 3].abs() + 0.5], dim=1
    )
    sdf = torch.randn(2, 1, 32, 32)
    condition = torch.tensor([[0.9, 5.0], [1.1, 10.0]])
    wall = (sdf.abs() < 0.1).float()
    far = torch.zeros_like(wall)
    far[:, :, :2] = 1
    far[:, :, -2:] = 1
    far[:, :, :, :2] = 1
    far[:, :, :, -2:] = 1
    spacing = torch.ones(2, 2)
    result = model.physical_energy(flow, sdf, condition, wall, far, spacing)
    assert torch.isfinite(result["total"])
    result["total"].backward()
