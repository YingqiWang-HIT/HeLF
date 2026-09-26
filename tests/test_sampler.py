from pathlib import Path

import torch

from helf import HeLF, load_config
from helf.models.sampler import LatentFusionSampler


ROOT = Path(__file__).resolve().parents[1]


def test_sampler_without_fusion_steps():
    config = load_config(ROOT / "configs/smoke.yaml")
    config["model"]["diffusion"]["sample_steps"] = 2
    config["model"]["physics"]["correction_count"] = 0
    model = HeLF(config).eval()
    sdf = torch.ones(1, 1, 64, 64)
    batch = {
        "sdf": sdf,
        "condition": torch.zeros(1, 2),
        "condition_physical": torch.tensor([[0.9, 5.0]]),
        "wall_mask": torch.zeros_like(sdf),
        "farfield_mask": torch.zeros_like(sdf),
        "spacing": torch.ones(1, 2),
    }
    batch["farfield_mask"][:, :, 0] = 1
    sampler = LatentFusionSampler(model, config)
    result = sampler.sample(batch)
    assert result["flow"].shape == (1, 4, 64, 64)
    assert torch.isfinite(result["flow"]).all()
