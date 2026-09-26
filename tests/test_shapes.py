from pathlib import Path

import torch

from helf import HeLF, load_config


ROOT = Path(__file__).resolve().parents[1]


def test_model_tensor_shapes():
    config = load_config(ROOT / "configs/smoke.yaml")
    model = HeLF(config)
    batch = 2
    flow = torch.randn(batch, 4, 64, 64)
    sdf = torch.randn(batch, 1, 64, 64)
    condition = torch.randn(batch, 2)
    vae = model.vae(flow)
    assert vae["latent"].shape == (batch, 4, 8, 8)
    code = model.condition_code(sdf, condition)
    assert code.shape == (batch, 64)
    timestep = torch.randint(0, model.diffusion.train_steps, (batch,))
    output = model.denoiser(vae["latent"], timestep, code)
    assert output["noise"].shape == vae["latent"].shape
    assert output["saliency"].shape[0:2] == (batch, 1)
