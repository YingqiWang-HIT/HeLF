import torch

from helf.models.diffusion import GaussianDiffusion


def test_diffusion_roundtrip_shapes():
    diffusion = GaussianDiffusion(train_steps=20)
    latent = torch.randn(3, 4, 8, 8)
    timestep = torch.tensor([0, 5, 19])
    noisy, noise = diffusion.q_sample(latent, timestep)
    predicted = diffusion.predict_start_from_noise(noisy, timestep, noise)
    assert predicted.shape == latent.shape
    assert torch.allclose(predicted, latent, atol=1e-4, rtol=1e-4)
