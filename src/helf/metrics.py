"""Reconstruction metrics used by HeLF."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def _flatten_per_sample(x: torch.Tensor) -> torch.Tensor:
    return x.reshape(x.shape[0], -1)


def mse_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (_flatten_per_sample(pred - target) ** 2).mean(dim=1)


def mae_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return _flatten_per_sample((pred - target).abs()).mean(dim=1)


def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> torch.Tensor:
    mse = mse_per_sample(pred, target).clamp_min(1e-12)
    return 10.0 * torch.log10(torch.as_tensor(data_range**2, device=pred.device) / mse)


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 7,
) -> torch.Tensor:
    """A compact differentiable SSIM implementation returning one value per sample."""
    padding = window_size // 2
    mu_x = F.avg_pool2d(pred, window_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(target, window_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(pred * pred, window_size, 1, padding) - mu_x.square()
    sigma_y = F.avg_pool2d(target * target, window_size, 1, padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(pred * target, window_size, 1, padding) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    score = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2)
    ).clamp_min(1e-12)
    return score.reshape(score.shape[0], -1).mean(dim=1)


def metric_dict(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> Dict[str, float]:
    return {
        "mae": mae_per_sample(pred, target).mean().item(),
        "mse": mse_per_sample(pred, target).mean().item(),
        "psnr": psnr(pred, target, data_range).mean().item(),
        "ssim": ssim(pred, target, data_range).mean().item(),
    }
