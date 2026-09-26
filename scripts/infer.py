#!/usr/bin/env python
"""Reconstruct one flow field from an NPZ geometry-condition sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from helf import HeLF, load_config
from helf.data.sdf import farfield_mask, wall_mask_from_sdf
from helf.models.sampler import LatentFusionSampler
from helf.training.checkpoint import load_checkpoint
from helf.training.ema import ExponentialMovingAverage
from helf.utils import resolve_device, seed_everything


def load_sample(path: str, config: dict, device: torch.device) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as sample:
        sdf = np.asarray(sample["sdf"], dtype=np.float32)
        condition_physical = np.asarray(sample["condition"], dtype=np.float32)
        wall = np.asarray(sample["wall_mask"], dtype=np.float32) if "wall_mask" in sample else wall_mask_from_sdf(sdf)
        far = np.asarray(sample["farfield_mask"], dtype=np.float32) if "farfield_mask" in sample else farfield_mask(*sdf.shape)
        spacing = np.asarray(sample["spacing"], dtype=np.float32) if "spacing" in sample else np.ones(2, dtype=np.float32)
    cond_mean = np.asarray(config["data"]["condition_mean"], dtype=np.float32)
    cond_std = np.asarray(config["data"]["condition_std"], dtype=np.float32)
    condition = (condition_physical - cond_mean) / np.maximum(cond_std, 1e-8)
    return {
        "sdf": torch.from_numpy(sdf)[None, None].to(device),
        "condition": torch.from_numpy(condition)[None].to(device),
        "condition_physical": torch.from_numpy(condition_physical)[None].to(device),
        "wall_mask": torch.from_numpy(wall)[None, None].to(device),
        "farfield_mask": torch.from_numpy(far)[None, None].to(device),
        "spacing": torch.from_numpy(spacing)[None].to(device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    seed_everything(args.seed if args.seed is not None else int(config["project"].get("seed", 42)))
    device = resolve_device(args.device)
    model = HeLF(config).to(device)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema_decay", 0.9999)))
    checkpoint = load_checkpoint(args.checkpoint, model, ema=ema, strict=False)
    model.eval()
    batch = load_sample(args.input, config, device)
    sampler = LatentFusionSampler(model, config)

    context = ema.average_parameters(model) if checkpoint.get("ema") is not None and not args.no_ema else _NullContext()
    with context:
        result = sampler.sample(batch, return_intermediates=bool(config["sampling"].get("save_intermediates", False)))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        flow=result["flow"][0].detach().cpu().numpy(),
        flow_normalized=result["flow_normalized"][0].detach().cpu().numpy(),
        saliency=result["saliency"][0].detach().cpu().numpy() if result["saliency"] is not None else np.array([]),
        fusion_diagnostics_json=np.asarray(json.dumps(result["fusion_diagnostics"])),
    )
    print(f"Prediction written to {output}")


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


if __name__ == "__main__":
    main()
