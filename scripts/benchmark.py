#!/usr/bin/env python
"""Measure end-to-end HeLF inference time with warm-up and synchronization."""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext

import numpy as np

import _bootstrap  # noqa: F401

from helf import HeLF, load_config
from helf.data.dataset import build_dataloader
from helf.models.sampler import LatentFusionSampler
from helf.training.checkpoint import load_checkpoint
from helf.training.ema import ExponentialMovingAverage
from helf.utils import resolve_device, synchronized_time, to_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--device")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--no-ema", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    device = resolve_device(args.device)
    model = HeLF(config).to(device).eval()
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema_decay", 0.9999)))
    checkpoint = load_checkpoint(args.checkpoint, model, ema=ema, strict=False)
    batch = to_device(next(iter(build_dataloader(config, args.split, batch_size=1, shuffle=False))), device)
    sampler = LatentFusionSampler(model, config)
    context = (
        ema.average_parameters(model)
        if checkpoint.get("ema") is not None and not args.no_ema
        else nullcontext()
    )
    times = []
    with context:
        for _ in range(args.warmup):
            sampler.sample(batch)
        for _ in range(args.runs):
            start = synchronized_time(device)
            sampler.sample(batch)
            end = synchronized_time(device)
            times.append(end - start)
    print(json.dumps({
        "runs": args.runs,
        "mean_seconds": float(np.mean(times)),
        "std_seconds": float(np.std(times)),
        "min_seconds": float(np.min(times)),
        "max_seconds": float(np.max(times)),
        "device": str(device),
    }, indent=2))


if __name__ == "__main__":
    main()
