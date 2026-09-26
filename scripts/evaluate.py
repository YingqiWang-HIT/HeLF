#!/usr/bin/env python
"""Evaluate a HeLF checkpoint on an NPZ dataset split.

Physical residuals are computed on the reference shock region by default, so the
numbers are directly comparable with baselines scored by
``scripts/evaluate_predictions.py``. Use ``--shock-region prediction`` to score
the region selected by HeLF's own saliency map (the review-stage behavior).
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import numpy as np
from tqdm import tqdm

import _bootstrap  # noqa: F401

from helf import HeLF, load_config
from helf.data.dataset import build_dataloader
from helf.evaluation import SHOCK_REGIONS, evaluate_fields
from helf.models.sampler import LatentFusionSampler
from helf.training.checkpoint import load_checkpoint
from helf.training.ema import ExponentialMovingAverage
from helf.utils import resolve_device, seed_everything, to_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output")
    parser.add_argument("--no-ema", action="store_true")
    parser.add_argument(
        "--shock-region",
        choices=SHOCK_REGIONS,
        default="reference",
        help="Region used for shock-dependent residuals (default: reference field)",
    )
    parser.add_argument(
        "--save-predictions",
        help="Optional .npz file for the predicted fields, in the format read by "
        "scripts/evaluate_predictions.py",
    )
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    seed_everything(int(config["project"].get("seed", 42)))
    device = resolve_device(args.device)
    model = HeLF(config).to(device)
    ema = ExponentialMovingAverage(model, float(config["training"].get("ema_decay", 0.9999)))
    checkpoint = load_checkpoint(args.checkpoint, model, ema=ema, strict=False)
    loader = build_dataloader(config, args.split, batch_size=1, shuffle=False)
    sampler = LatentFusionSampler(model, config)
    totals = defaultdict(float)
    count = 0
    saved_predictions = []
    saved_paths = []
    model.eval()

    context = (
        ema.average_parameters(model)
        if checkpoint.get("ema") is not None and not args.no_ema
        else nullcontext()
    )
    with context:
        for batch in tqdm(loader, desc=f"Evaluating {args.split}"):
            if args.max_samples is not None and count >= args.max_samples:
                break
            batch = to_device(batch, device)
            result = sampler.sample(batch)
            prediction = result["flow"]
            metrics = evaluate_fields(
                prediction,
                batch,
                model.physical_energy,
                shock_region=args.shock_region,
                saliency=result["saliency"],
            )
            if args.save_predictions:
                saved_predictions.append(prediction.detach().cpu().numpy())
                saved_paths.extend(batch["path"])
            for key, value in metrics.items():
                totals[key] += value
            count += 1

    summary = {key: value / max(count, 1) for key, value in totals.items()}
    summary["samples"] = count
    summary["checkpoint"] = args.checkpoint
    summary["shock_region"] = args.shock_region
    if args.save_predictions and saved_predictions:
        output_path = Path(args.save_predictions)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            output_path,
            prediction=np.concatenate(saved_predictions, axis=0).astype(np.float32),
            paths=np.asarray(saved_paths),
            method=np.asarray("HeLF"),
        )
    print(json.dumps(summary, indent=2))
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
