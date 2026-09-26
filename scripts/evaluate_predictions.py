#!/usr/bin/env python
"""Score externally produced predictions (e.g. baselines) with the HeLF metrics.

Input: an ``.npz`` file with

* ``prediction``: float array [N, 4, H, W] in physical units, ordered rho, u, v, p;
* ``paths`` (recommended): the N sample files the predictions belong to. Without
  it, predictions are matched to the split in sorted file order.

Every method, including HeLF (see ``scripts/evaluate.py --save-predictions``),
should be scored with this script and the same configuration so that PSNR, SSIM,
and physical residuals use identical operators and shock regions.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import _bootstrap  # noqa: F401

from helf import load_config
from helf.data.dataset import NPZFlowDataset
from helf.evaluation import SHOCK_REGIONS, evaluate_fields
from helf.models.physics import PhysicalMechanismEnergy
from helf.utils import resolve_device, to_device


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--override")
    parser.add_argument("--predictions", required=True, help="NPZ file with a 'prediction' array")
    parser.add_argument("--method", help="Method name recorded in the summary")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--device")
    parser.add_argument("--shock-region", choices=SHOCK_REGIONS, default="reference")
    parser.add_argument("--output")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    device = resolve_device(args.device)
    dataset = NPZFlowDataset(config["data"]["root"], args.split, config)
    energy = PhysicalMechanismEnergy(config["model"]["physics"]).to(device)

    with np.load(args.predictions, allow_pickle=False) as archive:
        predictions = np.asarray(archive["prediction"], dtype=np.float32)
        paths = [str(value) for value in archive["paths"]] if "paths" in archive else None
        stored_method = str(archive["method"]) if "method" in archive else None
    if predictions.ndim != 4 or predictions.shape[1] != 4:
        raise ValueError(f"'prediction' must have shape [N, 4, H, W], got {predictions.shape}")

    index_by_path = {str(path): index for index, path in enumerate(dataset.files)}
    index_by_relative = {
        path.relative_to(dataset.root).as_posix(): index for index, path in enumerate(dataset.files)
    }

    def locate(path: str) -> int | None:
        """Match a stored path exactly, or by its location inside the split directory."""
        if path in index_by_path:
            return index_by_path[path]
        normalized = path.replace("\\", "/")
        matches = [
            index
            for relative, index in index_by_relative.items()
            if normalized == relative or normalized.endswith("/" + relative)
        ]
        return matches[0] if len(matches) == 1 else None
    if paths is None:
        if len(predictions) != len(dataset):
            raise ValueError(
                f"{len(predictions)} predictions but {len(dataset)} samples in split "
                f"'{args.split}'; provide 'paths' to match them explicitly"
            )
        order = list(range(len(dataset)))
    else:
        if len(paths) != len(predictions):
            raise ValueError("'paths' and 'prediction' must have the same length")
        located = [locate(path) for path in paths]
        missing = [path for path, index in zip(paths, located) if index is None]
        if missing:
            raise KeyError(f"{len(missing)} prediction paths are not in the split, e.g. {missing[0]}")
        order = [int(index) for index in located]

    totals = defaultdict(float)
    for prediction_np, sample_index in zip(predictions, order):
        sample = dataset[sample_index]
        batch = {
            key: value.unsqueeze(0) if torch.is_tensor(value) else value
            for key, value in sample.items()
        }
        batch = to_device(batch, device)
        prediction = torch.from_numpy(prediction_np).unsqueeze(0).to(device)
        if prediction.shape[-2:] != batch["flow_physical"].shape[-2:]:
            raise ValueError(
                f"Prediction grid {tuple(prediction.shape[-2:])} does not match the "
                f"reference grid {tuple(batch['flow_physical'].shape[-2:])}"
            )
        metrics = evaluate_fields(prediction, batch, energy, shock_region=args.shock_region)
        for key, value in metrics.items():
            totals[key] += value

    count = len(order)
    summary = {key: value / max(count, 1) for key, value in totals.items()}
    summary.update(
        {
            "samples": count,
            "method": args.method or stored_method or Path(args.predictions).stem,
            "predictions": args.predictions,
            "shock_region": args.shock_region,
        }
    )
    text = json.dumps(summary, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
