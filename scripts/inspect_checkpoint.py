#!/usr/bin/env python
"""Print non-tensor metadata from an HeLF checkpoint."""

from __future__ import annotations

import argparse
import json

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    summary = {
        "stage": checkpoint.get("stage"),
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "project": checkpoint.get("config", {}).get("project", {}),
        "model_tensors": len(checkpoint.get("model", {})),
        "has_optimizer": checkpoint.get("optimizer") is not None,
        "has_scheduler": checkpoint.get("scheduler") is not None,
        "has_ema": checkpoint.get("ema") is not None,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
