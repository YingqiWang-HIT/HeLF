#!/usr/bin/env python
"""Train HeLF Stage A, B, or C."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from helf import HeLF, load_config
from helf.training.engine import run_training
from helf.utils import count_parameters, resolve_device, seed_everything


def build_parser(default_stage: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Base YAML configuration")
    parser.add_argument("--override", help="Optional private/local override YAML")
    parser.add_argument("--stage", choices=["A", "B", "C"], default=default_stage)
    parser.add_argument("--resume", help="Checkpoint used to initialize model weights")
    parser.add_argument("--device", help="Device string such as cuda, cuda:0, or cpu")
    parser.add_argument("--deterministic", action="store_true")
    return parser


def main(default_stage: str | None = None) -> None:
    args = build_parser(default_stage).parse_args()
    if args.stage is None:
        raise SystemExit("--stage is required")
    config = load_config(args.config, args.override)
    seed_everything(int(config["project"].get("seed", 42)), deterministic=args.deterministic)
    device = resolve_device(args.device)
    model = HeLF(config).to(device)
    print(f"Device: {device}")
    print(f"Total parameters: {count_parameters(model, trainable_only=False):,}")
    result = run_training(model, config, args.stage, device, resume=args.resume)
    print(f"Completed Stage {args.stage}. Checkpoints: {result['output_dir']}")


if __name__ == "__main__":
    main()
