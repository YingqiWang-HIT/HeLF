#!/usr/bin/env python
"""Generate a small analytic dataset for functional testing."""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401

from helf.data.synthetic import generate_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/synthetic")
    parser.add_argument("--train", type=int, default=32)
    parser.add_argument("--val", type=int, default=8)
    parser.add_argument("--test", type=int, default=8)
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    generate_dataset(
        args.output,
        {"train": args.train, "val": args.val, "test": args.test},
        args.height,
        args.width,
        args.seed,
    )
    print(f"Synthetic dataset written to {args.output}")
    print("This dataset is for software validation only and does not reproduce paper results.")


if __name__ == "__main__":
    main()
