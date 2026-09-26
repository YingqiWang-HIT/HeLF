#!/usr/bin/env python
"""Validate NPZ samples, shapes, finite values, and optional geometry leakage."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def scalar_string(value: np.ndarray) -> str:
    if value.shape == ():
        return str(value.item())
    return str(value.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    root = Path(args.root)
    report = {"root": str(root), "splits": {}, "errors": [], "warnings": []}
    geometry_splits: dict[str, set[str]] = defaultdict(set)

    for split in ("train", "val", "test"):
        files = sorted((root / split).rglob("*.npz")) if (root / split).exists() else []
        split_info = {"files": len(files), "resolutions": {}, "conditions": []}
        conditions = set()
        resolutions = defaultdict(int)
        for path in files:
            try:
                with np.load(path, allow_pickle=False) as sample:
                    for key in ("sdf", "flow", "condition"):
                        if key not in sample:
                            report["errors"].append(f"{path}: missing key {key}")
                    if not all(key in sample for key in ("sdf", "flow", "condition")):
                        continue
                    sdf = np.asarray(sample["sdf"])
                    flow = np.asarray(sample["flow"])
                    condition = np.asarray(sample["condition"])
                    if sdf.ndim != 2:
                        report["errors"].append(f"{path}: sdf shape {sdf.shape}")
                    if flow.ndim != 3 or flow.shape[0] != 4 or flow.shape[1:] != sdf.shape:
                        report["errors"].append(f"{path}: flow shape {flow.shape}, sdf {sdf.shape}")
                    if condition.shape != (2,):
                        report["errors"].append(f"{path}: condition shape {condition.shape}")
                    if not np.isfinite(sdf).all() or not np.isfinite(flow).all():
                        report["errors"].append(f"{path}: non-finite values")
                    resolutions[str(tuple(sdf.shape))] += 1
                    conditions.add(tuple(float(v) for v in condition.tolist()))
                    if "geometry_id" in sample:
                        geometry_splits[scalar_string(np.asarray(sample["geometry_id"]))].add(split)
            except Exception as exc:
                report["errors"].append(f"{path}: {type(exc).__name__}: {exc}")
        split_info["resolutions"] = dict(resolutions)
        split_info["conditions"] = sorted(conditions)
        report["splits"][split] = split_info

    leakage = {geometry: sorted(splits) for geometry, splits in geometry_splits.items() if len(splits) > 1}
    if leakage:
        report["errors"].append(f"Geometry leakage detected for {len(leakage)} identifiers")
        report["geometry_leakage"] = leakage
    elif not geometry_splits:
        report["warnings"].append("No geometry_id metadata found; cross-split leakage could not be checked")

    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    raise SystemExit(1 if report["errors"] else 0)


if __name__ == "__main__":
    main()
