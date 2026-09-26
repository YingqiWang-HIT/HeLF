"""Tests for the shared evaluation used for HeLF and baseline predictions."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from helf import load_config
from helf.data.dataset import NPZFlowDataset
from helf.data.synthetic import generate_dataset
from helf.evaluation import evaluate_fields, reference_shock_mask
from helf.models.physics import PhysicalMechanismEnergy

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    config = load_config(ROOT / "configs/smoke.yaml")
    config["data"]["root"] = str(tmp_path)
    return config


def test_reference_region_uses_stored_shock_mask(tmp_path):
    generate_dataset(tmp_path, {"train": 1, "val": 1, "test": 2}, 32, 32, seed=1)
    config = _config(tmp_path)
    sample = NPZFlowDataset(tmp_path, "test", config)[0]
    target = sample["flow_physical"][None]
    stored = sample["shock_mask"][None]
    energy = PhysicalMechanismEnergy(config["model"]["physics"])
    mask = reference_shock_mask(target, energy, stored)
    if float(stored.sum()) > 0:
        assert torch.equal(mask, stored)
    empty = torch.zeros_like(stored)
    fallback = reference_shock_mask(target, energy, empty)
    assert float(fallback.sum()) > 0


def test_perfect_prediction_scores_high(tmp_path):
    generate_dataset(tmp_path, {"train": 1, "val": 1, "test": 2}, 32, 32, seed=2)
    config = _config(tmp_path)
    sample = NPZFlowDataset(tmp_path, "test", config)[0]
    batch = {key: value[None] if torch.is_tensor(value) else value for key, value in sample.items()}
    energy = PhysicalMechanismEnergy(config["model"]["physics"])
    metrics = evaluate_fields(batch["flow_physical"].clone(), batch, energy)
    assert metrics["psnr"] > 60
    assert metrics["ssim"] > 0.99
    assert "physical_rh" in metrics


def test_evaluate_predictions_script(tmp_path):
    data_root = tmp_path / "data"
    generate_dataset(data_root, {"train": 1, "val": 1, "test": 3}, 32, 32, seed=4)
    config = _config(data_root)
    config_path = tmp_path / "config.yaml"
    import yaml

    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    dataset = NPZFlowDataset(data_root, "test", config)
    targets = np.stack([dataset[i]["flow_physical"].numpy() for i in range(len(dataset))])
    paths = [str(path) for path in dataset.files][::-1]
    predictions = targets[::-1].copy()
    prediction_path = tmp_path / "baseline.npz"
    np.savez_compressed(prediction_path, prediction=predictions, paths=np.asarray(paths))
    output = tmp_path / "summary.json"
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/evaluate_predictions.py"),
            "--config",
            str(config_path),
            "--predictions",
            str(prediction_path),
            "--method",
            "GrFormer",
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
    )
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["samples"] == 3
    assert summary["method"] == "GrFormer"
    assert summary["psnr"] > 60
