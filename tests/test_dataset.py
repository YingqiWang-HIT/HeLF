from pathlib import Path

from helf import load_config
from helf.data.dataset import NPZFlowDataset
from helf.data.synthetic import generate_dataset


ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_dataset_loading(tmp_path):
    generate_dataset(tmp_path, {"train": 2, "val": 1, "test": 1}, 32, 32, seed=3)
    config = load_config(ROOT / "configs/smoke.yaml")
    config["data"]["root"] = str(tmp_path)
    dataset = NPZFlowDataset(tmp_path, "train", config)
    sample = dataset[0]
    assert sample["flow"].shape == (4, 32, 32)
    assert sample["sdf"].shape == (1, 32, 32)
    assert sample["condition"].shape == (2,)
