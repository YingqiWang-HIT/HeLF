"""NPZ dataset loader for airfoil flow reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .sdf import farfield_mask, wall_mask_from_sdf


class NPZFlowDataset(Dataset):
    """Load one geometry-condition field from each compressed NPZ file."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        config: Mapping[str, Any],
        require_flow: bool = True,
    ) -> None:
        self.root = Path(root) / split
        self.split = split
        self.require_flow = require_flow
        if not self.root.exists():
            raise FileNotFoundError(
                f"Split directory does not exist: {self.root}. "
                "Generate synthetic data or configure the official dataset root."
            )
        self.files = sorted(self.root.rglob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {self.root}")

        data_cfg = config["data"]
        self.flow_mean = torch.tensor(data_cfg["flow_mean"], dtype=torch.float32)[:, None, None]
        self.flow_std = torch.tensor(data_cfg["flow_std"], dtype=torch.float32)[:, None, None]
        self.cond_mean = torch.tensor(data_cfg["condition_mean"], dtype=torch.float32)
        self.cond_std = torch.tensor(data_cfg["condition_std"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as sample:
            if "sdf" not in sample or "condition" not in sample:
                raise KeyError(f"{path} must contain 'sdf' and 'condition'")
            sdf_np = np.asarray(sample["sdf"], dtype=np.float32)
            condition_np = np.asarray(sample["condition"], dtype=np.float32)
            if sdf_np.ndim != 2 or condition_np.shape != (2,):
                raise ValueError(f"Invalid shapes in {path}: sdf={sdf_np.shape}, condition={condition_np.shape}")

            wall_np = (
                np.asarray(sample["wall_mask"], dtype=np.float32)
                if "wall_mask" in sample
                else wall_mask_from_sdf(sdf_np)
            )
            far_np = (
                np.asarray(sample["farfield_mask"], dtype=np.float32)
                if "farfield_mask" in sample
                else farfield_mask(*sdf_np.shape)
            )
            shock_np = (
                np.asarray(sample["shock_mask"], dtype=np.float32)
                if "shock_mask" in sample
                else np.zeros_like(sdf_np, dtype=np.float32)
            )
            spacing_np = (
                np.asarray(sample["spacing"], dtype=np.float32)
                if "spacing" in sample
                else np.ones(2, dtype=np.float32)
            )
            result: Dict[str, Any] = {
                "sdf": torch.from_numpy(sdf_np)[None],
                "condition": (torch.from_numpy(condition_np) - self.cond_mean) / self.cond_std.clamp_min(1e-8),
                "condition_physical": torch.from_numpy(condition_np),
                "wall_mask": torch.from_numpy(wall_np)[None],
                "farfield_mask": torch.from_numpy(far_np)[None],
                "shock_mask": torch.from_numpy(shock_np)[None],
                "spacing": torch.from_numpy(spacing_np),
                "path": str(path),
            }
            if "flow" in sample:
                flow_np = np.asarray(sample["flow"], dtype=np.float32)
                if flow_np.ndim != 3 or flow_np.shape[0] != 4:
                    raise ValueError(f"Invalid flow shape in {path}: {flow_np.shape}")
                flow = torch.from_numpy(flow_np)
                result["flow_physical"] = flow
                result["flow"] = (flow - self.flow_mean) / self.flow_std.clamp_min(1e-8)
            elif self.require_flow:
                raise KeyError(f"{path} does not contain required key 'flow'")
        return result

    def denormalize_flow(self, flow: torch.Tensor) -> torch.Tensor:
        mean = self.flow_mean.to(flow.device)
        std = self.flow_std.to(flow.device)
        return flow * std + mean


def build_dataloader(
    config: Mapping[str, Any],
    split: str,
    batch_size: Optional[int] = None,
    shuffle: Optional[bool] = None,
    require_flow: bool = True,
) -> DataLoader:
    dataset = NPZFlowDataset(config["data"]["root"], split, config, require_flow=require_flow)
    data_cfg = config["data"]
    if batch_size is None:
        batch_size = 1
    if shuffle is None:
        shuffle = split == "train"
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=int(data_cfg.get("num_workers", 0)),
        pin_memory=bool(data_cfg.get("pin_memory", False)),
        drop_last=split == "train" and len(dataset) >= batch_size,
    )
