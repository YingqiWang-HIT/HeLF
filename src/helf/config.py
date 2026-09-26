"""YAML configuration loading and recursive override support."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Union

import yaml

Config = Dict[str, Any]


def deep_merge(base: MutableMapping[str, Any], override: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Recursively merge ``override`` into ``base`` and return ``base``."""
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), MutableMapping):
            deep_merge(base[key], value)
        else:
            base[key] = deepcopy(value)
    return base


def _read_yaml(path: Union[str, Path]) -> Config:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML object must be a mapping: {path}")
    return data


def load_config(path: Union[str, Path], override: Optional[Union[str, Path]] = None) -> Config:
    """Load a base YAML configuration and optionally apply a private override."""
    config = _read_yaml(path)
    if override is not None:
        config = dict(deep_merge(config, _read_yaml(override)))
    return config


def get_stage_config(config: Mapping[str, Any], stage: str) -> Mapping[str, Any]:
    stage = stage.lower()
    key = f"stage_{stage}"
    try:
        return config["training"][key]
    except KeyError as exc:
        raise KeyError(f"Missing training configuration for stage {stage.upper()}") from exc
