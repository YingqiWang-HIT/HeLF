"""Exponential moving average for model parameters."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterator

import torch


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, decay: float = 0.9999) -> None:
        self.decay = float(decay)
        self.shadow: Dict[str, torch.Tensor] = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        self.backup: Dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name not in self.shadow:
                self.shadow[name] = parameter.detach().clone()
            else:
                self.shadow[name].lerp_(parameter.detach(), 1.0 - self.decay)

    def state_dict(self) -> dict:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: dict) -> None:
        self.decay = float(state["decay"])
        self.shadow = {key: value.clone() for key, value in state["shadow"].items()}

    @contextmanager
    def average_parameters(self, model: torch.nn.Module) -> Iterator[None]:
        self.backup = {}
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in self.shadow:
                    self.backup[name] = parameter.detach().clone()
                    parameter.copy_(self.shadow[name])
        try:
            yield
        finally:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in self.backup:
                        parameter.copy_(self.backup[name])
            self.backup = {}
