"""Three-stage training engine."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any, Dict, Mapping, Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from helf.config import get_stage_config
from helf.data.dataset import build_dataloader
from helf.training.checkpoint import load_checkpoint, save_checkpoint
from helf.training.ema import ExponentialMovingAverage
from helf.training.losses import compute_stage_loss
from helf.utils import ensure_dir, to_device


def configure_stage(model: torch.nn.Module, stage: str) -> None:
    stage = stage.upper()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if stage == "A":
        modules = [model.vae]
    elif stage == "B":
        modules = [model.geometry_encoder, model.denoiser]
    elif stage == "C":
        modules = [model.vae, model.geometry_encoder, model.denoiser]
    else:
        raise ValueError("stage must be one of A, B, or C")
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)


def _aggregate(records: list[Mapping[str, float]]) -> Dict[str, float]:
    if not records:
        return {}
    totals: Dict[str, float] = defaultdict(float)
    counts: Dict[str, int] = defaultdict(int)
    for record in records:
        for key, value in record.items():
            totals[key] += float(value)
            counts[key] += 1
    return {key: totals[key] / counts[key] for key in totals}


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    stage: str,
    epoch: int,
    config: Mapping[str, Any],
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer],
    scaler: Optional[torch.amp.GradScaler],
    ema: Optional[ExponentialMovingAverage],
    train: bool,
    max_batches: Optional[int],
) -> Dict[str, float]:
    model.train(train)
    records = []
    progress = tqdm(loader, desc=f"{stage} {'train' if train else 'val'} {epoch + 1}", leave=False)
    for batch_index, batch in enumerate(progress):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = to_device(batch, device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        amp_enabled = bool(config["training"].get("amp", False)) and device.type == "cuda"
        # Stage C includes higher-order physical energy calculations. Keeping it in fp32
        # is more stable and avoids unsafe autocast combinations.
        amp_enabled = amp_enabled and stage.upper() != "C"
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=amp_enabled):
                loss, metrics = compute_stage_loss(model, batch, stage, epoch, config)
            if train and optimizer is not None:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(config["training"].get("grad_clip", 1.0)),
                    )
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model.parameters() if p.requires_grad],
                        float(config["training"].get("grad_clip", 1.0)),
                    )
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
        records.append(metrics)
        progress.set_postfix(loss=f"{metrics['loss']:.4f}")
    return _aggregate(records)


def run_training(
    model: torch.nn.Module,
    config: Mapping[str, Any],
    stage: str,
    device: torch.device,
    resume: Optional[str] = None,
) -> Dict[str, Any]:
    stage = stage.upper()
    configure_stage(model, stage)
    stage_cfg = get_stage_config(config, stage)
    batch_size = int(stage_cfg["batch_size"])
    train_loader = build_dataloader(config, "train", batch_size=batch_size, shuffle=True)
    val_loader = build_dataloader(config, "val", batch_size=batch_size, shuffle=False)

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_cfg = config["training"]["optimizer"]
    optimizer = AdamW(
        parameters,
        lr=float(stage_cfg["learning_rate"]),
        betas=tuple(float(v) for v in optimizer_cfg.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_cfg.get("weight_decay", 1e-4)),
    )
    epochs = int(stage_cfg["epochs"])
    scheduler = CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    ema = ExponentialMovingAverage(model, decay=float(config["training"].get("ema_decay", 0.9999)))
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and bool(config["training"].get("amp", False)))

    start_epoch = 0
    if resume:
        checkpoint = load_checkpoint(resume, model, ema=ema, strict=False, load_training_state=False)
        print(f"Loaded model weights from {resume} (source stage: {checkpoint.get('stage', 'unknown')})")

    output_dir = ensure_dir(config["project"]["output_dir"])
    history = []
    best_val = float("inf")
    max_train = config["training"].get("max_train_batches")
    max_val = config["training"].get("max_val_batches")

    for epoch in range(start_epoch, epochs):
        train_metrics = _run_epoch(
            model,
            train_loader,
            stage,
            epoch,
            config,
            device,
            optimizer,
            scaler,
            ema,
            train=True,
            max_batches=max_train,
        )
        with ema.average_parameters(model):
            val_metrics = _run_epoch(
                model,
                val_loader,
                stage,
                epoch,
                config,
                device,
                optimizer=None,
                scaler=None,
                ema=None,
                train=False,
                max_batches=max_val,
            )
        scheduler.step()
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        print(json.dumps(record, indent=2))

        save_checkpoint(
            output_dir / f"stage_{stage.lower()}_last.pt",
            model,
            optimizer,
            scheduler,
            ema,
            epoch,
            stage,
            config,
            val_metrics,
        )
        if val_metrics.get("loss", float("inf")) < best_val:
            best_val = val_metrics["loss"]
            save_checkpoint(
                output_dir / f"stage_{stage.lower()}_best.pt",
                model,
                optimizer,
                scheduler,
                ema,
                epoch,
                stage,
                config,
                val_metrics,
            )
        save_every = int(config["training"].get("save_every", 0))
        if save_every > 0 and (epoch + 1) % save_every == 0:
            save_checkpoint(
                output_dir / f"stage_{stage.lower()}_epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scheduler,
                ema,
                epoch,
                stage,
                config,
                val_metrics,
            )
    return {"history": history, "best_val": best_val, "output_dir": str(output_dir)}
