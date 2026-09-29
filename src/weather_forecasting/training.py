from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy

import numpy as np
import torch


@dataclass(frozen=True)
class EpochStats:
    train_loss: float
    val_loss: float


def _run_epoch(model, loader, loss_fn, optimizer=None, device="cpu"):
    train = optimizer is not None
    model.train(train)
    total = 0.0
    count = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train:
            optimizer.zero_grad(set_to_none=True)
        pred = model(x)
        loss = loss_fn(pred, y)
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total += loss.item() * len(x)
        count += len(x)
    return total / max(count, 1)


def fit_model(
    model,
    train_loader,
    val_loader,
    *,
    epochs: int = 50,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 8,
    device: str = "cpu",
):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.SmoothL1Loss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=max(1, patience // 3)
    )

    best_state = deepcopy(model.state_dict())
    best_val = float("inf")
    stale = 0
    history: list[EpochStats] = []

    for _ in range(epochs):
        train_loss = _run_epoch(model, train_loader, loss_fn, optimizer, device)
        with torch.no_grad():
            val_loss = _run_epoch(model, val_loader, loss_fn, None, device)
        scheduler.step(val_loss)
        history.append(EpochStats(train_loss, val_loss))

        if val_loss < best_val:
            best_val = val_loss
            best_state = deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def predict(model, loader, device: str = "cpu"):
    model.eval().to(device)
    preds, targets = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)
