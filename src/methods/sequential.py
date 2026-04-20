from __future__ import annotations

from typing import NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainResult(NamedTuple):
    acc_matrix: np.ndarray    # shape (n_tasks, n_tasks)
    epoch_curves: np.ndarray  # shape (n_tasks, epochs_per_task, n_tasks), NaN where eval_task > train_task
    zero_shot_acc: np.ndarray # shape (n_tasks,) — acc on task i right before training on it


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
    return total_loss / len(loader.dataset)  # type: ignore[arg-type]


def eval_accuracy(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0.0


def run_sequential(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    lr: float,
    device: torch.device,
) -> TrainResult:
    """Train sequentially; return (acc_matrix, epoch_curves, zero_shot_acc).

    acc_matrix[t, i]      = accuracy on task i after finishing task t
    epoch_curves[t, e, i] = accuracy on task i after epoch e of task t (NaN where i > t)
    zero_shot_acc[i]      = accuracy on task i right before training on it
    """
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()

    for t, task in enumerate(tasks):
        zero_shot_acc[t] = eval_accuracy(model, task["test"], device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks}", leave=False):
            _train_epoch(model, task["train"], optimizer, criterion, device)
            for i in range(t + 1):
                epoch_curves[t, epoch, i] = eval_accuracy(model, tasks[i]["test"], device)

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)
