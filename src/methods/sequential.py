from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


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


def _eval_accuracy(
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
) -> tuple[np.ndarray, np.ndarray]:
    """Train sequentially and return (acc_matrix, epoch_curves).

    acc_matrix[t, i]      = accuracy on task i after finishing task t
    epoch_curves[t, e, i] = accuracy on task i after epoch e of task t (NaN where i > t)
    """
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()

    for t, task in enumerate(tasks):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks}", leave=False):
            _train_epoch(model, task["train"], optimizer, criterion, device)
            for i in range(t + 1):
                epoch_curves[t, epoch, i] = _eval_accuracy(model, tasks[i]["test"], device)

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

    return acc_matrix, epoch_curves
