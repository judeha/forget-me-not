"""Training loops for task-incremental (per-task head) continual learning.

All functions accept the same task dicts as single-head methods.
Tasks must have 'n_classes' and 'label_map' fields (added by data loaders).
Models must expose add_task_head or add_task, task_heads, etc.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.methods.sequential import TrainResult, _remap, eval_accuracy
from src.methods.ewc import EWC
from src.methods.overlap import overlap_loss, budget_loss, make_rho_schedule


def _add_head(model: nn.Module, task_id: int, n_classes: int, device: torch.device) -> None:
    if hasattr(model, "add_task"):
        model.add_task(task_id, n_classes, device)
    else:
        model.add_task_head(task_id, n_classes, device)


def _eval_th(
    model: nn.Module,
    tasks: list[dict],
    task_idx: int,
    device: torch.device,
) -> float:
    t = tasks[task_idx]
    return eval_accuracy(
        model, t["test"], device,
        task_id=task_idx,
        label_map=t.get("label_map"),
    )


def run_multihead_sequential(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    lr: float,
    device: torch.device,
) -> TrainResult:
    n_tasks = len(tasks)
    acc_matrix  = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)
    criterion = nn.CrossEntropyLoss()

    for t, task in enumerate(tasks):
        _add_head(model, t, task["n_classes"], device)
        model._current_task = t
        zero_shot_acc[t] = _eval_th(model, tasks, t, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lm = task.get("label_map")

        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks} [MH-Seq]", leave=False):
            model.train()
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                criterion(model(x, t), _remap(y, lm)).backward()
                optimizer.step()
            for i in range(t + 1):
                epoch_curves[t, epoch, i] = _eval_th(model, tasks, i, device)

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)


def run_multihead_ewc(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    lr: float,
    lambda_ewc: float,
    n_fisher_batches: int,
    device: torch.device,
) -> TrainResult:
    n_tasks = len(tasks)
    acc_matrix  = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)
    criterion = nn.CrossEntropyLoss()
    ewc = EWC(lambda_ewc=lambda_ewc, n_fisher_batches=n_fisher_batches)

    for t, task in enumerate(tasks):
        _add_head(model, t, task["n_classes"], device)
        model._current_task = t
        zero_shot_acc[t] = _eval_th(model, tasks, t, device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lm = task.get("label_map")

        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks} [MH-EWC]", leave=False):
            model.train()
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                (criterion(model(x, t), _remap(y, lm)) + ewc.penalty(model)).backward()
                optimizer.step()
            for i in range(t + 1):
                epoch_curves[t, epoch, i] = _eval_th(model, tasks, i, device)

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

        ewc.consolidate(model, task["train"], device)

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)


def run_multihead_masked_overlap(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    warmup_epochs: int,
    lr: float,
    lambda_overlap: float,
    lambda_budget: float,
    rho_sched: list[float],
    kappa: float,
    use_gradient_masking: bool,
    device: torch.device,
) -> TrainResult:
    """Overlap + budget regularization with optional gradient masking.

    Gradient masking zeros backbone gradients for neurons owned by past tasks
    after each optimizer step, preventing weight-level overwriting.
    """
    n_tasks = len(tasks)
    acc_matrix  = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)
    criterion = nn.CrossEntropyLoss()

    for t, task in enumerate(tasks):
        model.add_task(t, task["n_classes"], device)
        model._current_task = t
        zero_shot_acc[t] = _eval_th(model, tasks, t, device)

        opt_params = model.backbone_params() + model.task_params(t)
        optimizer = torch.optim.Adam(opt_params, lr=lr)
        lm = task.get("label_map")

        for epoch in tqdm(range(epochs_per_task),
                          desc=f"Task {t+1}/{n_tasks} [MH-{'GM' if use_gradient_masking else 'Overlap'}]",
                          leave=False):
            model.train()
            use_ov = (epoch >= warmup_epochs) and (t > 0)
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                ce  = criterion(model(x, t), _remap(y, lm))
                bud = lambda_budget * budget_loss(model, t, kappa)
                ov  = (lambda_overlap * overlap_loss(model, t, rho_sched)
                       if use_ov else torch.tensor(0.0, device=device))
                (ce + ov + bud).backward()
                if use_gradient_masking:
                    model.apply_gradient_mask()
                optimizer.step()

            for i in range(t + 1):
                model._current_task = i
                epoch_curves[t, epoch, i] = _eval_th(model, tasks, i, device)
            model._current_task = t

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

        model.freeze_task(t)
        if use_gradient_masking:
            model.update_ownership()

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)
