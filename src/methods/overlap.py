from __future__ import annotations

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from src.methods.sequential import TrainResult, eval_accuracy


def make_rho_schedule(
    n_layers: int,
    rho_max: float,
    rho_min: float,
    mode: str,
) -> list[float]:
    """Return per-layer target overlap values.

    uniform:      rho_l = (rho_max + rho_min) / 2 for all l  (constant at mean)
    hierarchical: linearly decreasing rho_max (layer 0) → rho_min (last layer)
    reversed:     linearly increasing rho_min (layer 0) → rho_max (last layer)
    """
    if mode == "uniform":
        return [(rho_max + rho_min) / 2] * n_layers
    if n_layers == 1:
        return [rho_max]
    if mode == "reversed":
        return [
            rho_min + (rho_max - rho_min) * i / (n_layers - 1)
            for i in range(n_layers)
        ]
    # hierarchical (default)
    return [
        rho_max + (rho_min - rho_max) * i / (n_layers - 1)
        for i in range(n_layers)
    ]


def overlap_loss(
    model: nn.Module,
    current_task: int,
    rho_sched: list[float],
) -> torch.Tensor:
    """Cosine-similarity overlap loss vs all past tasks.

    L_ov = mean over past tasks k, layers l of (cos(m_l^t, m_l^k) - rho_l)^2
    """
    device = next(model.parameters()).device
    if current_task == 0:
        return torch.tensor(0.0, device=device)

    current_masks = model.masks(current_task)
    total = torch.tensor(0.0, device=device)
    count = 0
    for past_t in range(current_task):
        past_masks = model.masks(past_t)
        for l, (cm, pm, rho) in enumerate(zip(current_masks, past_masks, rho_sched)):
            cos = F.cosine_similarity(cm.unsqueeze(0), pm.detach().unsqueeze(0))
            total = total + (cos - rho) ** 2
            count += 1
    return total / max(count, 1)


def budget_loss(
    model: nn.Module,
    task_id: int,
    kappa: float,
) -> torch.Tensor:
    """Budget regularizer: penalize deviation of mean mask from target kappa.

    L_budget = mean over layers l of (mean_j(m_l^t_j) - kappa)^2
    """
    masks = model.masks(task_id)
    device = masks[0].device
    total = torch.tensor(0.0, device=device)
    for m in masks:
        total = total + (m.mean() - kappa) ** 2
    return total / len(masks)


def run_overlap(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    warmup_epochs: int,
    lr: float,
    lambda_overlap: float,
    lambda_budget: float,
    rho_sched: list[float],
    kappa: float,
    device: torch.device,
) -> TrainResult:
    """Train with overlap + budget regularization; return TrainResult.

    Each task gets fresh mask alphas (init=0 → sigmoid=0.5).
    Past task masks are frozen after training.
    Overlap regularization is skipped during warmup_epochs and for task 0.
    """
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()

    for t, task in enumerate(tasks):
        model.add_task(t, device)
        model._current_task = t
        zero_shot_acc[t] = eval_accuracy(model, task["test"], device)

        opt_params = model.backbone_params() + model.task_params(t)
        optimizer = torch.optim.Adam(opt_params, lr=lr)

        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks} [Overlap]", leave=False):
            model.train()
            use_overlap = (epoch >= warmup_epochs) and (t > 0)
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                ce = criterion(model(x, t), y)
                bud = lambda_budget * budget_loss(model, t, kappa)
                ov = lambda_overlap * overlap_loss(model, t, rho_sched) if use_overlap else torch.tensor(0.0, device=device)
                (ce + ov + bud).backward()
                optimizer.step()

            for i in range(t + 1):
                model._current_task = i
                epoch_curves[t, epoch, i] = eval_accuracy(model, tasks[i]["test"], device)
            model._current_task = t

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

        model.freeze_task(t)

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)


def collect_mask_artifacts(model: nn.Module, n_tasks: int) -> dict:
    """Collect mask densities and pairwise cosine overlaps after training."""
    artifacts: dict = {
        "mask_density": {},
        "pairwise_overlap": {},
    }
    present = [t for t in range(n_tasks) if str(t) in model.task_alphas]

    for t in present:
        masks = [m.detach().cpu() for m in model.masks(t)]
        artifacts["mask_density"][t] = [round(float(m.mean()), 5) for m in masks]

    for i, t1 in enumerate(present):
        for t2 in present[i + 1:]:
            m1 = [m.detach().cpu() for m in model.masks(t1)]
            m2 = [m.detach().cpu() for m in model.masks(t2)]
            cos_per_layer = [
                round(float(F.cosine_similarity(m1[l].unsqueeze(0), m2[l].unsqueeze(0)).item()), 5)
                for l in range(len(m1))
            ]
            artifacts["pairwise_overlap"][f"{t1}_{t2}"] = cos_per_layer

    return artifacts
