from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from src.methods.sequential import TrainResult, eval_accuracy
from src.methods.ewc import EWC
from src.methods.overlap import overlap_loss, budget_loss


def run_ewc_overlap(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    warmup_epochs: int,
    lr: float,
    lambda_ewc: float,
    lambda_overlap: float,
    lambda_budget: float,
    rho_sched: list[float],
    kappa: float,
    n_fisher_batches: int,
    device: torch.device,
) -> TrainResult:
    """Train with EWC on backbone weights + overlap/budget on task masks.

    EWC consolidates only backbone params (task alphas are frozen before
    consolidation, so they're excluded from Fisher estimation).
    """
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()
    ewc = EWC(lambda_ewc=lambda_ewc, n_fisher_batches=n_fisher_batches)

    for t, task in enumerate(tasks):
        model.add_task(t, device)
        model._current_task = t
        zero_shot_acc[t] = eval_accuracy(model, task["test"], device)

        opt_params = model.backbone_params() + model.task_params(t)
        optimizer = torch.optim.Adam(opt_params, lr=lr)

        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks} [EWC+Overlap]", leave=False):
            model.train()
            use_overlap = (epoch >= warmup_epochs) and (t > 0)
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                ce = criterion(model(x, t), y)
                ewc_pen = ewc.penalty(model)
                bud = lambda_budget * budget_loss(model, t, kappa)
                ov = (lambda_overlap * overlap_loss(model, t, rho_sched)
                      if use_overlap else torch.tensor(0.0, device=device))
                (ce + ewc_pen + ov + bud).backward()
                optimizer.step()

            for i in range(t + 1):
                model._current_task = i
                epoch_curves[t, epoch, i] = eval_accuracy(model, tasks[i]["test"], device)
            model._current_task = t

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

        # Freeze task masks first so EWC Fisher only covers backbone weights.
        model.freeze_task(t)
        model._current_task = t
        ewc.consolidate(model, task["train"], device)

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)
