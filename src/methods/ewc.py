from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.methods.sequential import TrainResult, eval_accuracy


class EWC:
    """Elastic Weight Consolidation regularizer (per-task Fisher storage)."""

    def __init__(self, lambda_ewc: float, n_fisher_batches: int = 50) -> None:
        self.lambda_ewc = lambda_ewc
        self.n_fisher_batches = n_fisher_batches
        # List of (fisher_dict, theta_star_dict) one entry per consolidated task
        self._consolidated: list[tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]] = []

    def consolidate(self, model: nn.Module, loader: DataLoader, device: torch.device) -> None:
        """Estimate diagonal Fisher on loader and snapshot current params."""
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device="cpu")
            for n, p in model.named_parameters() if p.requires_grad
        }

        model.eval()
        n_batches = 0
        for x, y in loader:
            if n_batches >= self.n_fisher_batches:
                break
            x = x.to(device)
            model.zero_grad()
            log_probs = F.log_softmax(model(x), dim=1)
            # Use model's own distribution to sample labels (empirical Fisher)
            sampled = log_probs.detach().exp().multinomial(1).squeeze(1)
            loss = F.nll_loss(log_probs, sampled)
            loss.backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().cpu() ** 2
            n_batches += 1

        if n_batches > 0:
            for n in fisher:
                fisher[n] /= n_batches

        theta_star = {
            n: p.detach().cpu().clone()
            for n, p in model.named_parameters() if p.requires_grad
        }
        self._consolidated.append((fisher, theta_star))

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC penalty: (lambda/2) * sum_t sum_j F_j^t * (theta_j - theta_star_j^t)^2"""
        if not self._consolidated:
            return torch.tensor(0.0, device=next(model.parameters()).device)

        device = next(model.parameters()).device
        total = torch.tensor(0.0, device=device)
        for fisher, theta_star in self._consolidated:
            for n, p in model.named_parameters():
                if p.requires_grad and n in fisher:
                    f = fisher[n].to(device)
                    ts = theta_star[n].to(device)
                    total = total + (f * (p - ts) ** 2).sum()
        return (self.lambda_ewc / 2) * total

    def fisher_stats(self) -> dict:
        """Per-layer Fisher summary across all consolidated tasks."""
        stats: dict = {"n_tasks_consolidated": len(self._consolidated)}
        if not self._consolidated:
            return stats
        accum: dict[str, torch.Tensor] = {}
        for fisher, _ in self._consolidated:
            for n, f in fisher.items():
                accum[n] = accum[n] + f if n in accum else f.clone()
        per_param = {}
        for n, f in accum.items():
            per_param[n] = {
                "mean": round(float(f.mean()), 6),
                "max": round(float(f.max()), 6),
                "sparsity": round(float((f < 1e-10).float().mean()), 4),
            }
        stats["per_param"] = per_param
        return stats


def run_ewc(
    model: nn.Module,
    tasks: list[dict],
    epochs_per_task: int,
    lr: float,
    lambda_ewc: float,
    device: torch.device,
    n_fisher_batches: int = 50,
) -> TrainResult:
    """Train with EWC regularization; return TrainResult matching run_sequential."""
    n_tasks = len(tasks)
    acc_matrix = np.zeros((n_tasks, n_tasks), dtype=np.float32)
    epoch_curves = np.full((n_tasks, epochs_per_task, n_tasks), np.nan, dtype=np.float32)
    zero_shot_acc = np.zeros(n_tasks, dtype=np.float32)

    criterion = nn.CrossEntropyLoss()
    ewc = EWC(lambda_ewc=lambda_ewc, n_fisher_batches=n_fisher_batches)

    for t, task in enumerate(tasks):
        zero_shot_acc[t] = eval_accuracy(model, task["test"], device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        for epoch in tqdm(range(epochs_per_task), desc=f"Task {t+1}/{n_tasks} [EWC]", leave=False):
            model.train()
            for x, y in task["train"]:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                loss = criterion(model(x), y) + ewc.penalty(model)
                loss.backward()
                optimizer.step()
            for i in range(t + 1):
                epoch_curves[t, epoch, i] = eval_accuracy(model, tasks[i]["test"], device)

        for i in range(t + 1):
            acc_matrix[t, i] = epoch_curves[t, -1, i]

        ewc.consolidate(model, task["train"], device)

    return TrainResult(acc_matrix, epoch_curves, zero_shot_acc)
