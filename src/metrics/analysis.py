from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader


def extract_layer_activations(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    task_id: int = 0,
    n_samples: int = 200,
) -> list[np.ndarray]:
    """Extract post-activation representations at each layer for n_samples examples.

    Calls model.get_layer_activations(x, task_id) if task_id param is accepted,
    otherwise model.get_layer_activations(x).
    Returns list of arrays, one per layer, shape (n_samples, hidden_dim).
    """
    import inspect
    sig = inspect.signature(model.get_layer_activations)
    has_task_id = "task_id" in sig.parameters

    model.eval()
    all_acts: list[list[np.ndarray]] | None = None
    n_collected = 0
    with torch.no_grad():
        for x, _ in loader:
            if n_collected >= n_samples:
                break
            x = x[:min(n_samples - n_collected, len(x))].to(device)
            acts = (
                model.get_layer_activations(x, task_id=task_id)
                if has_task_id
                else model.get_layer_activations(x)
            )
            if all_acts is None:
                all_acts = [[] for _ in acts]
            for l, a in enumerate(acts):
                all_acts[l].append(a.cpu().numpy())
            n_collected += x.size(0)
    if all_acts is None:
        return []
    return [np.concatenate(a, axis=0) for a in all_acts]


def rsa_per_layer(
    activations_by_task: dict[int, list[np.ndarray]],
) -> dict[int, np.ndarray]:
    """Cosine similarity between mean layer activations across tasks.

    activations_by_task: {task_id -> list[layer_array]}
    Returns: {layer_idx -> (n_tasks, n_tasks) cosine similarity matrix}
    """
    task_ids = sorted(activations_by_task.keys())
    n_tasks = len(task_ids)
    if n_tasks == 0:
        return {}
    n_layers = len(activations_by_task[task_ids[0]])

    result: dict[int, np.ndarray] = {}
    for l in range(n_layers):
        means = [activations_by_task[t][l].mean(axis=0) for t in task_ids]
        mat = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                vi, vj = means[i], means[j]
                denom = np.linalg.norm(vi) * np.linalg.norm(vj) + 1e-8
                mat[i, j] = np.dot(vi, vj) / denom
        result[l] = mat
    return result


def fisher_overlap_per_layer(
    fishers_by_task: list[dict[str, torch.Tensor]],
) -> dict[str, np.ndarray]:
    """Normalized Fisher dot product between tasks, per parameter group.

    fishers_by_task: list of {param_name -> Fisher tensor}, one dict per task.
    Returns: {param_name -> (n_tasks, n_tasks) overlap matrix}
    """
    n_tasks = len(fishers_by_task)
    if n_tasks == 0:
        return {}
    param_names = list(fishers_by_task[0].keys())

    result: dict[str, np.ndarray] = {}
    for name in param_names:
        vecs = [fishers_by_task[t][name].cpu().numpy().ravel() for t in range(n_tasks)]
        mat = np.zeros((n_tasks, n_tasks))
        for i in range(n_tasks):
            for j in range(n_tasks):
                norm = np.linalg.norm(vecs[i]) * np.linalg.norm(vecs[j]) + 1e-8
                mat[i, j] = np.dot(vecs[i], vecs[j]) / norm
        result[name] = mat
    return result


def compute_posthoc_fishers(
    model: torch.nn.Module,
    tasks: list[dict],
    device: torch.device,
    n_batches: int = 20,
) -> list[dict[str, torch.Tensor]]:
    """Compute diagonal Fisher for each task using the (already-trained) model."""
    import torch.nn.functional as F

    fishers = []
    model.eval()
    for task in tasks:
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p, device="cpu")
            for n, p in model.named_parameters() if p.requires_grad
        }
        count = 0
        for x, _ in task["train"]:
            if count >= n_batches:
                break
            x = x.to(device)
            model.zero_grad()
            log_probs = F.log_softmax(model(x), dim=1)
            sampled = log_probs.detach().exp().multinomial(1).squeeze(1)
            F.nll_loss(log_probs, sampled).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach().cpu() ** 2
            count += 1
        if count > 0:
            for n in fisher:
                fisher[n] /= count
        fishers.append(fisher)
    return fishers
