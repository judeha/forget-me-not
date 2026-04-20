from __future__ import annotations

import numpy as np


def final_average_accuracy(acc_matrix: np.ndarray) -> float:
    """Mean accuracy on all tasks after training on the last task."""
    n_tasks = acc_matrix.shape[0]
    return float(acc_matrix[n_tasks - 1, :].mean())


def forgetting(acc_matrix: np.ndarray) -> float:
    """Mean forgetting: average drop from peak to final accuracy per task."""
    n_tasks = acc_matrix.shape[0]
    f_per_task = []
    for i in range(n_tasks - 1):
        peak = acc_matrix[i:, i].max()
        final = acc_matrix[n_tasks - 1, i]
        f_per_task.append(peak - final)
    return float(np.mean(f_per_task)) if f_per_task else 0.0


def forward_transfer(zero_shot_acc: np.ndarray, random_acc: np.ndarray) -> float:
    """Mean forward transfer: how much prior learning helps on future tasks.

    FWT = mean over tasks i > 0 of [zero_shot_acc[i] - random_acc[i]]

    zero_shot_acc[i]: accuracy on task i right before training on it
                      (after training on tasks 0..i-1)
    random_acc[i]:    accuracy on task i with a freshly initialized model
                      (no training at all)

    Positive FWT means prior tasks help; negative means they hurt (interference).
    """
    if len(zero_shot_acc) < 2:
        return 0.0
    return float(np.mean(zero_shot_acc[1:] - random_acc[1:]))
