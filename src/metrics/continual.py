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
        # peak accuracy on task i while training up to any task t <= n_tasks-1
        peak = acc_matrix[i:, i].max()
        final = acc_matrix[n_tasks - 1, i]
        f_per_task.append(peak - final)
    return float(np.mean(f_per_task)) if f_per_task else 0.0


def forward_transfer(acc_matrix: np.ndarray) -> float:
    # TODO: compute forward transfer as the influence of learning task t on the
    # performance of future task i (requires a random-init baseline; not yet
    # available in this pipeline).
    raise NotImplementedError("Forward transfer requires a random-init baseline — see TODO above.")
