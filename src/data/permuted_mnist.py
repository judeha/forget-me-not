from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def _permute_dataset(
    mnist: datasets.MNIST,
    permutation: np.ndarray,
    subset_size: Optional[int],
) -> TensorDataset:
    targets = torch.tensor(mnist.targets) if not isinstance(mnist.targets, torch.Tensor) else mnist.targets
    n = len(targets) if subset_size is None else min(subset_size, len(targets))
    data = mnist.data[:n].float().view(-1, 784) / 255.0
    data = (data - 0.1307) / 0.3081
    data = data[:, permutation]  # apply fixed pixel permutation
    return TensorDataset(data, targets[:n])


def get_permuted_mnist(
    data_dir: str = "data/mnist",
    n_tasks: int = 10,
    batch_size: int = 256,
    subset_size: Optional[int] = None,
    seed: int = 42,
) -> list[dict]:
    """Return a list of task dicts with 'train' and 'test' DataLoaders.

    Each task uses the same MNIST data with a distinct fixed pixel permutation.
    All 10 digit classes appear in every task.
    """
    rng = np.random.RandomState(seed)
    permutations = [rng.permutation(784) for _ in range(n_tasks)]

    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=None)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=None)

    tasks = []
    for perm in permutations:
        train_ds = _permute_dataset(train_full, perm, subset_size)
        test_ds = _permute_dataset(test_full, perm, subset_size)
        tasks.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
        })
    return tasks
