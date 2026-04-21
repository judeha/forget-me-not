from __future__ import annotations

import ssl
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

def _make_task_classes(n_tasks: int) -> list[list[int]]:
    """Divide 100 CIFAR-100 classes evenly across n_tasks."""
    cpt = 100 // n_tasks  # classes per task (floor)
    return [list(range(i * cpt, (i + 1) * cpt)) for i in range(n_tasks)]

_MEAN = (0.5071, 0.4867, 0.4408)
_STD = (0.2675, 0.2565, 0.2761)


def _filter_cifar(
    dataset: datasets.CIFAR100,
    classes: list[int],
    subset_size: Optional[int],
) -> TensorDataset:
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, classes))[0]
    if subset_size is not None:
        indices = indices[:subset_size]
    data = torch.from_numpy(dataset.data[indices]).permute(0, 3, 1, 2).float() / 255.0
    mean = torch.tensor(_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_STD).view(1, 3, 1, 1)
    data = (data - mean) / std
    labels = torch.tensor(targets[indices], dtype=torch.long)
    return TensorDataset(data, labels)


def get_split_cifar100(
    data_dir: str = "data/cifar100",
    n_tasks: int = 10,
    batch_size: int = 128,
    subset_size: Optional[int] = None,
) -> list[dict]:
    """Return task dicts for Split CIFAR-100.

    Supports n_tasks in {5,10,15,20}: 100//n_tasks classes per task.
    Each dict has train/test loaders, n_classes, and label_map.
    """
    # Bypass SSL cert issues in some environments
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=None)
        test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=None)
    finally:
        ssl._create_default_https_context = _orig

    task_classes = _make_task_classes(n_tasks)
    tasks = []
    for classes in task_classes:
        train_ds = _filter_cifar(train_full, classes, subset_size)
        test_ds = _filter_cifar(test_full, classes, subset_size)
        tasks.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "classes": classes,
            "n_classes": len(classes),
            "label_map": {c: i for i, c in enumerate(classes)},
        })
    return tasks
