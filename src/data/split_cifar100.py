from __future__ import annotations

import ssl
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets

# 10 tasks of 10 classes each
TASK_CLASSES: list[list[int]] = [list(range(i * 10, (i + 1) * 10)) for i in range(10)]

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
    """Return a list of task dicts with 'train' and 'test' DataLoaders.

    10 tasks of 10 classes each (classes 0-9, 10-19, ..., 90-99).
    """
    # Bypass SSL cert issues in some environments
    _orig = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        train_full = datasets.CIFAR100(data_dir, train=True, download=True, transform=None)
        test_full = datasets.CIFAR100(data_dir, train=False, download=True, transform=None)
    finally:
        ssl._create_default_https_context = _orig

    tasks = []
    for classes in TASK_CLASSES[:n_tasks]:
        train_ds = _filter_cifar(train_full, classes, subset_size)
        test_ds = _filter_cifar(test_full, classes, subset_size)
        tasks.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "classes": classes,
        })
    return tasks
