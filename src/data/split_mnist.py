from __future__ import annotations

from typing import Optional
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms


TASK_CLASSES: list[tuple[int, int]] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
]

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])


def _filter_by_classes(
    dataset: datasets.MNIST,
    classes: tuple[int, int],
    subset_size: Optional[int],
) -> TensorDataset:
    targets = torch.tensor(dataset.targets) if not isinstance(dataset.targets, torch.Tensor) else dataset.targets
    mask = (targets == classes[0]) | (targets == classes[1])
    indices = mask.nonzero(as_tuple=True)[0]
    if subset_size is not None:
        indices = indices[:subset_size]
    data = dataset.data[indices].float().view(-1, 784) / 255.0
    # Normalize
    data = (data - 0.1307) / 0.3081
    labels = targets[indices]
    return TensorDataset(data, labels)


def get_split_mnist(
    data_dir: str = "data/mnist",
    batch_size: int = 256,
    subset_size: Optional[int] = None,
) -> list[dict[str, DataLoader]]:
    """Return a list of task dicts with 'train' and 'test' DataLoaders."""
    train_full = datasets.MNIST(data_dir, train=True, download=True, transform=None)
    test_full = datasets.MNIST(data_dir, train=False, download=True, transform=None)

    tasks = []
    for classes in TASK_CLASSES:
        train_ds = _filter_by_classes(train_full, classes, subset_size)
        test_ds = _filter_by_classes(test_full, classes, subset_size)
        tasks.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True),
            "test": DataLoader(test_ds, batch_size=batch_size, shuffle=False),
            "classes": classes,
            "n_classes": 2,
            "label_map": {classes[0]: 0, classes[1]: 1},
        })
    return tasks
