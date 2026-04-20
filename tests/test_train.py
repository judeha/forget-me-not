import numpy as np
import torch
import pytest

from src.data.split_mnist import get_split_mnist
from src.methods.sequential import run_sequential
from src.models.mlp import MLP
from src.utils.seed import set_seed


def test_accuracy_matrix_shape():
    set_seed(0)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=64, subset_size=30)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[64], output_size=10)
    acc_matrix = run_sequential(model, tasks, epochs_per_task=1, lr=1e-3, device=device)
    assert acc_matrix.shape == (2, 2)


def test_accuracy_matrix_range():
    set_seed(1)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=64, subset_size=30)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[64], output_size=10)
    acc_matrix = run_sequential(model, tasks, epochs_per_task=1, lr=1e-3, device=device)
    # Only filled positions (t >= i) should be in [0, 1]
    for t in range(2):
        for i in range(t + 1):
            assert 0.0 <= acc_matrix[t, i] <= 1.0


def test_smoke_train():
    """Full smoke run: 2 tasks, 1 epoch, tiny subset."""
    set_seed(42)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=32, subset_size=40)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[32, 32], output_size=10)
    acc_matrix = run_sequential(model, tasks, epochs_per_task=1, lr=1e-3, device=device)
    assert acc_matrix.shape == (2, 2)
    assert not np.isnan(acc_matrix).any()
