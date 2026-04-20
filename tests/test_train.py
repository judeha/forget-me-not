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
    acc_matrix, epoch_curves, zero_shot_acc = run_sequential(
        model, tasks, epochs_per_task=1, lr=1e-3, device=device
    )
    assert acc_matrix.shape == (2, 2)
    assert epoch_curves.shape == (2, 1, 2)
    assert zero_shot_acc.shape == (2,)


def test_accuracy_matrix_range():
    set_seed(1)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=64, subset_size=30)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[64], output_size=10)
    acc_matrix, epoch_curves, zero_shot_acc = run_sequential(
        model, tasks, epochs_per_task=1, lr=1e-3, device=device
    )
    for t in range(2):
        for i in range(t + 1):
            assert 0.0 <= acc_matrix[t, i] <= 1.0
            assert 0.0 <= epoch_curves[t, 0, i] <= 1.0
    for v in zero_shot_acc:
        assert 0.0 <= float(v) <= 1.0


def test_smoke_train():
    """Full smoke run: 2 tasks, 1 epoch, tiny subset."""
    set_seed(42)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=32, subset_size=40)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[32, 32], output_size=10)
    acc_matrix, epoch_curves, zero_shot_acc = run_sequential(
        model, tasks, epochs_per_task=1, lr=1e-3, device=device
    )
    assert acc_matrix.shape == (2, 2)
    assert not np.isnan(acc_matrix).any()
    # NaN only in upper triangle of epoch_curves
    assert not np.isnan(epoch_curves[0, :, 0]).any()
    assert np.isnan(epoch_curves[0, :, 1]).all()
