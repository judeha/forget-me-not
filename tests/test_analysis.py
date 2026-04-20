import numpy as np
import torch
import pytest

from src.metrics.analysis import rsa_per_layer, fisher_overlap_per_layer, extract_layer_activations
from src.models.mlp import MLP
from src.data.split_mnist import get_split_mnist
from src.utils.seed import set_seed


@pytest.fixture
def two_task_setup():
    set_seed(0)
    device = torch.device("cpu")
    tasks = get_split_mnist("data/mnist", batch_size=32, subset_size=40)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[32, 32], output_size=10)
    return model, tasks, device


def test_rsa_matrix_shape():
    acts = {
        0: [np.random.randn(50, 32), np.random.randn(50, 32)],
        1: [np.random.randn(50, 32), np.random.randn(50, 32)],
    }
    result = rsa_per_layer(acts)
    assert set(result.keys()) == {0, 1}
    for mat in result.values():
        assert mat.shape == (2, 2)


def test_rsa_diagonal_is_one():
    v = np.random.randn(50, 32)
    acts = {0: [v], 1: [v * 2]}  # same direction, different scale
    result = rsa_per_layer(acts)
    assert result[0][0, 0] == pytest.approx(1.0, abs=1e-5)
    assert result[0][1, 1] == pytest.approx(1.0, abs=1e-5)


def test_fisher_overlap_shape():
    f1 = {"net.0": torch.rand(32, 10), "net.1": torch.rand(32)}
    f2 = {"net.0": torch.rand(32, 10), "net.1": torch.rand(32)}
    result = fisher_overlap_per_layer([f1, f2])
    assert "net.0" in result
    assert result["net.0"].shape == (2, 2)


def test_fisher_overlap_diagonal_is_one():
    v = torch.rand(32)
    f = {"w": v}
    result = fisher_overlap_per_layer([f, f])
    assert result["w"][0, 0] == pytest.approx(1.0, abs=1e-5)


def test_extract_activations_shape(two_task_setup):
    model, tasks, device = two_task_setup
    acts = extract_layer_activations(model, tasks[0]["test"], device, n_samples=40)
    assert len(acts) == 2  # two hidden layers
    assert acts[0].shape[0] <= 40
    assert acts[0].shape[1] == 32
