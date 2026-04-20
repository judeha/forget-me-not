import copy
import torch
import pytest

from src.data.split_mnist import get_split_mnist
from src.methods.ewc import EWC, run_ewc
from src.models.mlp import MLP
from src.utils.seed import set_seed


@pytest.fixture
def small_setup():
    set_seed(0)
    device = torch.device("cpu")
    tasks = get_split_mnist(data_dir="data/mnist", batch_size=32, subset_size=40)
    tasks = tasks[:2]
    model = MLP(input_size=784, hidden_sizes=[32, 32], output_size=10)
    return model, tasks, device


def test_fisher_shapes_match_params(small_setup):
    model, tasks, device = small_setup
    ewc = EWC(lambda_ewc=1.0, n_fisher_batches=2)
    ewc.consolidate(model, tasks[0]["train"], device)
    fisher, _ = ewc._consolidated[0]
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert name in fisher, f"Fisher missing for {name}"
            assert fisher[name].shape == param.shape, (
                f"Shape mismatch for {name}: {fisher[name].shape} vs {param.shape}"
            )


def test_ewc_penalty_zero_at_theta_star(small_setup):
    model, tasks, device = small_setup
    ewc = EWC(lambda_ewc=100.0, n_fisher_batches=2)
    ewc.consolidate(model, tasks[0]["train"], device)
    # Penalty must be exactly zero when params == theta_star
    penalty = ewc.penalty(model)
    assert float(penalty) == pytest.approx(0.0, abs=1e-6)


def test_ewc_penalty_positive_after_perturbation(small_setup):
    model, tasks, device = small_setup
    ewc = EWC(lambda_ewc=100.0, n_fisher_batches=5)
    ewc.consolidate(model, tasks[0]["train"], device)
    # Perturb parameters
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    penalty = ewc.penalty(model)
    assert float(penalty) > 0.0


def test_run_ewc_result_shape(small_setup):
    model, tasks, device = small_setup
    result = run_ewc(model, tasks, epochs_per_task=1, lr=1e-3,
                     lambda_ewc=100.0, device=device, n_fisher_batches=2)
    acc_matrix, epoch_curves, zero_shot_acc = result
    assert acc_matrix.shape == (2, 2)
    assert epoch_curves.shape == (2, 1, 2)
    assert zero_shot_acc.shape == (2,)


def test_run_ewc_zero_shot_acc_range(small_setup):
    model, tasks, device = small_setup
    result = run_ewc(model, tasks, epochs_per_task=1, lr=1e-3,
                     lambda_ewc=100.0, device=device, n_fisher_batches=2)
    for v in result.zero_shot_acc:
        assert 0.0 <= float(v) <= 1.0
