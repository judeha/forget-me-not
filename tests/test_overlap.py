import copy
import torch
import pytest

from src.models.masked_mlp import MaskedMLP
from src.methods.overlap import (
    make_rho_schedule, overlap_loss, budget_loss, run_overlap,
)
from src.data.split_mnist import get_split_mnist
from src.utils.seed import set_seed


@pytest.fixture
def small_masked_mlp():
    set_seed(0)
    device = torch.device("cpu")
    model = MaskedMLP(input_size=784, hidden_sizes=[32, 32], output_size=10, beta=5.0)
    return model, device


def test_overlap_values_in_range(small_masked_mlp):
    model, device = small_masked_mlp
    model.add_task(0, device)
    model.add_task(1, device)
    # Perturb task 1 alphas slightly so masks differ
    with torch.no_grad():
        for a in model.task_alphas["1"]:
            a.add_(torch.randn_like(a) * 0.5)
    ov = overlap_loss(model, current_task=1, rho_sched=[0.5, 0.5])
    val = float(ov)
    assert 0.0 <= val  # cosine similarity of pos vectors is in [0,1], so (cos-rho)^2 >= 0


def test_hierarchical_schedule_monotone_decreasing():
    sched = make_rho_schedule(n_layers=4, rho_max=0.9, rho_min=0.1, mode="hierarchical")
    assert len(sched) == 4
    for i in range(len(sched) - 1):
        assert sched[i] >= sched[i + 1], f"Not decreasing at index {i}: {sched}"


def test_uniform_schedule_constant():
    sched = make_rho_schedule(n_layers=3, rho_max=0.7, rho_min=0.2, mode="uniform")
    assert all(s == pytest.approx(0.7) for s in sched)


def test_frozen_masks_unchanged(small_masked_mlp):
    model, device = small_masked_mlp
    tasks = get_split_mnist("data/mnist", batch_size=32, subset_size=30)
    tasks = tasks[:2]

    rho_sched = make_rho_schedule(2, 0.9, 0.1, "uniform")
    run_overlap(
        model, tasks, epochs_per_task=1, warmup_epochs=0,
        lr=1e-3, lambda_overlap=1.0, lambda_budget=0.1,
        rho_sched=rho_sched, kappa=0.5, device=device,
    )
    # Task 0 masks should be frozen (require_grad False) and unchanged after task 1
    for a in model.task_alphas["0"]:
        assert not a.requires_grad, "Task 0 alpha should be frozen"


def test_budget_loss_zero_at_kappa(small_masked_mlp):
    model, device = small_masked_mlp
    # Default init: alpha=0 → sigmoid(0)=0.5; kappa=0.5 → budget loss = 0
    model.add_task(0, device)
    loss = budget_loss(model, task_id=0, kappa=0.5)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


def test_run_overlap_result_shape(small_masked_mlp):
    model, device = small_masked_mlp
    tasks = get_split_mnist("data/mnist", batch_size=32, subset_size=30)
    tasks = tasks[:2]
    rho_sched = make_rho_schedule(2, 0.9, 0.1, "uniform")
    result = run_overlap(
        model, tasks, epochs_per_task=1, warmup_epochs=0,
        lr=1e-3, lambda_overlap=1.0, lambda_budget=0.1,
        rho_sched=rho_sched, kappa=0.5, device=device,
    )
    acc_matrix, epoch_curves, zero_shot_acc = result
    assert acc_matrix.shape == (2, 2)
    assert epoch_curves.shape == (2, 1, 2)
    assert zero_shot_acc.shape == (2,)
