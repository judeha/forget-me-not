"""Entry point: python -m scripts.train --config <yaml>"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml

from src.data.split_mnist import get_split_mnist
from src.data.permuted_mnist import get_permuted_mnist
from src.metrics.continual import final_average_accuracy, forgetting, forward_transfer
from src.methods.sequential import TrainResult, eval_accuracy, run_sequential
from src.methods.ewc import EWC, run_ewc
from src.models.mlp import MLP
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _build_tasks(cfg: dict) -> list[dict]:
    dataset = cfg.get("dataset", "split_mnist")
    if dataset == "split_mnist":
        tasks = get_split_mnist(
            data_dir=cfg.get("data_dir", "data/mnist"),
            batch_size=cfg.get("batch_size", 256),
            subset_size=cfg.get("subset_size"),
        )
    elif dataset == "permuted_mnist":
        tasks = get_permuted_mnist(
            data_dir=cfg.get("data_dir", "data/mnist"),
            n_tasks=cfg.get("n_tasks", 10),
            batch_size=cfg.get("batch_size", 256),
            subset_size=cfg.get("subset_size"),
            seed=cfg.get("seed", 42),
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return tasks[: cfg.get("n_tasks", len(tasks))]


def save_artifacts(
    artifact_dir: Path,
    cfg: dict,
    result: TrainResult,
    metrics: dict,
    fisher_stats: dict | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    acc_matrix, epoch_curves, _ = result

    with open(artifact_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    pd.DataFrame(acc_matrix).to_csv(artifact_dir / "accuracy_matrix.csv", index=False)

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    if fisher_stats is not None:
        with open(artifact_dir / "fisher_stats.json", "w") as f:
            json.dump(fisher_stats, f, indent=2)

    # Long-format per-epoch accuracy
    n_tasks, epochs_per_task, _ = epoch_curves.shape
    rows = []
    for t in range(n_tasks):
        for e in range(epochs_per_task):
            for i in range(t + 1):
                rows.append({"train_task": t, "epoch": e, "eval_task": i,
                              "accuracy": float(epoch_curves[t, e, i])})
    pd.DataFrame(rows).to_csv(artifact_dir / "epoch_curves.csv", index=False)

    # Plot 1: accuracy per task after each task completes
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(n_tasks):
        xs = list(range(i, n_tasks))
        ys = [acc_matrix[t, i] for t in xs]
        ax.plot(xs, ys, marker="o", label=f"Task {i+1}")
    ax.set_xlabel("Task trained up to")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy per task — {cfg.get('method', 'sequential')}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_vs_task.png", dpi=150)
    plt.close(fig)

    # Plot 2: per-epoch accuracy curves, one subplot per trained task
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        for i in range(t + 1):
            ys = epoch_curves[t, :, i]
            ax.plot(range(1, epochs_per_task + 1), ys, marker=".", label=f"Task {i+1}")
        ax.set_title(f"Training task {t+1}")
        ax.set_xlabel("Epoch")
        if t == 0:
            ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=7)
    fig.suptitle(f"Per-epoch accuracy — {cfg.get('method', 'sequential')}")
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_vs_epoch.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tasks = _build_tasks(cfg)

    model = MLP(
        input_size=784,
        hidden_sizes=cfg.get("hidden_sizes", [256, 256, 256]),
        output_size=10,
    ).to(device)

    # Random baseline for forward transfer (evaluate untrained model on all tasks)
    random_acc = np.array([eval_accuracy(model, t["test"], device) for t in tasks], dtype=np.float32)

    method = cfg.get("method", "sequential")
    fisher_stats: dict | None = None

    if method == "sequential":
        result = run_sequential(
            model=model,
            tasks=tasks,
            epochs_per_task=cfg.get("epochs_per_task", 5),
            lr=cfg.get("lr", 1e-3),
            device=device,
        )
    elif method == "ewc":
        ewc_obj = EWC(
            lambda_ewc=cfg.get("lambda_ewc", 400.0),
            n_fisher_batches=cfg.get("n_fisher_batches", 50),
        )
        result = run_ewc(
            model=model,
            tasks=tasks,
            epochs_per_task=cfg.get("epochs_per_task", 5),
            lr=cfg.get("lr", 1e-3),
            lambda_ewc=cfg.get("lambda_ewc", 400.0),
            device=device,
            n_fisher_batches=cfg.get("n_fisher_batches", 50),
        )
        # Re-run consolidation on trained model to get stats (already done inside run_ewc)
        # We collect Fisher stats from a fresh EWC obj built during run_ewc — expose via helper
        fisher_stats = _collect_fisher_stats(model, tasks, cfg, device)
    else:
        raise ValueError(f"Unknown method: {method}")

    acc_matrix, _, zero_shot_acc = result
    fwt = forward_transfer(zero_shot_acc, random_acc)
    metrics = {
        "method": method,
        "final_average_accuracy": final_average_accuracy(acc_matrix),
        "forgetting": forgetting(acc_matrix),
        "forward_transfer": fwt,
    }

    print("\nAccuracy matrix (rows=task trained, cols=task evaluated):")
    print(np.round(acc_matrix, 4))
    print("\nMetrics:", metrics)

    artifact_dir = Path(cfg.get("artifact_dir", "artifacts/run"))
    save_artifacts(artifact_dir, cfg, result, metrics, fisher_stats)
    print(f"\nArtifacts saved to {artifact_dir}/")


def _collect_fisher_stats(
    model: nn.Module,
    tasks: list[dict],
    cfg: dict,
    device: torch.device,
) -> dict:
    """Estimate Fisher on all tasks with the trained model for stats summary."""
    from src.methods.ewc import EWC as _EWC
    stats_ewc = _EWC(
        lambda_ewc=cfg.get("lambda_ewc", 400.0),
        n_fisher_batches=cfg.get("n_fisher_batches", 50),
    )
    for task in tasks:
        stats_ewc.consolidate(model, task["train"], device)
    return stats_ewc.fisher_stats()


if __name__ == "__main__":
    main()
