"""Entry point: python -m scripts.train --config configs/split_mnist_sequential.yaml"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from src.data.split_mnist import get_split_mnist
from src.metrics.continual import final_average_accuracy, forgetting
from src.methods.sequential import run_sequential
from src.models.mlp import MLP
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_artifacts(
    artifact_dir: Path,
    cfg: dict,
    acc_matrix: np.ndarray,
    metrics: dict,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)

    with open(artifact_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    pd.DataFrame(acc_matrix).to_csv(artifact_dir / "accuracy_matrix.csv", index=False)

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    n_tasks = acc_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(n_tasks):
        vals = [acc_matrix[t, i] if t >= i else None for t in range(n_tasks)]
        xs = [t for t, v in enumerate(vals) if v is not None]
        ys = [v for v in vals if v is not None]
        ax.plot(xs, ys, marker="o", label=f"Task {i+1}")
    ax.set_xlabel("Task trained up to")
    ax.set_ylabel("Test accuracy")
    ax.set_title("Accuracy per task over sequential training")
    ax.legend()
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_vs_task.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tasks = get_split_mnist(
        data_dir=cfg.get("data_dir", "data/mnist"),
        batch_size=cfg.get("batch_size", 256),
        subset_size=cfg.get("subset_size"),
    )
    tasks = tasks[: cfg.get("n_tasks", 5)]

    model = MLP(
        input_size=784,
        hidden_sizes=cfg.get("hidden_sizes", [256, 256, 256]),
        output_size=10,
    ).to(device)

    acc_matrix = run_sequential(
        model=model,
        tasks=tasks,
        epochs_per_task=cfg.get("epochs_per_task", 5),
        lr=cfg.get("lr", 1e-3),
        device=device,
    )

    metrics = {
        "final_average_accuracy": final_average_accuracy(acc_matrix),
        "forgetting": forgetting(acc_matrix),
    }
    print("\nAccuracy matrix (rows=task trained, cols=task evaluated):")
    print(np.round(acc_matrix, 4))
    print("\nMetrics:", metrics)

    artifact_dir = Path(cfg.get("artifact_dir", "artifacts/run"))
    save_artifacts(artifact_dir, cfg, acc_matrix, metrics)
    print(f"\nArtifacts saved to {artifact_dir}/")


if __name__ == "__main__":
    main()
