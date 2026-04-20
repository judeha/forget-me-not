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

from src.metrics.continual import final_average_accuracy, forgetting, forward_transfer
from src.methods.sequential import TrainResult, eval_accuracy, run_sequential
from src.methods.ewc import EWC, run_ewc
from src.models.mlp import MLP
from src.utils.seed import set_seed


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ── dataset builders ──────────────────────────────────────────────────────────

def _build_tasks(cfg: dict) -> list[dict]:
    dataset = cfg.get("dataset", "split_mnist")
    kw = dict(data_dir=cfg.get("data_dir", "data/mnist"),
               batch_size=cfg.get("batch_size", 256),
               subset_size=cfg.get("subset_size"))
    if dataset == "split_mnist":
        from src.data.split_mnist import get_split_mnist
        tasks = get_split_mnist(**kw)
    elif dataset == "permuted_mnist":
        from src.data.permuted_mnist import get_permuted_mnist
        tasks = get_permuted_mnist(**kw, n_tasks=cfg.get("n_tasks", 10), seed=cfg.get("seed", 42))
    elif dataset == "split_cifar100":
        from src.data.split_cifar100 import get_split_cifar100
        tasks = get_split_cifar100(
            data_dir=cfg.get("data_dir", "data/cifar100"),
            n_tasks=cfg.get("n_tasks", 10),
            batch_size=cfg.get("batch_size", 128),
            subset_size=cfg.get("subset_size"),
        )
        return tasks  # CIFAR returns exactly n_tasks
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    return tasks[: cfg.get("n_tasks", len(tasks))]


# ── model builders ────────────────────────────────────────────────────────────

def _build_model(cfg: dict, device: torch.device) -> nn.Module:
    model_type = cfg.get("model_type", "mlp")
    method = cfg.get("method", "sequential")
    masked = method in ("overlap_uniform", "overlap_hierarchical")

    if model_type == "mlp":
        if masked:
            from src.models.masked_mlp import MaskedMLP
            model = MaskedMLP(
                input_size=784,
                hidden_sizes=cfg.get("hidden_sizes", [256, 256, 256]),
                output_size=10,
                beta=cfg.get("mask_beta", 5.0),
            )
        else:
            model = MLP(
                input_size=784,
                hidden_sizes=cfg.get("hidden_sizes", [256, 256, 256]),
                output_size=10,
            )
    elif model_type == "cnn":
        if masked:
            from src.models.cnn import MaskedCNN
            model = MaskedCNN(
                in_channels=3,
                conv_channels=cfg.get("conv_channels", [32, 64, 128]),
                output_size=cfg.get("output_size", 100),
                input_hw=cfg.get("input_hw", 32),
                beta=cfg.get("mask_beta", 5.0),
            )
        else:
            from src.models.cnn import CNN
            model = CNN(
                in_channels=3,
                conv_channels=cfg.get("conv_channels", [32, 64, 128]),
                output_size=cfg.get("output_size", 100),
                input_hw=cfg.get("input_hw", 32),
            )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    return model.to(device)


# ── method runners ────────────────────────────────────────────────────────────

def _run_method(
    model: nn.Module,
    tasks: list[dict],
    cfg: dict,
    device: torch.device,
) -> tuple[TrainResult, dict | None]:
    method = cfg.get("method", "sequential")
    extra: dict | None = None  # optional extra artifacts (Fisher stats, mask artifacts)

    if method == "sequential":
        result = run_sequential(
            model=model, tasks=tasks,
            epochs_per_task=cfg.get("epochs_per_task", 5),
            lr=cfg.get("lr", 1e-3), device=device,
        )

    elif method == "ewc":
        result = run_ewc(
            model=model, tasks=tasks,
            epochs_per_task=cfg.get("epochs_per_task", 5),
            lr=cfg.get("lr", 1e-3),
            lambda_ewc=cfg.get("lambda_ewc", 400.0),
            device=device,
            n_fisher_batches=cfg.get("n_fisher_batches", 50),
        )
        extra = _collect_fisher_stats(model, tasks, cfg, device)

    elif method in ("overlap_uniform", "overlap_hierarchical"):
        from src.methods.overlap import make_rho_schedule, run_overlap, collect_mask_artifacts
        mode = "uniform" if method == "overlap_uniform" else "hierarchical"
        rho_sched = make_rho_schedule(
            model.n_mask_layers,
            cfg.get("rho_max", 0.9),
            cfg.get("rho_min", 0.1),
            mode,
        )
        result = run_overlap(
            model=model, tasks=tasks,
            epochs_per_task=cfg.get("epochs_per_task", 5),
            warmup_epochs=cfg.get("warmup_epochs", 1),
            lr=cfg.get("lr", 1e-3),
            lambda_overlap=cfg.get("lambda_overlap", 1.0),
            lambda_budget=cfg.get("lambda_budget", 0.1),
            rho_sched=rho_sched,
            kappa=cfg.get("kappa", 0.5),
            device=device,
        )
        extra = collect_mask_artifacts(model, len(tasks))
    else:
        raise ValueError(f"Unknown method: {method}")

    return result, extra


# ── analysis ──────────────────────────────────────────────────────────────────

def _run_analysis(
    model: nn.Module,
    tasks: list[dict],
    device: torch.device,
    n_tasks: int,
) -> dict:
    from src.metrics.analysis import (
        extract_layer_activations, rsa_per_layer,
        fisher_overlap_per_layer, compute_posthoc_fishers,
    )
    # RSA: activations on probe examples from each task
    acts_by_task = {}
    for t in range(n_tasks):
        task_id = t if hasattr(model, "task_alphas") else None
        kw = dict(task_id=task_id) if task_id is not None else {}
        acts_by_task[t] = extract_layer_activations(model, tasks[t]["test"], device, n_samples=100, **kw)

    rsa = rsa_per_layer(acts_by_task)

    # Post-hoc Fisher overlap
    fishers = compute_posthoc_fishers(model, tasks, device, n_batches=10)
    fisher_overlap = fisher_overlap_per_layer(fishers)

    return {"rsa": rsa, "fisher_overlap": fisher_overlap, "fishers": fishers}


# ── artifact saving ───────────────────────────────────────────────────────────

def save_artifacts(
    artifact_dir: Path,
    cfg: dict,
    result: TrainResult,
    metrics: dict,
    extra: dict | None = None,
    analysis: dict | None = None,
) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    acc_matrix, epoch_curves, _ = result

    with open(artifact_dir / "config.yaml", "w") as f:
        yaml.dump(cfg, f)

    pd.DataFrame(acc_matrix).to_csv(artifact_dir / "accuracy_matrix.csv", index=False)

    with open(artifact_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Extra method-specific artifacts
    if extra is not None:
        fname = "fisher_stats.json" if "per_param" in extra or "n_tasks_consolidated" in extra else "mask_artifacts.json"
        with open(artifact_dir / fname, "w") as f:
            json.dump(extra, f, indent=2)

    # Long-format per-epoch accuracy
    n_tasks, epochs_per_task, _ = epoch_curves.shape
    rows = []
    for t in range(n_tasks):
        for e in range(epochs_per_task):
            for i in range(t + 1):
                rows.append({"train_task": t, "epoch": e, "eval_task": i,
                              "accuracy": float(epoch_curves[t, e, i])})
    pd.DataFrame(rows).to_csv(artifact_dir / "epoch_curves.csv", index=False)

    _save_plots(artifact_dir, acc_matrix, epoch_curves, cfg, analysis)


def _save_plots(
    artifact_dir: Path,
    acc_matrix: np.ndarray,
    epoch_curves: np.ndarray,
    cfg: dict,
    analysis: dict | None,
) -> None:
    method = cfg.get("method", "sequential")
    n_tasks, epochs_per_task, _ = epoch_curves.shape

    # Accuracy vs task (line plot)
    fig, ax = plt.subplots(figsize=(7, 4))
    for i in range(n_tasks):
        xs = list(range(i, n_tasks))
        ys = [acc_matrix[t, i] for t in xs]
        ax.plot(xs, ys, marker="o", label=f"Task {i+1}")
    ax.set_xlabel("Task trained up to")
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Accuracy per task — {method}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_vs_task.png", dpi=150)
    plt.close(fig)

    # Accuracy matrix heatmap
    fig, ax = plt.subplots(figsize=(max(4, n_tasks), max(3, n_tasks)))
    masked = np.where(np.tril(np.ones_like(acc_matrix, dtype=bool)), acc_matrix, np.nan)
    im = ax.imshow(masked, vmin=0, vmax=1, cmap="viridis")
    plt.colorbar(im, ax=ax)
    ax.set_xlabel("Eval task")
    ax.set_ylabel("Train task")
    ax.set_title(f"Accuracy matrix — {method}")
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_matrix_heatmap.png", dpi=150)
    plt.close(fig)

    # Forgetting bar plot
    f_per_task = []
    for i in range(n_tasks - 1):
        peak = acc_matrix[i:, i].max()
        f_per_task.append(peak - acc_matrix[-1, i])
    if f_per_task:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.bar(range(1, len(f_per_task) + 1), f_per_task)
        ax.set_xlabel("Task")
        ax.set_ylabel("Forgetting")
        ax.set_title(f"Per-task forgetting — {method}")
        fig.tight_layout()
        fig.savefig(artifact_dir / "forgetting_bar.png", dpi=150)
        plt.close(fig)

    # Per-epoch accuracy curves
    fig, axes = plt.subplots(1, n_tasks, figsize=(4 * n_tasks, 4), sharey=True)
    if n_tasks == 1:
        axes = [axes]
    for t, ax in enumerate(axes):
        for i in range(t + 1):
            ys = epoch_curves[t, :, i]
            ax.plot(range(1, epochs_per_task + 1), ys, marker=".", label=f"Task {i+1}")
        ax.set_title(f"Train task {t+1}")
        ax.set_xlabel("Epoch")
        if t == 0:
            ax.set_ylabel("Test accuracy")
        ax.legend(fontsize=6)
    fig.suptitle(f"Per-epoch accuracy — {method}")
    fig.tight_layout()
    fig.savefig(artifact_dir / "accuracy_vs_epoch.png", dpi=150)
    plt.close(fig)

    # Analysis plots (M4)
    if analysis is not None:
        _save_analysis_plots(artifact_dir, analysis, n_tasks)


def _save_analysis_plots(artifact_dir: Path, analysis: dict, n_tasks: int) -> None:
    import pandas as pd

    # RSA heatmaps per layer
    rsa = analysis.get("rsa", {})
    for l, mat in rsa.items():
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        plt.colorbar(im, ax=ax)
        ax.set_title(f"RSA layer {l}")
        ax.set_xlabel("Task")
        ax.set_ylabel("Task")
        fig.tight_layout()
        fig.savefig(artifact_dir / f"rsa_layer{l}.png", dpi=150)
        plt.close(fig)

    # Fisher overlap: summarize by layer (mean across param groups in each layer)
    fisher_ov = analysis.get("fisher_overlap", {})
    if fisher_ov:
        # Group by layer index from param name
        layer_means: dict[str, list] = {}
        for name, mat in fisher_ov.items():
            parts = name.split(".")
            layer_key = parts[0] if len(parts) > 1 else name
            layer_means.setdefault(layer_key, []).append(mat)
        rows_fo = []
        for lk, mats in layer_means.items():
            avg = np.mean(mats, axis=0)
            for i in range(n_tasks):
                for j in range(i + 1, n_tasks):
                    rows_fo.append({"layer": lk, "task_i": i, "task_j": j, "overlap": avg[i, j]})
        if rows_fo:
            df_fo = pd.DataFrame(rows_fo)
            df_fo.to_csv(artifact_dir / "fisher_overlap.csv", index=False)


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tasks = _build_tasks(cfg)
    model = _build_model(cfg, device)

    # Random baseline for forward transfer.
    # For masked models, pre-register task masks (init=zeros → sigmoid=0.5) so
    # forward() has a valid task_id key; run_overlap's add_task is idempotent.
    if hasattr(model, "task_alphas"):
        for t in range(len(tasks)):
            model.add_task(t, device)
    model.eval()
    with torch.no_grad():
        random_acc_list = []
        for t, task in enumerate(tasks):
            if hasattr(model, "_current_task"):
                model._current_task = t
            random_acc_list.append(eval_accuracy(model, task["test"], device))
        random_acc = np.array(random_acc_list, dtype=np.float32)
    if hasattr(model, "_current_task"):
        model._current_task = 0

    result, extra = _run_method(model, tasks, cfg, device)

    acc_matrix, _, zero_shot_acc = result
    metrics = {
        "method": cfg.get("method", "sequential"),
        "dataset": cfg.get("dataset", "split_mnist"),
        "final_average_accuracy": final_average_accuracy(acc_matrix),
        "forgetting": forgetting(acc_matrix),
        "forward_transfer": forward_transfer(zero_shot_acc, random_acc),
    }

    print("\nAccuracy matrix (rows=task trained, cols=task evaluated):")
    print(np.round(acc_matrix, 4))
    print("\nMetrics:", metrics)

    # Optional post-hoc analysis (M4)
    analysis = None
    if cfg.get("run_analysis", False):
        print("\nRunning post-hoc analysis...")
        analysis = _run_analysis(model, tasks, device, len(tasks))

    artifact_dir = Path(cfg.get("artifact_dir", "artifacts/run"))
    save_artifacts(artifact_dir, cfg, result, metrics, extra, analysis)
    print(f"\nArtifacts saved to {artifact_dir}/")


def _collect_fisher_stats(
    model: nn.Module,
    tasks: list[dict],
    cfg: dict,
    device: torch.device,
) -> dict:
    stats_ewc = EWC(
        lambda_ewc=cfg.get("lambda_ewc", 400.0),
        n_fisher_batches=cfg.get("n_fisher_batches", 50),
    )
    for task in tasks:
        stats_ewc.consolidate(model, task["train"], device)
    return stats_ewc.fisher_stats()


if __name__ == "__main__":
    main()
