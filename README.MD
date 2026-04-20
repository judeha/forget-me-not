# forget-me-not

Continual learning experiments: sequential, EWC, and mask-overlap methods on Split MNIST and Split CIFAR-100.

## Setup

```bash
pip install torch torchvision numpy pandas matplotlib pyyaml scipy pytest
```

MNIST data downloads automatically. For CIFAR-100 it also downloads automatically on first run.

## Running experiments

### Single run
```bash
python -m scripts.train --config configs/split_mnist_sequential.yaml
python -m scripts.train --config configs/split_mnist_ewc.yaml
python -m scripts.train --config configs/split_mnist_overlap_uniform.yaml
python -m scripts.train --config configs/split_mnist_overlap_hierarchical.yaml
```

### CIFAR-100
```bash
python -m scripts.train --config configs/split_cifar100_sequential.yaml
python -m scripts.train --config configs/split_cifar100_ewc.yaml
python -m scripts.train --config configs/split_cifar100_overlap_uniform.yaml
python -m scripts.train --config configs/split_cifar100_overlap_hierarchical.yaml
```

### Multi-seed runs (seeds 42/43/44)
```bash
python -m scripts.multi_run --config configs/split_mnist_sequential.yaml --artifact-dir artifacts/multirun_sequential
python -m scripts.multi_run --config configs/split_mnist_ewc.yaml --artifact-dir artifacts/multirun_ewc
python -m scripts.multi_run --config configs/split_mnist_overlap_uniform.yaml --artifact-dir artifacts/multirun_overlap_uniform
python -m scripts.multi_run --config configs/split_mnist_overlap_hierarchical.yaml --artifact-dir artifacts/multirun_overlap_hierarchical
```

### Comparison report
```bash
python -m scripts.report artifacts/multirun_sequential artifacts/multirun_ewc artifacts/multirun_overlap_uniform artifacts/multirun_overlap_hierarchical
```

## Tests

```bash
python -m pytest tests/ -v
```

## Artifacts

Each run saves to its artifact directory:
- `config.yaml` — config snapshot
- `accuracy_matrix.csv` — A[t,i]: accuracy on task i after training task t
- `epoch_curves.csv` — per-epoch per-task accuracy (long format)
- `metrics.json` — FAA, forgetting, forward transfer
- `accuracy_vs_task.png` — accuracy per task over time
- `accuracy_vs_epoch.png` — per-epoch accuracy curves
- `accuracy_matrix_heatmap.png` — heatmap of accuracy matrix
- `forgetting_bar.png` — per-task forgetting

Multi-seed runs also produce `summary.json` (mean/std over seeds) and `all_metrics.csv`.
