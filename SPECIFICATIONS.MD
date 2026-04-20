# Specifications

## Agent: M1 (Sequential Baseline)

### Design decisions
- **Task stream**: Split MNIST — 5 binary tasks, classes [0,1],[2,3],[4,5],[6,7],[8,9]. Normalization baked into TensorDataset (mean=0.1307, std=0.3081), no transform pipeline needed.
- **Model**: Shared-head MLP (784 → hidden_sizes → 10). Single output head for all tasks; no task ID given at inference.
- **Training loop**: `run_sequential` in `src/methods/sequential.py`. Returns `TrainResult(acc_matrix, epoch_curves, zero_shot_acc)` (NamedTuple).
  - `acc_matrix[t, i]` — accuracy on task i after finishing task t
  - `epoch_curves[t, e, i]` — accuracy on task i after epoch e of task t (NaN where i > t)
  - `zero_shot_acc[i]` — accuracy on task i right before training on it
- **Optimizer**: Fresh Adam per task (lr from config). Rationale: no cross-task optimizer state leakage for the baseline.
- **Artifacts**: config.yaml, accuracy_matrix.csv, epoch_curves.csv (long-format), metrics.json, accuracy_vs_task.png, accuracy_vs_epoch.png — all under `artifact_dir`.
- **Seeding**: `set_seed` in `src/utils/seed.py` — seeds Python, NumPy, PyTorch, and sets cuDNN deterministic.

---

## Agent: M2 (EWC Baseline)

### Design decisions
- **EWC implementation**: `EWC` class in `src/methods/ewc.py`.
  - Per-task Fisher storage: list of `(fisher_dict, theta_star_dict)` pairs, one per consolidated task. Avoids the approximation error of online (accumulated) EWC; memory is O(T × |params|) which is fine for Split MNIST scale.
  - Fisher estimation: empirical Fisher via squared gradients of NLL w.r.t. model's own sampled labels (not true labels). Averaged over `n_fisher_batches` minibatches.
  - `consolidate(model, loader, device)` called **after** training on each task.
  - `penalty(model)` sums `(λ/2) * Σ_t Σ_j F_j^t * (θ_j - θ_star_j^t)^2` over all past tasks.
- **`run_ewc`**: mirrors `run_sequential` signature and returns the same `TrainResult` NamedTuple. Optimizer reset per task; EWC penalty added to CE loss each step.
- **Method dispatch**: `scripts/train.py` reads `method: sequential|ewc` from config. Random baseline (random-init accuracy on all tasks) computed before training for forward transfer.
- **Forward transfer** (`src/metrics/continual.py`): `forward_transfer(zero_shot_acc, random_acc)` = mean over tasks i>0 of `[zero_shot_acc[i] - random_acc[i]]`. Positive = prior tasks help; negative = interference.
- **Fisher stats artifact**: `fisher_stats.json` saved for EWC runs — per-parameter mean/max Fisher and sparsity (fraction < 1e-10).
- **Permuted MNIST**: task stream in `src/data/permuted_mnist.py` — 10 tasks, fixed per-task pixel permutations seeded from config seed. Task stream only; no full experiment yet.
- **Configs added**: `configs/split_mnist_ewc.yaml` (λ=400, n_fisher_batches=50), `configs/permuted_mnist_sequential.yaml`.

### Known limitations / future work
- EWC with tiny subsets (<200 examples/task) shows no benefit over sequential — Fisher estimates are noisy and the penalty doesn't overcome gradient noise. Full dataset shows expected forgetting reduction.
- `forward_transfer` returns 0.0 for single-task runs (guard on `len < 2`).
