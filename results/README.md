# Results Database

`results_db.csv` — append-only log of every multi-seed run.

## Schema

| Column | Description |
|---|---|
| run_id | ISO timestamp + config hash (8 chars) |
| method | e.g. sequential, ewc, overlap_hier_gm |
| dataset | split_mnist, permuted_mnist, split_cifar100 |
| n_tasks | number of tasks |
| multihead | true/false |
| gradient_masking | true/false |
| epochs_per_task | |
| lr | learning rate |
| lambda_ewc | EWC regularization (0 if unused) |
| lambda_overlap | overlap loss weight (0 if unused) |
| rho_max | max target overlap (0 if unused) |
| rho_min | min target overlap (0 if unused) |
| seed | individual seed |
| faa | final average accuracy |
| forgetting | forgetting metric |
| fwt | forward transfer |
| artifact_dir | path to per-seed artifacts |

## Querying

```bash
python -m scripts.query_results --method ewc --dataset permuted_mnist
python -m scripts.query_results --dataset split_cifar100 --n-tasks 20
python -m scripts.query_results --summary  # aggregated mean/std per (method, dataset, n_tasks)
```
