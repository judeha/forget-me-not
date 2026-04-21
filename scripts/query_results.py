"""Query and display results_db.csv.

Usage:
  python -m scripts.query_results                          # full table
  python -m scripts.query_results --summary                # mean/std by group
  python -m scripts.query_results --method ewc             # filter by method
  python -m scripts.query_results --dataset split_cifar100 --n-tasks 20
  python -m scripts.query_results --multihead              # only multihead runs
  python -m scripts.query_results --gradient-masking       # only gm runs
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).parent.parent / "results" / "results_db.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method",           default=None)
    parser.add_argument("--dataset",          default=None)
    parser.add_argument("--n-tasks",          type=int, default=None)
    parser.add_argument("--multihead",        action="store_true")
    parser.add_argument("--gradient-masking", action="store_true")
    parser.add_argument("--summary",          action="store_true",
                        help="Aggregate mean±std per (method,dataset,n_tasks,multihead)")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print("No results_db.csv found yet.")
        return

    df = pd.read_csv(DB_PATH)

    if args.method:
        df = df[df["method"] == args.method]
    if args.dataset:
        df = df[df["dataset"] == args.dataset]
    if args.n_tasks is not None:
        df = df[df["n_tasks"] == args.n_tasks]
    if args.multihead:
        df = df[df["multihead"] == True]
    if args.gradient_masking:
        df = df[df["gradient_masking"] == True]

    if df.empty:
        print("No matching rows.")
        return

    if args.summary:
        group_cols = ["method", "dataset", "n_tasks", "multihead", "gradient_masking"]
        agg = df.groupby(group_cols)[["faa", "forgetting", "fwt"]].agg(["mean", "std"])
        agg.columns = ["faa_mean", "faa_std", "fgt_mean", "fgt_std", "fwt_mean", "fwt_std"]
        agg = agg.reset_index()

        w = [28, 14, 7, 9, 17, 20, 20, 14]
        header = (f"{'Method':<{w[0]}}  {'Dataset':<{w[1]}}  {'Tasks':>{w[2]}}  "
                  f"{'MH':>{w[3]}}  {'FAA ↑':<{w[4]}}  {'Forgetting ↓':<{w[5]}}  {'FWT':<{w[6]}}")
        sep = "─" * (sum(w) + 2 * (len(w) - 1))
        print(f"\n{sep}")
        print(header)
        print(sep)
        for _, row in agg.iterrows():
            faa = f"{row.faa_mean:.4f}±{row.faa_std:.4f}"
            fgt = f"{row.fgt_mean:.4f}±{row.fgt_std:.4f}"
            fwt = f"{row.fwt_mean:.4f}±{row.fwt_std:.4f}"
            print(f"{str(row.method):<{w[0]}}  {str(row.dataset):<{w[1]}}  "
                  f"{int(row.n_tasks):>{w[2]}}  {str(row.multihead):>{w[3]}}  "
                  f"{faa:<{w[4]}}  {fgt:<{w[5]}}  {fwt:<{w[6]}}")
        print(f"{sep}\n")
    else:
        display_cols = ["method", "dataset", "n_tasks", "multihead",
                        "gradient_masking", "seed", "faa", "forgetting", "fwt"]
        print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
