"""Run a config over multiple seeds and aggregate metrics.

Usage: python -m scripts.multi_run --config <yaml> --seeds 42 43 44 --out <dir>

Appends each run to results/results_db.csv for cross-experiment analysis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DB_PATH = Path(__file__).parent.parent / "results" / "results_db.csv"

DB_COLS = [
    "run_id", "method", "dataset", "n_tasks", "multihead", "gradient_masking",
    "epochs_per_task", "lr", "lambda_ewc", "lambda_overlap", "rho_max", "rho_min",
    "seed", "faa", "forgetting", "fwt", "artifact_dir",
]


def _append_to_db(cfg: dict, metrics: dict, seed: int, run_dir: str) -> None:
    cfg_hash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + f"_{cfg_hash}"
    method = cfg.get("method", "sequential")
    row = {
        "run_id":            run_id,
        "method":            method,
        "dataset":           cfg.get("dataset", ""),
        "n_tasks":           cfg.get("n_tasks", ""),
        "multihead":         cfg.get("multihead", False),
        "gradient_masking":  method in ("overlap_hier_gm", "overlap_hier_gm_ewc"),
        "epochs_per_task":   cfg.get("epochs_per_task", ""),
        "lr":                cfg.get("lr", ""),
        "lambda_ewc":        cfg.get("lambda_ewc", 0),
        "lambda_overlap":    cfg.get("lambda_overlap", 0),
        "rho_max":           cfg.get("rho_max", 0),
        "rho_min":           cfg.get("rho_min", 0),
        "seed":              seed,
        "faa":               metrics.get("final_average_accuracy", ""),
        "forgetting":        metrics.get("forgetting", ""),
        "fwt":               metrics.get("forward_transfer", ""),
        "artifact_dir":      run_dir,
    }
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame([row])[DB_COLS]
    if DB_PATH.exists():
        df_new.to_csv(DB_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(DB_PATH, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = yaml.safe_load(Path(args.config).read_text())
    metrics_list = []

    for seed in args.seeds:
        run_dir = out_dir / f"seed_{seed}"
        cfg = {**base_cfg, "seed": seed, "artifact_dir": str(run_dir)}
        tmp_cfg = out_dir / f"_tmp_{seed}.yaml"
        tmp_cfg.write_text(yaml.dump(cfg))
        try:
            subprocess.run(
                [sys.executable, "-m", "scripts.train", "--config", str(tmp_cfg)],
                check=True,
            )
        finally:
            tmp_cfg.unlink(missing_ok=True)

        with open(run_dir / "metrics.json") as f:
            m = json.load(f)
        m["seed"] = seed
        metrics_list.append(m)
        _append_to_db(base_cfg, m, seed, str(run_dir))

    df = pd.DataFrame(metrics_list)
    metric_cols = ["final_average_accuracy", "forgetting", "forward_transfer"]

    summary: dict = {}
    for col in metric_cols:
        if col in df.columns:
            summary[col] = {"mean": round(float(df[col].mean()), 6),
                            "std": round(float(df[col].std()), 6)}

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    df.to_csv(out_dir / "all_metrics.csv", index=False)

    print(f"\n{'Metric':<32} {'Mean':>8} {'Std':>8}")
    print("-" * 52)
    for col, s in summary.items():
        print(f"{col:<32} {s['mean']:>8.4f} {s['std']:>8.4f}")
    print(f"\nSummary saved to {out_dir}/")


if __name__ == "__main__":
    main()
