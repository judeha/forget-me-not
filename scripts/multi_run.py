"""Run a config over multiple seeds and aggregate metrics.

Usage: python -m scripts.multi_run --config <yaml> --seeds 42 43 44 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


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
