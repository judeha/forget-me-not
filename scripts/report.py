"""Print a comparison table of metrics across artifact directories.

Usage: python -m scripts.report <dir1> <dir2> ...
       python -m scripts.report artifacts/run_sequential artifacts/run_ewc ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


COLS = [
    ("method",                  28),
    ("dataset",                 14),
    ("final_average_accuracy",  18),
    ("forgetting",              18),
    ("forward_transfer",        18),
]

HEADERS = {
    "method":                "Method",
    "dataset":               "Dataset",
    "final_average_accuracy":"FAA (↑)",
    "forgetting":            "Forgetting (↓)",
    "forward_transfer":      "FWT",
}


def _fmt(val: object, width: int) -> str:
    if isinstance(val, float):
        return f"{val:.4f}"
    return str(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--title", default="Results")
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        mpath = Path(run_dir) / "metrics.json"
        if not mpath.exists():
            # Check for multi-run summary
            spath = Path(run_dir) / "summary.json"
            if spath.exists():
                summary = json.load(open(spath))
                row = {"method": Path(run_dir).name, "dataset": "—"}
                for k, v in summary.items():
                    row[k] = f"{v['mean']:.4f}±{v['std']:.4f}"
                rows.append(row)
            continue
        m = json.load(open(mpath))
        m.setdefault("dataset", Path(run_dir).name)
        rows.append(m)

    if not rows:
        print("No metrics found.")
        return

    # Header
    sep = "─"
    total_w = sum(w + 2 for _, w in COLS) + 1
    print()
    print(f"  {args.title}")
    print("  " + sep * total_w)
    header = "  " + "  ".join(f"{HEADERS.get(c, c):<{w}}" for c, w in COLS)
    print(header)
    print("  " + sep * total_w)
    for row in rows:
        line = "  " + "  ".join(_fmt(row.get(c, "—"), w).ljust(w) for c, w in COLS)
        print(line)
    print("  " + sep * total_w)
    print()


if __name__ == "__main__":
    main()
