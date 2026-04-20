"""M5 comparison table + auto-generated markdown summary.

Usage:
  python -m scripts.m5_report \\
    --split  <seq_dir> <ewc_dir> <ov_uni_dir> <ov_hier_dir> <ewc_ov_dir> <ov_rev_dir> \\
    --permuted <seq_dir> <ewc_dir> <ov_uni_dir> <ov_hier_dir> <ewc_ov_dir> <ov_rev_dir> \\
    --out summary_m5.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


METHOD_LABELS = {
    "sequential":           "Sequential",
    "ewc":                  "EWC",
    "overlap_uniform":      "Overlap (uniform)",
    "overlap_hierarchical": "Overlap (hier.)",
    "ewc_overlap":          "EWC + Overlap",
    "overlap_reversed":     "Overlap (reversed)",
}

COL_W = [22, 18, 18, 14]


def _load(run_dir: str) -> dict | None:
    p = Path(run_dir)
    summary = p / "summary.json"
    seed_dir = p / "seed_42"
    if not summary.exists() or not seed_dir.exists():
        return None
    s = json.load(open(summary))
    m = json.load(open(seed_dir / "metrics.json"))
    mask_path = seed_dir / "mask_artifacts.json"
    masks = json.load(open(mask_path)) if mask_path.exists() else None
    return {
        "method": m.get("method", p.name),
        "faa":    s["final_average_accuracy"]["mean"],
        "faa_s":  s["final_average_accuracy"]["std"],
        "fgt":    s["forgetting"]["mean"],
        "fgt_s":  s["forgetting"]["std"],
        "fwt":    s["forward_transfer"]["mean"],
        "fwt_s":  s["forward_transfer"]["std"],
        "masks":  masks,
    }


def _table(rows: list[dict], title: str) -> str:
    sep = "─"
    w0, w1, w2, w3 = COL_W
    total = w0 + w1 + w2 + w3 + 3 * 2 + 2
    lines = [
        f"\n### {title}",
        "```",
        f"{'Method':<{w0}}  {'FAA ↑':<{w1}}  {'Forgetting ↓':<{w2}}  {'FWT':<{w3}}",
        sep * total,
    ]
    for r in rows:
        method = METHOD_LABELS.get(r["method"], r["method"])
        faa  = f"{r['faa']:.4f}±{r['faa_s']:.4f}"
        fgt  = f"{r['fgt']:.4f}±{r['fgt_s']:.4f}"
        fwt  = f"{r['fwt']:.4f}±{r['fwt_s']:.4f}"
        lines.append(f"{method:<{w0}}  {faa:<{w1}}  {fgt:<{w2}}  {fwt:<{w3}}")
    lines.append("```")
    return "\n".join(lines)


def _mask_overlap_with_depth(rows: list[dict]) -> str | None:
    """Check whether cosine overlap between consecutive tasks decreases with layer depth."""
    for r in rows:
        if r.get("masks") and r["masks"].get("pairwise_overlap"):
            po = r["masks"]["pairwise_overlap"]
            if not po:
                continue
            pair_key = sorted(po.keys())[0]
            vals = po[pair_key]
            if len(vals) < 2:
                continue
            decreasing = all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))
            trend = "decreases" if decreasing else "does not consistently decrease"
            method = METHOD_LABELS.get(r["method"], r["method"])
            return f"For **{method}**, pairwise mask overlap {trend} with layer depth (layer values: {[round(v,3) for v in vals]})."
    return None


def _narrative(split_rows: list[dict], perm_rows: list[dict]) -> str:
    parts: list[str] = []

    def best(rows, key, higher_better=True):
        ranked = sorted(rows, key=lambda r: r[key], reverse=higher_better)
        return ranked[0] if ranked else None

    if perm_rows:
        b_faa = best(perm_rows, "faa")
        b_fgt = best(perm_rows, "fgt", higher_better=False)
        parts.append(f"- **Best FAA (Permuted MNIST):** {METHOD_LABELS.get(b_faa['method'], b_faa['method'])} "
                     f"({b_faa['faa']:.4f}±{b_faa['faa_s']:.4f})")
        parts.append(f"- **Least forgetting (Permuted MNIST):** {METHOD_LABELS.get(b_fgt['method'], b_fgt['method'])} "
                     f"({b_fgt['fgt']:.4f}±{b_fgt['fgt_s']:.4f})")

        hier = next((r for r in perm_rows if r["method"] == "overlap_hierarchical"), None)
        rev  = next((r for r in perm_rows if r["method"] == "overlap_reversed"), None)
        if hier and rev:
            delta_faa = hier["faa"] - rev["faa"]
            verdict = "hurt" if delta_faa > 0.005 else ("matched" if abs(delta_faa) < 0.005 else "outperformed")
            parts.append(f"- **Reversed vs normal hierarchy:** reversed {verdict} normal "
                         f"(ΔFAA={delta_faa:+.4f}, ΔForgetting={hier['fgt']-rev['fgt']:+.4f})")

        ewc    = next((r for r in perm_rows if r["method"] == "ewc"), None)
        ewc_ov = next((r for r in perm_rows if r["method"] == "ewc_overlap"), None)
        hier   = next((r for r in perm_rows if r["method"] == "overlap_hierarchical"), None)
        if ewc and ewc_ov and hier:
            parts.append(f"- **EWC+Overlap vs EWC alone:** ΔFAA={ewc_ov['faa']-ewc['faa']:+.4f}, "
                         f"ΔForgetting={ewc_ov['fgt']-ewc['fgt']:+.4f}")
            parts.append(f"- **EWC+Overlap vs Overlap(hier.) alone:** ΔFAA={ewc_ov['faa']-hier['faa']:+.4f}, "
                         f"ΔForgetting={ewc_ov['fgt']-hier['fgt']:+.4f}")

    all_rows = perm_rows + split_rows
    depth_note = _mask_overlap_with_depth(all_rows)
    if depth_note:
        parts.append(f"- **Mask overlap with depth:** {depth_note}")

    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split",    nargs="+", default=[])
    parser.add_argument("--permuted", nargs="+", default=[])
    parser.add_argument("--out", default="artifacts/summary_m5.md")
    args = parser.parse_args()

    split_rows   = [r for d in args.split    if (r := _load(d)) is not None]
    perm_rows    = [r for d in args.permuted if (r := _load(d)) is not None]

    md_lines = ["# M5 Results Summary\n"]

    if split_rows:
        md_lines.append(_table(split_rows, "Split MNIST — 5 tasks, class-incremental (3 seeds)"))
    if perm_rows:
        md_lines.append(_table(perm_rows, "Permuted MNIST — 10 tasks (3 seeds)"))

    md_lines.append("\n### Key findings\n")
    md_lines.append(_narrative(split_rows, perm_rows))

    md = "\n".join(md_lines)
    print(md)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"\n[saved to {out}]")


if __name__ == "__main__":
    main()
