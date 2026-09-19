#!/usr/bin/env python3
"""Summarize Pocket2Mol real47 coverage before/after adapter fix."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def summarize(path: Path, label: str) -> dict:
    df = pd.read_csv(path)
    orig = df[df["perturbation_tag"].astype(str) == "original"]
    n = orig["pocket_id"].nunique()
    ok = int((pd.to_numeric(orig["validity"], errors="coerce") > 0).sum())
    return {
        "label": label,
        "path": str(path),
        "n_pockets": n,
        "n_generating": ok,
        "coverage_pct": round(100.0 * ok / n, 1) if n else 0.0,
        "mean_validity": round(float(orig["validity"].mean()), 3) if n else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full.csv"))
    ap.add_argument("--updated", default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv"))
    ap.add_argument("--out", default=str(_ROOT / "paper/pocket2mol_real47_coverage_v2.txt"))
    args = ap.parse_args()
    rows = []
    for label, p in [("baseline_v1", args.baseline), ("fixed_adapter_v2", args.updated)]:
        path = Path(p)
        if path.is_file():
            rows.append(summarize(path, label))
    lines = ["Pocket2Mol real47 original-condition coverage", ""]
    for r in rows:
        lines.append(
            f"{r['label']}: {r['n_generating']}/{r['n_pockets']} pockets ({r['coverage_pct']}%) "
            f"mean_validity={r['mean_validity']}  [{Path(r['path']).name}]"
        )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
