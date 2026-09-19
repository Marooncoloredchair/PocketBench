#!/usr/bin/env python3
"""Paired Wilcoxon signed-rank tests for the crop_radius_minus_1.5 perturbation.

Thin multi-panel wrapper around ``sbdd_robust.report.crop_paired_wilcoxon`` (the shared
source of truth, also exposed as ``pocketbench crop-test``). Pairs every pocket's
crop_radius_minus_1.5 condition against its own ``original`` condition and tests ΔQED and
ΔSA on the real47 and real100 DiffSBDD panels.

Output: paper/crop_radius_wilcoxon.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.report import INVARIANT_TAGS, crop_paired_wilcoxon, filter_model, tag_col

REAL100_PATH = Path("D:/metrics_per_condition__runreal100.csv")
RESULTS_DIR = _ROOT / "data" / "results"


def find_diffsbdd_real47() -> Path:
    best: tuple[int, Path] | None = None
    for p in sorted(RESULTS_DIR.glob("metrics_per_condition__*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model_name" not in df.columns or "pocket_id" not in df.columns:
            continue
        dfd = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]
        if dfd.empty:
            continue
        if not set(INVARIANT_TAGS).issubset(set(dfd[tag_col(dfd)].astype(str))):
            continue
        n_pockets = dfd["pocket_id"].astype(str).nunique()
        if 40 <= n_pockets <= 55:
            score = abs(n_pockets - 47)
            if best is None or score < best[0]:
                best = (score, p)
    if best is None:
        raise FileNotFoundError("Could not auto-detect a DiffSBDD real47 metrics CSV in data/results/")
    return best[1]


def main() -> None:
    paper = _ROOT / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    real47_path = find_diffsbdd_real47()
    datasets["real47"] = filter_model(pd.read_csv(real47_path), "diffsbdd")
    print(f"real47 source: {real47_path}")
    if REAL100_PATH.is_file():
        datasets["real100"] = filter_model(pd.read_csv(REAL100_PATH), "diffsbdd")
        print(f"real100 source: {REAL100_PATH}")
    else:
        print(f"WARN: real100 not found at {REAL100_PATH}; skipping.")

    out = pd.concat(
        [crop_paired_wilcoxon(df, dataset=name) for name, df in datasets.items()],
        ignore_index=True,
    )
    out_path = paper / "crop_radius_wilcoxon.csv"
    out.to_csv(out_path, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n" + "=" * 78)
    print("PAIRED WILCOXON -- crop_radius_minus_1.5 vs original (DiffSBDD)")
    print("=" * 78)
    print(out.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
