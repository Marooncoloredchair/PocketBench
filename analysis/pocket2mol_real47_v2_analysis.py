#!/usr/bin/env python3
"""Coverage + normalized brittleness + crop Wilcoxon for Pocket2Mol real47 v2 panel.

Mirrors the DiffSBDD analyses in ``analysis/recompute_normalized_brittleness.py`` and
``analysis/crop_radius_wilcoxon.py``, using the same ``sbdd_robust.report`` helpers.

Output: paper/pocket2mol_real47_v2_analysis.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.report import (
    NORMALIZED_METRICS,
    crop_paired_wilcoxon,
    filter_model,
    normalized_brittleness,
)

V2_PATH = _ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv"
V1_PATH = _ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full.csv"
OUT_PATH = _ROOT / "paper/pocket2mol_real47_v2_analysis.csv"
COVERAGE_TXT = _ROOT / "paper/pocket2mol_real47_coverage_v2.txt"


def coverage_stats(path: Path, label: str) -> dict:
    df = pd.read_csv(path)
    df = filter_model(df, "pocket2mol")
    orig = df[df["perturbation_tag"].astype(str) == "original"].copy()
    n_pockets = int(orig["pocket_id"].nunique())
    n_valid = pd.to_numeric(orig["n_valid"], errors="coerce").fillna(0)
    covered = orig[n_valid > 0]
    n_generating = int(len(covered))
    mean_validity = (
        round(float(covered["validity"].mean()), 4) if n_generating else float("nan")
    )
    return {
        "analysis": "coverage",
        "label": label,
        "n_pockets": n_pockets,
        "n_generating": n_generating,
        "coverage_fraction": round(n_generating / n_pockets, 4) if n_pockets else 0.0,
        "mean_validity_covered": mean_validity,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v2", type=Path, default=V2_PATH)
    ap.add_argument("--v1", type=Path, default=V1_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args()

    if not args.v2.is_file():
        raise SystemExit(f"Missing v2 panel: {args.v2}")

    df = filter_model(pd.read_csv(args.v2), "pocket2mol")
    n_rows = len(df)
    n_pockets = df["pocket_id"].nunique()
    if n_rows != 235 or n_pockets != 47:
        raise SystemExit(
            f"v2 panel failed sanity check: rows={n_rows} (expected 235), "
            f"pockets={n_pockets} (expected 47)"
        )

    rows: list[dict] = []

    # --- coverage: baseline v1 vs fixed v2 ---
    baseline = coverage_stats(args.v1, "baseline_v1_adapter_bug") if args.v1.is_file() else None
    fixed = coverage_stats(args.v2, "fixed_adapter_v2")
    rows.append(fixed)
    if baseline:
        rows.append(baseline)
        gap_total = 47 - baseline["n_generating"]
        gap_closed = fixed["n_generating"] - baseline["n_generating"]
        rows.append(
            {
                "analysis": "coverage_comparison",
                "label": "adapter_fix_attribution",
                "baseline_generating": baseline["n_generating"],
                "fixed_generating": fixed["n_generating"],
                "baseline_coverage_fraction": baseline["coverage_fraction"],
                "fixed_coverage_fraction": fixed["coverage_fraction"],
                "pockets_recovered_by_fix": gap_closed,
                "fraction_of_coverage_gap_closed": (
                    round(gap_closed / gap_total, 4) if gap_total else float("nan")
                ),
                "remaining_zero_valid_original": 47 - fixed["n_generating"],
            }
        )

    # --- normalized brittleness (identical to DiffSBDD pipeline) ---
    brit = normalized_brittleness(
        df, dataset="pocket2mol_real47_v2", metrics=NORMALIZED_METRICS
    )
    for rec in brit.to_dict("records"):
        rec["analysis"] = "normalized_brittleness"
        rows.append(rec)

    # --- crop-radius Wilcoxon (identical to DiffSBDD pipeline) ---
    wilcox = crop_paired_wilcoxon(df, dataset="pocket2mol_real47_v2")
    for rec in wilcox.to_dict("records"):
        rec["analysis"] = "crop_radius_wilcoxon"
        rows.append(rec)

    out_df = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)

    # Human-readable coverage summary
    lines = [
        "Pocket2Mol real47 original-condition coverage",
        "",
        f"baseline_v1: {baseline['n_generating']}/{baseline['n_pockets']} pockets "
        f"({100 * baseline['coverage_fraction']:.1f}%) "
        f"mean_validity={baseline['mean_validity_covered']:.3f} on covered pockets "
        f"[{args.v1.name}]"
        if baseline
        else "baseline_v1: (missing)",
        f"fixed_adapter_v2: {fixed['n_generating']}/{fixed['n_pockets']} pockets "
        f"({100 * fixed['coverage_fraction']:.1f}%) "
        f"mean_validity={fixed['mean_validity_covered']:.3f} on covered pockets "
        f"[{args.v2.name}]",
    ]
    if baseline:
        gap_closed = fixed["n_generating"] - baseline["n_generating"]
        gap_total = 47 - baseline["n_generating"]
        pct = 100.0 * gap_closed / gap_total if gap_total else 0.0
        lines.extend(
            [
                "",
                f"Adapter fix recovered {gap_closed}/{gap_total} previously failing pockets "
                f"({pct:.1f}% of the 13/47→47 coverage gap). "
                f"Remaining {47 - fixed['n_generating']} pockets still have n_valid=0 on original.",
            ]
        )
    COVERAGE_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\nNormalized brittleness (tau=0.10):")
    b10 = brit[brit["tau"] == 0.10].iloc[0]
    print(
        f"  {int(b10['n_brittle'])}/{int(b10['total'])} brittle "
        f"(rate={b10['brittleness_rate']:.4f})"
    )
    print("\nCrop-radius Wilcoxon:")
    print(wilcox[["metric", "n_pairs", "median_delta", "p_value"]].to_string(index=False))
    print(f"\nWrote {args.out}")
    print(f"Wrote {COVERAGE_TXT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
