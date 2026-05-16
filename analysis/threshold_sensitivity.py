#!/usr/bin/env python3
"""
Recompute brittleness_rate at several std thresholds without re-running generation.

Uses the same logic as ``sbdd_robust.metrics.robustness_score.flag_invariant_brittleness`` on
``metrics_per_condition__run*.csv`` (or ``metrics_flagged__*`` with brittle columns stripped).

``robustness_summary__*.csv`` is optional: cross-checks rates (should match) via *_inv_std columns.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root = parent of analysis/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged
from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

DEFAULT_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)

DEFAULT_INVARIANT_TAGS = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)


def load_metrics_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ("brittle_invariant", "brittleness_note"):
        if col in df.columns:
            df = df.drop(columns=[col])
    return df


def brittleness_vs_thresholds(
    df: pd.DataFrame,
    thresholds,
    invariant_tags,
) -> pd.DataFrame:
    rows: list[dict] = []
    for t in thresholds:
        flagged = flag_invariant_brittleness(
            df,
            invariant_tags=invariant_tags,
            original_tag="original",
            metric_std_threshold=float(t),
        )
        stats = brittleness_rate_from_flagged(flagged)
        rows.append(
            {
                "threshold": float(t),
                "n_brittle": int(stats["brittle_pairs"]),
                "total": int(stats["total_pairs"]),
                "brittleness_rate": float(stats["brittleness_rate"]),
            }
        )
    return pd.DataFrame(rows)


def brittleness_from_summary(summary_df: pd.DataFrame, threshold: float) -> tuple[int, int, float]:
    """
    Each * _inv_std column is np.std(metric across invariant perturbations) for that pocket.
    Brittle if any such std > threshold (same rule as flagging).
    """
    inv_std_cols = [c for c in summary_df.columns if c.endswith("_inv_std")]
    if not inv_std_cols:
        raise ValueError("No *_inv_std columns in robustness summary")

    n_brittle = 0
    for _, row in summary_df.iterrows():
        if any(
            pd.notna(row[c]) and float(row[c]) > threshold
            for c in inv_std_cols
        ):
            n_brittle += 1
    total = int(summary_df.shape[0])
    rate = float(n_brittle / total) if total else 0.0
    return n_brittle, total, rate


def validate_against_summary(
    summary_df: pd.DataFrame,
    table: pd.DataFrame,
    thresholds,
    invariant_tags,
    rtol: float = 1e-9,
) -> None:
    for t in thresholds:
        n1 = int(table.loc[table["threshold"] == float(t), "n_brittle"].iloc[0])
        nb, tot, _ = brittleness_from_summary(summary_df, float(t))
        if n1 != nb:
            raise RuntimeError(
                f"Mismatch at threshold {t}: metrics-based n_brittle={n1}, "
                f"summary-based n_brittle={nb} (total summary rows={tot})"
            )


def plot_curve(table: pd.DataFrame, out_pdf: Path) -> None:
    """Figure 1: DiffSBDD brittleness vs τ (blue, 300 dpi, white background)."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    ax.set_facecolor("white")
    ax.plot(
        table["threshold"],
        table["brittleness_rate"],
        marker="o",
        color="#1f77b4",
        markeredgecolor="#d62728",
        markeredgewidth=1.2,
        linewidth=2.0,
        markersize=7,
    )
    ax.set_xlabel(r"Threshold $\tau$ on metric SD (invariant conditions, unitless)")
    ax.set_ylabel("Brittleness rate (dimensionless)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
    fig.tight_layout()
    fig.savefig(
        out_pdf,
        format="pdf",
        bbox_inches="tight",
        dpi=300,
        facecolor="white",
    )
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="metrics_per_condition__run*.csv or metrics_flagged__* (brittle cols ignored)",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional robustness_summary__*.csv; cross-check n_brittle vs metrics path",
    )
    p.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=list(DEFAULT_THRESHOLDS),
        help="Std thresholds (default: %(default)s)",
    )
    p.add_argument(
        "--invariant-tags",
        type=str,
        nargs="+",
        default=list(DEFAULT_INVARIANT_TAGS),
        help="Perturbation tags treated as invariants (must match the benchmark YAML)",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        default=Path("paper/threshold_sensitivity.csv"),
    )
    p.add_argument(
        "--out-figure",
        type=Path,
        default=Path("paper/figures/threshold_sensitivity.pdf"),
    )
    args = p.parse_args(argv)

    df = load_metrics_csv(args.metrics)
    table = brittleness_vs_thresholds(
        df,
        thresholds=args.thresholds,
        invariant_tags=args.invariant_tags,
    )

    if args.summary is not None:
        summary_df = pd.read_csv(args.summary)
        validate_against_summary(
            summary_df,
            table,
            thresholds=args.thresholds,
            invariant_tags=args.invariant_tags,
        )
        print("OK: robustness_summary cross-check passed for all thresholds.")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv.resolve())

    plot_curve(table, args.out_figure)
    print("Wrote", args.out_figure.resolve())


if __name__ == "__main__":
    main()
