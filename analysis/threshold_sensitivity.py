#!/usr/bin/env python3
"""
Recompute brittleness_rate at several std thresholds without re-running generation.

Uses the same logic as ``sbdd_robust.metrics.robustness_score.flag_invariant_brittleness`` on
``metrics_per_condition__run*.csv`` (or ``metrics_flagged__*`` with brittle columns stripped).

Plots DiffSBDD and Pocket2Mol on one axis (blue / orange); Pocket2Mol rates are computed on
**covered** pockets only (original $n_{\\mathrm{valid}}>0$).

``robustness_summary__*.csv`` is optional: cross-checks DiffSBDD rates (should match) via *_inv_std columns.
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
sys.path.insert(0, str(_ROOT / "paper" / "figures"))
from nmi_style import (
    COLOR_DIFFSBDD,
    COLOR_POCKET2MOL,
    figsize_single,
    save_figure,
    setup_rc,
)

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged
from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

# Include τ = 0.05, 0.10, 0.15, 0.20 for alignment with Table 1 plus intermediates for curve shape.
DEFAULT_THRESHOLDS = (0.03, 0.05, 0.08, 0.10, 0.15, 0.20)

# Axes limits (slight padding beyond first/last τ so markers are not clipped).
DEFAULT_XLIM = (0.025, 0.205)
DEFAULT_XTICKS = (0.05, 0.10, 0.15, 0.20)
DEFAULT_FIGSIZE_IN = figsize_single()

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


def _norm_pid(s: pd.Series | np.ndarray) -> pd.Series:
    return s.astype(str).str.strip()


def _good_original_pocket_ids(df: pd.DataFrame) -> set[str]:
    """Pockets with at least one valid molecule on the original (reference) condition."""
    tag_col = "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"
    o = df[_norm_pid(df[tag_col]) == "original"].copy()
    if o.empty:
        return set()
    nv = pd.to_numeric(o["n_valid"], errors="coerce").fillna(0)
    o = o.assign(_nv=nv)
    return set(_norm_pid(o.loc[o["_nv"] > 0, "pocket_id"]))


def _filter_pockets(df: pd.DataFrame, pocket_ids: set[str]) -> pd.DataFrame:
    return df[_norm_pid(df["pocket_id"]).isin(pocket_ids)].copy()


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


def plot_curve(
    table_diffsbdd: pd.DataFrame,
    table_pocket2mol: pd.DataFrame,
    out_pdf: Path,
    *,
    xlim: tuple[float, float] = DEFAULT_XLIM,
    xticks: tuple[float, ...] = DEFAULT_XTICKS,
    figsize: tuple[float, float] = DEFAULT_FIGSIZE_IN,
) -> None:
    """Brittleness vs τ for DiffSBDD and Pocket2Mol on covered pockets."""
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    setup_rc(True)
    fig, ax = plt.subplots(figsize=figsize, facecolor="white")
    ax.set_facecolor("white")
    kw = dict(linewidth=2.0, markersize=6.0, markeredgewidth=1.0)
    ax.plot(
        table_diffsbdd["threshold"],
        table_diffsbdd["brittleness_rate"],
        marker="o",
        color=COLOR_DIFFSBDD,
        markeredgecolor="#004466",
        label=r"DiffSBDD (covered $N=47$)",
        **kw,
    )
    ax.plot(
        table_pocket2mol["threshold"],
        table_pocket2mol["brittleness_rate"],
        marker="s",
        color=COLOR_POCKET2MOL,
        markeredgecolor="#996000",
        label=r"Pocket2Mol (covered $N=13$)",
        **kw,
    )
    ax.set_xlabel(r"Threshold $\tau$ on metric SD (featurization conditions, unitless)")
    ax.set_ylabel("Brittleness rate (dimensionless)")
    ax.set_xlim(xlim)
    ax.set_xticks(list(xticks))
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
    ax.legend(loc="best", framealpha=0.92)
    ax.set_title("Brittleness vs dispersion threshold", fontweight="normal")
    fig.tight_layout()
    save_figure(fig, out_pdf.parent / out_pdf.stem)
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--metrics",
        type=Path,
        required=True,
        help="DiffSBDD metrics_per_condition__run*.csv (brittle cols ignored if present)",
    )
    p.add_argument(
        "--pocket2mol-metrics",
        type=Path,
        default=_ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full.csv",
        help="Pocket2Mol metrics CSV; brittleness is computed on covered pockets only",
    )
    p.add_argument(
        "--summary",
        type=Path,
        default=None,
        help="Optional robustness_summary__*.csv; cross-check n_brittle vs DiffSBDD metrics path",
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

    diffsbdd_path = Path(args.metrics).resolve()
    p2m_path = Path(args.pocket2mol_metrics).resolve()
    if not diffsbdd_path.is_file():
        raise FileNotFoundError(f"DiffSBDD metrics not found: {diffsbdd_path}")
    if not p2m_path.is_file():
        raise FileNotFoundError(f"Pocket2Mol metrics not found: {p2m_path}")

    df_d = load_metrics_csv(diffsbdd_path)
    df_p = load_metrics_csv(p2m_path)
    good_p = _good_original_pocket_ids(df_p)
    df_p_cov = _filter_pockets(df_p, good_p)

    table_d = brittleness_vs_thresholds(
        df_d,
        thresholds=args.thresholds,
        invariant_tags=args.invariant_tags,
    )
    table_p = brittleness_vs_thresholds(
        df_p_cov,
        thresholds=args.thresholds,
        invariant_tags=args.invariant_tags,
    )

    if args.summary is not None:
        summary_df = pd.read_csv(args.summary)
        validate_against_summary(
            summary_df,
            table_d,
            thresholds=args.thresholds,
            invariant_tags=args.invariant_tags,
        )
        print("OK: robustness_summary cross-check passed for all thresholds.")

    merged = table_d.merge(
        table_p,
        on="threshold",
        how="outer",
        suffixes=("_diffsbdd", "_pocket2mol"),
    ).sort_values("threshold")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(args.out_csv, index=False)
    print("Wrote", args.out_csv.resolve())

    plot_curve(table_d, table_p, args.out_figure)
    print("Wrote", args.out_figure.resolve())


if __name__ == "__main__":
    main()
