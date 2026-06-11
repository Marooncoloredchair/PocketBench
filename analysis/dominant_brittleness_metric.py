#!/usr/bin/env python3
"""
For each covered pocket, identify which summary metric has the largest cross-condition
standard deviation among metrics that exceed τ (same rule as brittleness flagging).

Produces a per-pocket bar chart (DiffSBDD panel) and optional CSV for supplements.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbdd_robust.metrics.robustness_score import _metric_columns, flag_invariant_brittleness

DEFAULT_TAGS = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)

_METRIC_LABELS: dict[str, str] = {
    "validity": "Validity",
    "uniqueness": "Uniqueness",
    "n_valid": "$n_{\\mathrm{valid}}$",
    "n_unique_valid": "Unique valid count",
    "n_total": "Batch size $n_{\\mathrm{total}}$",
    "mean_qed": "Mean QED",
    "std_qed": "Std QED",
    "mean_sa": "Mean SA",
    "std_sa": "Std SA",
    "vina_score_mean": "Mean Vina",
    "vina_score_std": "Std Vina",
}


def _label(col: str) -> str:
    return _METRIC_LABELS.get(col, col.replace("_", " "))


def dominant_metric_table(
    df: pd.DataFrame,
    *,
    invariant_tags: tuple[str, ...],
    tau: float,
    model_name: str | None,
) -> pd.DataFrame:
    tag_col = "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"
    work = df.copy()
    if model_name:
        work = work[work["model_name"].astype(str).str.lower() == model_name.lower()]
    metrics = _metric_columns(work)
    inv = set(invariant_tags)
    rows: list[dict[str, object]] = []
    for (pid, mdl), grp in work.groupby(["pocket_id", "model_name"], sort=False):
        inv_rows = grp[grp[tag_col].astype(str).isin(inv)]
        if inv_rows.shape[0] < 2:
            continue
        best_m = ""
        best_std = -1.0
        any_trig = False
        for m in metrics:
            if m not in inv_rows.columns:
                continue
            vals = pd.to_numeric(inv_rows[m], errors="coerce").astype(float).values
            vals = vals[np.isfinite(vals)]
            if vals.size < 2:
                continue
            std = float(np.std(vals))
            if std > tau:
                any_trig = True
                if std > best_std + 1e-12 or (
                    abs(std - best_std) <= 1e-12 and (not best_m or m < best_m)
                ):
                    best_std = std
                    best_m = m
        o = grp[grp[tag_col].astype(str) == "original"]
        covered = False
        if not o.empty and "n_valid" in o.columns:
            nv = pd.to_numeric(o.iloc[0]["n_valid"], errors="coerce")
            covered = pd.notna(nv) and float(nv) > 0
        if not covered:
            continue
        rows.append(
            {
                "pocket_id": str(pid),
                "model_name": str(mdl),
                "dominant_metric": best_m if any_trig else "",
                "dominant_std": float(best_std) if any_trig else 0.0,
                "brittle_at_tau": bool(any_trig),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    metric_order = sorted(out["dominant_metric"].unique(), key=lambda x: (x == "", x))
    cat_map = {m: i for i, m in enumerate(metric_order)}
    out["_sort_metric"] = out["dominant_metric"].map(lambda x: cat_map.get(x, -1))
    out = out.sort_values(["_sort_metric", "pocket_id"]).drop(columns=["_sort_metric"])
    return out.reset_index(drop=True)


def plot_dominant(
    table: pd.DataFrame,
    out_pdf: Path,
    *,
    tau: float,
    title_model: str,
) -> None:
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    sub = table[table["brittle_at_tau"]].copy()
    if sub.empty:
        raise ValueError("No brittle rows to plot (check tau and CSV).")

    metrics_list = sorted(
        sub["dominant_metric"].unique(),
        key=lambda x: (-(sub["dominant_metric"] == x).sum(), x),
    )
    cmap = plt.colormaps["tab10"]
    color_map = {m: cmap(i % 10) for i, m in enumerate(metrics_list)}

    n = len(sub)
    fig_h = max(8.0, 0.22 * n)
    fig, ax = plt.subplots(figsize=(7.2, fig_h), facecolor="white")
    ax.set_facecolor("white")
    y = np.arange(n)
    colors = [color_map[str(m)] for m in sub["dominant_metric"]]
    ax.barh(y, sub["dominant_std"].values, color=colors, height=0.82, edgecolor="white", linewidth=0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(sub["pocket_id"].tolist(), fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(r"Largest exceeding $\tau$: std across 4 featurization conditions (unitless)")
    ax.set_title(
        f"{title_model}: dominant brittleness-driving summary metric per pocket "
        f"($\\tau = {tau:.2f}$, covered pockets with flag)"
    )
    ax.set_xlim(0, max(float(sub["dominant_std"].max()) * 1.08, tau * 1.5))
    ax.axvline(tau, color="#555", linestyle="--", linewidth=1, label=f"$\\tau = {tau:.2f}$")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_map[m], ec="none") for m in metrics_list
    ]
    ax.legend(
        handles,
        [_label(str(m)) for m in metrics_list],
        loc="lower right",
        fontsize=8,
        frameon=True,
        title="Dominant metric",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__run1778615375.csv",
    )
    ap.add_argument(
        "--tau",
        type=float,
        default=0.10,
        help="Same threshold as brittleness flagging",
    )
    ap.add_argument(
        "--model",
        type=str,
        default="diffsbdd",
        help="model_name filter (lowercase)",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO / "paper/dominant_brittleness_metric_diffsbdd.csv",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=_REPO / "paper/figures/dominant_brittleness_metric.pdf",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    for col in ("brittle_invariant", "brittleness_note"):
        if col in df.columns:
            df = df.drop(columns=[col])

    table = dominant_metric_table(
        df,
        invariant_tags=DEFAULT_TAGS,
        tau=float(args.tau),
        model_name=args.model,
    )
    if table.empty:
        print("No rows in dominant metric table.", file=sys.stderr)
        return 1

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(table)} covered pockets)")

    flagged = flag_invariant_brittleness(
        df[df["model_name"].astype(str).str.lower() == args.model.lower()],
        invariant_tags=DEFAULT_TAGS,
        metric_std_threshold=float(args.tau),
    )
    n_brittle = flagged.groupby(["pocket_id", "model_name"])["brittle_invariant"].any().sum()
    print(f"Cross-check: {int(n_brittle)} brittle pairs at tau={args.tau} (flag_invariant_brittleness)")

    brittle_count = int(table["brittle_at_tau"].sum())
    plot_dominant(table, args.out_pdf, tau=float(args.tau), title_model="DiffSBDD")
    print(f"Wrote {args.out_pdf} ({brittle_count} brittle rows plotted)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
