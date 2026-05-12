"""Paper-style plots comparing original vs perturbed metric summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def scatter_original_vs_invariant_mean(
    summary_df: pd.DataFrame,
    metric: str,
    out_path: Path,
    title: Optional[str] = None,
) -> Path:
    """
    Scatter x = metric on original pocket, y = mean across invariant perturbations.

    Expects columns ``{metric}_original`` and ``{metric}_inv_mean`` from
    :func:`sbdd_robust.metrics.robustness_score.summarize_robustness_vs_original`.
    """
    xcol = f"{metric}_original"
    ycol = f"{metric}_inv_mean"
    if xcol not in summary_df.columns or ycol not in summary_df.columns:
        raise KeyError(f"Summary missing columns for metric {metric!r}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(6, 6))
    xs = summary_df[xcol].astype(float).values
    ys = summary_df[ycol].astype(float).values
    ax.scatter(xs, ys, alpha=0.85, edgecolor="k", linewidth=0.5)
    lims = [
        np.nanmin([np.nanmin(xs), np.nanmin(ys)]),
        np.nanmax([np.nanmax(xs), np.nanmax(ys)]),
    ]
    if np.isfinite(lims[0]) and np.isfinite(lims[1]):
        ax.plot(lims, lims, ls="--", c="gray", lw=1, label="y = x")
    ax.set_xlabel(f"{metric} (original)")
    ax.set_ylabel(f"{metric} (mean, invariant perturbations)")
    ax.set_title(title or f"{metric}: robustness view")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_all_metrics(
    summary_df: pd.DataFrame,
    metrics: Iterable[str],
    out_dir: Path,
    prefix: str = "scatter",
) -> list[Path]:
    written: list[Path] = []
    for m in metrics:
        p = out_dir / f"{prefix}_{m}.png"
        written.append(scatter_original_vs_invariant_mean(summary_df, m, p))
    return written
