#!/usr/bin/env python3
"""Figure 3: Mean SA (original) vs mean SA averaged over invariant conditions — DiffSBDD."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
INVARIANT_TAGS = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__run1778615375.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "fig3_sa_drift.pdf",
    )
    args = ap.parse_args()
    df = pd.read_csv(args.metrics)
    df = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]

    orig = df[df["perturbation_tag"].astype(str) == "original"]
    orig = orig.drop_duplicates(subset=["pocket_id"]).set_index("pocket_id")
    xs: list[float] = []
    ys: list[float] = []
    for pid in orig.index.unique():
        inv = df[
            (df["pocket_id"].astype(str) == str(pid))
            & (df["perturbation_tag"].astype(str).isin(INVARIANT_TAGS))
        ]
        if len(inv) != 4:
            continue
        sa_o = pd.to_numeric(orig.loc[pid]["mean_sa"], errors="coerce")
        if isinstance(sa_o, pd.Series):
            sa_o = float(sa_o.iloc[0])
        sa_i = pd.to_numeric(inv["mean_sa"], errors="coerce").mean()
        if np.isfinite(sa_o) and np.isfinite(sa_i):
            xs.append(float(sa_o))
            ys.append(float(sa_i))

    lims = [min(xs + ys) * 0.97, max(xs + ys) * 1.03]

    fig, ax = plt.subplots(figsize=(5.5, 5.5), facecolor="white")
    ax.set_facecolor("white")
    ax.scatter(xs, ys, c="#1f77b4", edgecolors="white", linewidths=0.6, s=55, alpha=0.9)
    ax.plot(lims, lims, color="#b0b0b0", linestyle="--", linewidth=1.2, label="Identity")
    ax.set_xlabel("Mean SA, original pocket (RDKit SA score, unitless)")
    ax.set_ylabel("Mean SA, average over 4 invariant conditions (unitless)")
    ax.set_title("DiffSBDD: synthetic-accessibility drift under invariants (47 pockets)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, linestyle="--", alpha=0.35, color="#cccccc")
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", args.out.resolve(), f"({len(xs)} points)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
