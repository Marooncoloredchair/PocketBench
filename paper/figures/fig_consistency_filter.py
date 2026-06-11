#!/usr/bin/env python3
"""Line plot: consistency-filter pass rate vs mean-summary delta threshold."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO / "paper" / "figures") not in sys.path:
    sys.path.insert(0, str(_REPO / "paper" / "figures"))
from nmi_style import COLOR_DIFFSBDD, figsize_single, save_figure, setup_rc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        type=Path,
        default=_REPO / "paper" / "consistency_filter_results.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "paper" / "figures" / "fig_consistency_filter",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    setup_rc(True)
    fig, ax = plt.subplots(figsize=figsize_single(), facecolor="white")
    ax.set_facecolor("white")

    rates: list[tuple[float, float]] = []
    for thr, g in df.groupby("threshold"):
        n = len(g)
        if n == 0:
            continue
        rates.append((float(thr), float(g["passed_combined"].sum()) / n))

    xs = [r[0] for r in sorted(rates)]
    ys = [r[1] for r in sorted(rates)]
    ax.plot(xs, ys, marker="o", color=COLOR_DIFFSBDD, linewidth=2.0, markersize=5)
    ax.set_xlabel("Threshold on |Δ mean QED| (and |Δ mean SA|)")
    ax.set_ylabel("Fraction of pockets passing")
    ax.set_title("Consistency filter pass rate vs threshold (original vs atom_shuffle)")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    pdf, png = save_figure(fig, args.out)
    plt.close(fig)
    print("Wrote", pdf, png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
