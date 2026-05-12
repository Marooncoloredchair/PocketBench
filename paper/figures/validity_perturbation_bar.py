#!/usr/bin/env python3
"""Grouped bar chart: validity by pocket and perturbation tag (matplotlib only)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
CSV = ROOT / "data" / "results" / "metrics_per_condition__run1778460947.csv"
OUT_DIR = Path(__file__).resolve().parent
OUT_PDF = OUT_DIR / "fig1_validity.pdf"

TAG_ORDER = [
    "original",
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
]

# Original: blue; four invariants: orange shades (matplotlib tab20c oranges)
COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#ff9f40",
    "#ffb870",
    "#ffd199",
]


def main() -> None:
    if not CSV.is_file():
        raise FileNotFoundError(CSV)

    df = pd.read_csv(CSV)
    pockets = []
    for p in df["pocket_id"].unique():
        pockets.append(p)

    n_p = len(pockets)
    n_b = len(TAG_ORDER)
    x = np.arange(n_p)
    width = 0.16

    fig, ax = plt.subplots(figsize=(9, 4.5))

    for i, tag in enumerate(TAG_ORDER):
        heights = []
        for pocket in pockets:
            row = df[(df["pocket_id"] == pocket) & (df["perturbation_tag"] == tag)]
            if row.empty:
                heights.append(0.0)
            else:
                heights.append(float(row["validity"].iloc[0]))
        offset = (i - (n_b - 1) / 2) * width
        ax.bar(
            x + offset,
            heights,
            width,
            label=tag.replace("_", " "),
            color=COLORS[i],
            edgecolor="black",
            linewidth=0.3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(pockets)
    ax.set_ylabel("Validity (fraction RDKit-sanitizable)")
    ax.set_xlabel("Pocket")
    ax.set_ylim(0, 1.05)
    ax.legend(title="Perturbation", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    ax.set_title("DiffSBDD validity by pocket and pocket-input perturbation")
    fig.tight_layout()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, format="pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PDF}")


if __name__ == "__main__":
    main()
