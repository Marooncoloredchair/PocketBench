#!/usr/bin/env python3
"""Figure 4: Schematic — one pocket passing through four featurization stress transforms."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "fig4_schematic.pdf",
    )
    args = ap.parse_args()

    plt.rcParams.update({"font.size": 11})
    fig, ax = plt.subplots(figsize=(10.5, 2.85), facecolor="white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.55)
    ax.axis("off")

    def box(cx, text, w=1.55, h=1.05, fc="#e8f4fc", ec="#1f77b4"):
        x = cx - w / 2
        y = 0.55
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.45,
            edgecolor=ec,
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(cx, y + h / 2, text, ha="center", va="center", fontsize=10.5, linespacing=1.15)

    centers = [1.0, 3.0, 5.0, 7.0, 9.0]
    texts = [
        "Reference\npocket",
        "Atom-order\nshuffle",
        "Coordinate\njitter",
        r"Crop $+$1.5 Å",
        r"Crop $-$1.5 Å",
    ]
    colors = [("#dedede", "#555555")] + [( "#e8f4fc", "#1f77b4")] * 4
    for cx, txt, (fc, ec) in zip(centers, texts, colors):
        box(cx, txt, fc=fc, ec=ec)

    for i in range(len(centers) - 1):
        y_mid = 0.55 + 1.05 / 2
        a = FancyArrowPatch(
            (centers[i] + 0.78, y_mid),
            (centers[i + 1] - 0.78, y_mid),
            arrowstyle="->",
            mutation_scale=14,
            color="#222",
            linewidth=1.15,
        )
        ax.add_patch(a)

    ax.text(
        5.5,
        2.18,
        "Nominally chemistry-preserving pocket featurization / definition stress (same structure file, altered tensor)",
        ha="center",
        fontsize=10,
        style="italic",
        color="#333",
    )

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
