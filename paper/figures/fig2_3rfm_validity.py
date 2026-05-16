#!/usr/bin/env python3
"""Figure 2: 3RFM validity across five conditions (DiffSBDD)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
_ORDER = [
    "original",
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
]
_LABELS = [
    "Original",
    "Atom shuffle",
    "Coordinate jitter",
    r"Crop $+$1.5 Å",
    r"Crop $-$1.5 Å",
]


def _load_metrics(primary: Path, fallback: Path, pocket_id: str) -> pd.DataFrame:
    df = pd.read_csv(primary)
    sub = df[df["pocket_id"].astype(str).str.upper() == pocket_id.upper()]
    if sub.empty and fallback.is_file():
        sys.stderr.write(
            f"NOTE: pocket {pocket_id!r} not in {primary.name}; "
            f"using fallback {fallback.name} (same protocol, prior run export).\n"
        )
        df = pd.read_csv(fallback)
        sub = df[df["pocket_id"].astype(str).str.upper() == pocket_id.upper()]
    if sub.empty:
        raise FileNotFoundError(f"No rows for pocket_id={pocket_id!r}")
    return sub


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__run1778615375.csv",
    )
    ap.add_argument(
        "--fallback",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__run1778460947.csv",
    )
    ap.add_argument("--pocket-id", default="3RFM")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "fig2_3rfm_validity.pdf",
    )
    args = ap.parse_args()

    sub = _load_metrics(args.metrics.resolve(), args.fallback.resolve(), args.pocket_id)
    tag_to_v: dict[str, float] = {}
    for _, r in sub.iterrows():
        tag = str(r["perturbation_tag"])
        tag_to_v[tag] = float(pd.to_numeric(r["validity"], errors="coerce"))

    heights = [tag_to_v[t] for t in _ORDER]
    colors = ["#b0b0b0"] + ["#1f77b4"] * 4

    fig, ax = plt.subplots(figsize=(7.5, 4.0), facecolor="white")
    ax.set_facecolor("white")
    x = range(len(_ORDER))
    ax.bar(x, heights, color=colors, edgecolor="#333333", linewidth=0.4)
    ax.set_xticks(list(x))
    ax.set_xticklabels(_LABELS, rotation=25, ha="right")
    ax.set_ylabel("Validity (fraction)")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Pocket {args.pocket_id}: validity under invariant perturbations (DiffSBDD)")
    ax.axhline(1.0, color="#dddddd", linestyle="--", linewidth=0.8)
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Wrote", args.out.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
