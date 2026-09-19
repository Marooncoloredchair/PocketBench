#!/usr/bin/env python3
"""Build matched-4-pocket metrics slices for DiffSBDD (Colab) where face_peel is absent."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
PANEL = ["1AO7", "1B0R", "1HXC", "1I4F"]
MATCHED_TAGS = {
    "original",
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_minus_1.5",
    "face_peel_0.25",
}


def main() -> None:
    local = _ROOT / "data/results/metrics_per_condition__rundiffsbdd_isr_matched4_v2.csv"
    src = local if local.is_file() else _ROOT / "data/results/metrics_per_condition__run1778615375.csv"
    out = _ROOT / "data/results/metrics_per_condition__rundiffsbdd_isr_matched4_panel.csv"
    df = pd.read_csv(src)
    df["pocket_id"] = df["pocket_id"].astype(str).str.upper()
    sub = df[
        df["pocket_id"].isin(PANEL)
        & df["perturbation_tag"].astype(str).isin(MATCHED_TAGS)
    ].copy()
    sub.to_csv(out, index=False)
    has_peel = (sub["perturbation_tag"] == "face_peel_0.25").any()
    print(f"Source: {src}")
    print(f"Wrote {out} ({len(sub)} rows; face_peel present={has_peel})")
    if not has_peel:
        print("NOTE: DiffSBDD Colab real50 run has no face_peel — ISR panel will be crop-only for DiffSBDD.")


if __name__ == "__main__":
    main()
