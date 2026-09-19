#!/usr/bin/env python3
"""Merge matched4 rerun CSVs into anchor510 / diffsbdd panel metrics for ISR."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
PANEL = ["1AO7", "1B0R", "1HXC", "1I4F"]
MATCHED_TAGS = {"original", "atom_shuffle", "coordinate_jitter", "crop_radius_minus_1.5", "face_peel_0.25"}


def _merge(base: Path, patch: Path, out: Path, pocket_ids: list[str]) -> None:
    patch_df = pd.read_csv(patch)
    patch_df["pocket_id"] = patch_df["pocket_id"].astype(str).str.upper()
    patch_df = patch_df[
        patch_df["pocket_id"].isin(pocket_ids)
        & patch_df["perturbation_tag"].astype(str).isin(MATCHED_TAGS)
    ].copy()

    if base.is_file():
        base_df = pd.read_csv(base)
        base_df["pocket_id"] = base_df["pocket_id"].astype(str).str.upper()
        keep = base_df[
            ~(
                base_df["pocket_id"].isin(pocket_ids)
                & base_df["perturbation_tag"].astype(str).isin(MATCHED_TAGS)
            )
        ]
        out_df = pd.concat([keep, patch_df], ignore_index=True)
    else:
        out_df = patch_df

    out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(out_df)} rows; patched {len(patch_df)} from {patch.name})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2m-rerun", default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_isr_matched4_rerun.csv"))
    ap.add_argument("--p2m-out", default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_isr_matched4_merged.csv"))
    ap.add_argument(
        "--p2m-base",
        default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_isr_smoke5_anchor510.csv"),
    )
    ap.add_argument("--dsb-rerun", default=str(_ROOT / "data/results/metrics_per_condition__rundiffsbdd_isr_matched4.csv"))
    ap.add_argument("--dsb-out", default=str(_ROOT / "data/results/metrics_per_condition__rundiffsbdd_isr_matched4_merged.csv"))
    args = ap.parse_args()

    if Path(args.p2m_rerun).is_file():
        _merge(Path(args.p2m_base), Path(args.p2m_rerun), Path(args.p2m_out), ["1B0R", "1I4F"])
    else:
        print(f"SKIP P2M merge: {args.p2m_rerun} not found")

    if Path(args.dsb_rerun).is_file():
        out = Path(args.dsb_out)
        import shutil

        shutil.copy2(args.dsb_rerun, out)
        print(f"Copied DiffSBDD matched4 -> {out}")
    else:
        print(f"SKIP DiffSBDD: {args.dsb_rerun} not found")


if __name__ == "__main__":
    main()
