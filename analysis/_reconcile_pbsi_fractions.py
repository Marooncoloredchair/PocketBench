"""Reconcile frac_shrinking_lowers_qed (PBSI) vs the paired crop-minus 'degraded' fraction.

Raw-data only: recomputes both estimators from the metrics CSVs so the two published
numbers can be attributed to a definition difference and/or a panel difference.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sbdd_robust.report import (
    crop_paired_wilcoxon,
    pocket_boundary_sensitivity,
    tag_col,
)

FILES = [
    "data/results/metrics_per_condition__runreal100.csv",
    "data/results/metrics_per_condition__rundiffsbdd_real100.csv",
    "data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv",
]


def paired_frac(df: pd.DataFrame, metric: str = "mean_qed") -> dict:
    """Per-pocket (crop_minus - original) and (plus - minus) fractions/medians."""
    tcol = tag_col(df)
    d = df.copy()
    d["_tag"] = d[tcol].astype(str)
    piv = d.pivot_table(index="pocket_id", columns="_tag", values=metric, aggfunc="first")
    out: dict = {}
    if {"original", "crop_radius_minus_1.5"} <= set(piv.columns):
        pair = piv[["original", "crop_radius_minus_1.5"]].dropna()
        delta = pair["crop_radius_minus_1.5"].to_numpy() - pair["original"].to_numpy()
        out["n_pairs_minus_vs_orig"] = int(delta.size)
        out["median_dQED_minus_vs_orig"] = round(float(np.median(delta)), 5)
        out["frac_minus_lowers_qed"] = round(float(np.mean(delta < 0)), 4)
    if {"crop_radius_plus_1.5", "crop_radius_minus_1.5"} <= set(piv.columns):
        pair2 = piv[["crop_radius_plus_1.5", "crop_radius_minus_1.5"]].dropna()
        d2 = pair2["crop_radius_plus_1.5"].to_numpy() - pair2["crop_radius_minus_1.5"].to_numpy()
        out["n_pairs_plus_vs_minus"] = int(d2.size)
        out["frac_plus_gt_minus"] = round(float(np.mean(d2 > 0)), 4)
    return out


def main() -> None:
    for f in FILES:
        p = Path(f)
        if not p.exists():
            print(f"MISSING {f}")
            continue
        df = pd.read_csv(p)
        _, summary = pocket_boundary_sensitivity(df, dataset=p.stem)
        pf = paired_frac(df)
        w = crop_paired_wilcoxon(df, dataset=p.stem, n_boot=2000)
        wq = w[w["metric"] == "dQED"]

        print(f"=== {p.name} ===")
        print(f"  pockets_in_csv            : {df['pocket_id'].nunique()}")
        print(f"  PBSI n_pockets            : {summary['n_pockets']}")
        print(f"  frac_shrinking_lowers_qed : {summary['frac_shrinking_lowers_qed']}  (= mean(slope>0))")
        print(f"  frac_plus_gt_minus        : {pf.get('frac_plus_gt_minus')}  (plus vs minus)")
        print(f"  frac_minus_lowers_qed     : {pf.get('frac_minus_lowers_qed')}  (minus vs original)")
        print(f"  median_dQED_minus_vs_orig : {pf.get('median_dQED_minus_vs_orig')}")
        print(f"  n_pairs minus/orig        : {pf.get('n_pairs_minus_vs_orig')}")
        if not wq.empty:
            r = wq.iloc[0]
            print(f"  wilcoxon median_delta     : {r['median_delta']}  p={r['p_value']:.3g}  n={r['n_pairs']}")

    # Is the 3-point slope algebraically the plus-vs-minus contrast?
    print("=== slope identity check (symmetric 3-point fit) ===")
    df = pd.read_csv(FILES[0])
    pp, _ = pocket_boundary_sensitivity(df, dataset="identity")
    endpoint_slope = (pp["qed_at_plus"] - pp["qed_at_minus"]) / 3.0
    diff = (pp["pbsi_slope_qed_per_A"] - endpoint_slope).abs().max()
    print(f"  max |fitted_slope - (plus-minus)/3A| = {diff:.3e}")
    print(f"  agreement of signs: {float((np.sign(pp['pbsi_slope_qed_per_A']) == np.sign(endpoint_slope)).mean()):.4f}")


if __name__ == "__main__":
    main()
