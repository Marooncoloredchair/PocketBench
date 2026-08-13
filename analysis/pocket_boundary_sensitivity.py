#!/usr/bin/env python3
"""Pocket Boundary Sensitivity Index (PBSI): a dose-response measure of how much a
model's drug-likeness depends on the pocket crop radius.

Thin multi-panel wrapper around ``sbdd_robust.report.pocket_boundary_sensitivity`` (the
shared source of truth, also exposed as ``pocketbench pbsi``). See that module for the
full metric definition. Runs both the real47 and real100 DiffSBDD panels.

Outputs:
  paper/pocket_boundary_sensitivity.csv         - per-pocket PBSI rows (both panels)
  paper/pocket_boundary_sensitivity_summary.csv - per-panel headline PBSI summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.report import INVARIANT_TAGS, filter_model, pocket_boundary_sensitivity, tag_col

# In-repo copy of the n=99 DiffSBDD panel metrics (moved off D:\ in Phase 1C).
REAL100_PATH = _ROOT / "data" / "results" / "metrics_per_condition__runreal100.csv"
RESULTS_DIR = _ROOT / "data" / "results"


def find_diffsbdd_real47() -> Path:
    best: tuple[int, Path] | None = None
    for p in sorted(RESULTS_DIR.glob("metrics_per_condition__*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model_name" not in df.columns or "pocket_id" not in df.columns:
            continue
        dfd = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]
        if dfd.empty:
            continue
        if not set(INVARIANT_TAGS).issubset(set(dfd[tag_col(dfd)].astype(str))):
            continue
        n_pockets = dfd["pocket_id"].astype(str).nunique()
        if 40 <= n_pockets <= 55:
            score = abs(n_pockets - 47)
            if best is None or score < best[0]:
                best = (score, p)
    if best is None:
        raise FileNotFoundError("Could not auto-detect a DiffSBDD real47 metrics CSV in data/results/")
    return best[1]


def main() -> None:
    paper = _ROOT / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    real47_path = find_diffsbdd_real47()
    datasets["real47"] = filter_model(pd.read_csv(real47_path), "diffsbdd")
    print(f"real47 source: {real47_path}")
    if REAL100_PATH.is_file():
        datasets["real100"] = filter_model(pd.read_csv(REAL100_PATH), "diffsbdd")
        print(f"real100 source: {REAL100_PATH}")
    else:
        print(f"WARN: real100 not found at {REAL100_PATH}; skipping.")

    per_pocket_all, summaries = [], []
    for name, df in datasets.items():
        pp, summary = pocket_boundary_sensitivity(df, dataset=name)
        per_pocket_all.append(pp)
        summaries.append(summary)

    pp_df = pd.concat(per_pocket_all, ignore_index=True)
    sum_df = pd.DataFrame(summaries)
    pp_out = paper / "pocket_boundary_sensitivity.csv"
    sum_out = paper / "pocket_boundary_sensitivity_summary.csv"
    pp_df.to_csv(pp_out, index=False)
    sum_df.to_csv(sum_out, index=False)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 60)
    print("\n" + "=" * 78)
    print("POCKET BOUNDARY SENSITIVITY INDEX (PBSI)")
    print("=" * 78)
    print(sum_df.to_string(index=False))
    print(f"\nWrote {pp_out}\nWrote {sum_out}")

    print("\nPlain English:")
    for s in summaries:
        print(
            f"  {s['dataset']} ({s['n_pockets']} pockets): PBSI = {s['PBSI_median_abs_slope_qed_per_A']:.3f} "
            f"QED/A (median |slope|); shrinking the pocket lowers QED in "
            f"{s['frac_shrinking_lowers_qed']:.0%} of pockets. A 1.5 A crop change moves QED a median of "
            f"{s['median_snr_1.5A']:.1f}x the featurization-noise std, and exceeds that noise floor in "
            f"{s['frac_boundary_exceeds_noise']:.0%} of pockets."
        )

    print("\nTop 10 most boundary-sensitive pockets (by |slope|):")
    top = pp_df.reindex(pp_df["abs_slope"].sort_values(ascending=False).index).head(10)
    for _, r in top.iterrows():
        print(
            f"  {r['dataset']} {r['pocket_id']}: slope={r['pbsi_slope_qed_per_A']:+.3f} QED/A, "
            f"range={r['boundary_range']:.3f}, snr_1.5A={r['snr_1.5A']}"
        )


if __name__ == "__main__":
    main()
