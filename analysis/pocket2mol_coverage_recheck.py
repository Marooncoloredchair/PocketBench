#!/usr/bin/env python3
"""Summarize Pocket2Mol coverage recheck vs original real47 failure baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
BASELINE = _ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full.csv"
RECHECK = _ROOT / "data/results/metrics_per_condition__runpocket2mol_coverage_recheck10.csv"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=str(BASELINE))
    ap.add_argument("--recheck", default=str(RECHECK))
    ap.add_argument("--out", default=str(_ROOT / "paper/pocket2mol_coverage_recheck.csv"))
    ap.add_argument("--summary-out", default=str(_ROOT / "paper/pocket2mol_coverage_recheck_summary.txt"))
    args = ap.parse_args()

    base = pd.read_csv(args.baseline)
    base["pocket_id"] = base["pocket_id"].astype(str).str.upper()
    orig = base[base["perturbation_tag"].astype(str) == "original"].copy()
    orig_fail = set(orig.loc[orig["n_valid"].fillna(0) == 0, "pocket_id"])

    rec = pd.read_csv(args.recheck)
    rec["pocket_id"] = rec["pocket_id"].astype(str).str.upper()
    rec_orig = rec[rec["perturbation_tag"].astype(str) == "original"].copy()

    rows: list[dict] = []
    for _, r in rec_orig.iterrows():
        pid = str(r["pocket_id"])
        b = orig[orig["pocket_id"] == pid]
        rows.append(
            {
                "pocket_id": pid,
                "baseline_n_valid": int(b.iloc[0]["n_valid"]) if not b.empty else None,
                "baseline_failed": pid in orig_fail,
                "recheck_n_valid": int(r["n_valid"]),
                "recheck_n_total": int(r["n_total"]),
                "recheck_validity": float(r["validity"]),
                "recheck_mean_qed": r.get("mean_qed"),
                "now_generates": int(r["n_valid"]) > 0,
                "harness_recovered": pid in orig_fail and int(r["n_valid"]) > 0,
            }
        )

    out_df = pd.DataFrame(rows)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)

    n = len(out_df)
    n_recovered = int(out_df["harness_recovered"].sum())
    n_still_fail = int((~out_df["now_generates"]).sum())
    extrapolated_harness = int(round(34 * n_recovered / max(n, 1)))

    summary = (
        f"Pocket2Mol coverage recheck ({n} pockets that failed in real47 original runs).\n"
        f"With the fixed full-PDB + ligand-centered adapter, {n_recovered}/{n} "
        f"({100.0 * n_recovered / max(n, 1):.0f}%) now produce valid molecules.\n"
        f"Still zero valid mols: {n_still_fail}/{n} — likely true model/pocket difficulty, not input format.\n"
        f"Original baseline: 13/47 pockets generated (34 failed). This sample implies ~"
        f"{100.0 * n_recovered / max(n, 1):.0f}% of those failures were harness/input-format artifacts "
        f"(~{extrapolated_harness} of 34 pockets, extrapolated), with ~{34 - extrapolated_harness} "
        f"remaining hard failures.\n"
        f"Baseline: {Path(args.baseline).name}; recheck: {Path(args.recheck).name}.\n"
    )
    Path(args.summary_out).write_text(summary, encoding="utf-8")

    # Append one-line summary row for spreadsheet readers.
    summary_row = {
        "pocket_id": "__SUMMARY__",
        "baseline_n_valid": 13,
        "baseline_failed": 34,
        "recheck_n_valid": n_recovered,
        "recheck_n_total": n,
        "recheck_validity": round(n_recovered / max(n, 1), 3),
        "recheck_mean_qed": None,
        "now_generates": None,
        "harness_recovered": None,
        "notes": (
            f"{n_recovered}/{n} recovered; ~{100.0 * n_recovered / max(n, 1):.0f}% of original "
            f"34 failures likely input-format related (~{extrapolated_harness}/34 extrapolated)"
        ),
    }
    pd.concat([out_df, pd.DataFrame([summary_row])], ignore_index=True).to_csv(out_path, index=False)
    print(summary)
    print(f"Wrote {out_path}")
    print(f"Wrote {args.summary_out}")


if __name__ == "__main__":
    main()
