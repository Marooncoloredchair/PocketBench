#!/usr/bin/env python3
"""Append 1AO7 rows from targetdiff_isr_1ao7_rerun into the smoke5 metrics CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rerun",
        default=str(_ROOT / "data/results/metrics_per_condition__runtargetdiff_isr_1ao7_rerun.csv"),
    )
    ap.add_argument(
        "--smoke",
        default=str(_ROOT / "data/results/metrics_per_condition__runtargetdiff_isr_smoke5.csv"),
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output path (default: overwrite --smoke in place).",
    )
    args = ap.parse_args()

    rerun_path = Path(args.rerun)
    smoke_path = Path(args.smoke)
    if not rerun_path.is_file():
        raise SystemExit(f"Rerun metrics not found: {rerun_path}")

    rerun = pd.read_csv(rerun_path)
    rerun = rerun[rerun["pocket_id"].astype(str).str.upper() == "1AO7"].copy()
    if rerun.empty:
        raise SystemExit("No 1AO7 rows in rerun CSV.")

    # Normalize run_id to smoke panel for a single combined table.
    rerun["run_id"] = "targetdiff_isr_smoke5"

    if smoke_path.is_file():
        smoke = pd.read_csv(smoke_path)
        smoke = smoke[smoke["pocket_id"].astype(str).str.upper() != "1AO7"]
        out_df = pd.concat([smoke, rerun], ignore_index=True)
    else:
        out_df = rerun

    out_path = Path(args.out) if args.out else smoke_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out_df)} rows, 1AO7 from rerun)")


if __name__ == "__main__":
    main()
