#!/usr/bin/env python3
"""Merge per-pocket SLURM shard CSVs into one metrics_per_condition panel.

Expects shards named:
  metrics_per_condition__run{RUN_ID}__pNNNN.csv

Writes:
  metrics_per_condition__run{RUN_ID}.csv
  metrics_per_condition__run{RUN_ID}__merge_report.txt
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

_SHARD_RE = re.compile(r"^metrics_per_condition__run(.+)__p(\d{4})\.csv$")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results-dir",
        type=Path,
        required=True,
        help="Directory containing shard CSVs (usually $POCKETBENCH_RESULTS).",
    )
    ap.add_argument(
        "--run-id",
        required=True,
        help="Run id matching metrics_per_condition__run{id}__pNNNN.csv",
    )
    ap.add_argument(
        "--expected-pockets",
        type=int,
        default=None,
        help="If set, warn when shard count or unique pocket_ids differ.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Merged CSV path (default: results-dir/metrics_per_condition__run{id}.csv).",
    )
    args = ap.parse_args()

    results = args.results_dir
    if not results.is_dir():
        raise SystemExit(f"results-dir not found: {results}")

    shards: list[tuple[int, Path]] = []
    for p in sorted(results.glob(f"metrics_per_condition__run{args.run_id}__p*.csv")):
        m = _SHARD_RE.match(p.name)
        if not m or m.group(1) != args.run_id:
            continue
        shards.append((int(m.group(2)), p))

    if not shards:
        raise SystemExit(
            f"No shards matching metrics_per_condition__run{args.run_id}__pNNNN.csv in {results}"
        )

    frames = []
    for idx, path in shards:
        df = pd.read_csv(path)
        if df.empty:
            print(f"WARN: empty shard {path.name}")
            continue
        df["_shard_index"] = idx
        frames.append(df)

    if not frames:
        raise SystemExit("All shards empty.")

    merged = pd.concat(frames, ignore_index=True)
    if "_shard_index" in merged.columns:
        merged = merged.drop(columns=["_shard_index"])

    # Prefer last write if a pocket somehow appears twice
    if {"pocket_id", "perturbation_tag"}.issubset(merged.columns):
        merged = merged.drop_duplicates(subset=["pocket_id", "perturbation_tag"], keep="last")

    out = args.out or (results / f"metrics_per_condition__run{args.run_id}.csv")
    merged.to_csv(out, index=False)

    n_pockets = merged["pocket_id"].nunique() if "pocket_id" in merged.columns else 0
    n_rows = len(merged)
    report_lines = [
        f"run_id: {args.run_id}",
        f"shards_found: {len(shards)}",
        f"shards_used: {len(frames)}",
        f"merged_rows: {n_rows}",
        f"unique_pockets: {n_pockets}",
        f"out: {out}",
    ]
    if args.expected_pockets is not None:
        report_lines.append(f"expected_pockets: {args.expected_pockets}")
        if n_pockets != args.expected_pockets:
            report_lines.append(
                f"WARN: unique_pockets ({n_pockets}) != expected ({args.expected_pockets})"
            )
        missing_idx = sorted(set(range(args.expected_pockets)) - {i for i, _ in shards})
        if missing_idx:
            report_lines.append(f"missing_shard_indices: {missing_idx[:40]}{'…' if len(missing_idx)>40 else ''}")

    report = results / f"metrics_per_condition__run{args.run_id}__merge_report.txt"
    report.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print("\n".join(report_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
