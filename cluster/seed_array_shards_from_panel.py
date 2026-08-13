#!/usr/bin/env python3
"""Split a panel metrics CSV into per-pocket SLURM shard files for --resume.

Writes metrics_per_condition__run{RUN_ID}__pNNNN.csv using pocket order from a YAML config.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", type=Path, required=True, help="Existing panel metrics CSV")
    ap.add_argument("--config", type=Path, required=True, help="YAML with pockets:[] order")
    ap.add_argument("--results-dir", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    pockets = [str(p["id"]) for p in cfg["pockets"]]
    id_to_idx = {pid: i for i, pid in enumerate(pockets)}

    df = pd.read_csv(args.panel)
    if "pocket_id" not in df.columns:
        raise SystemExit("panel missing pocket_id")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for pid, g in df.groupby(df["pocket_id"].astype(str), sort=False):
        if pid not in id_to_idx:
            print(f"WARN: pocket {pid} in panel but not in config — skip")
            continue
        idx = id_to_idx[pid]
        out = args.results_dir / f"metrics_per_condition__run{args.run_id}__p{idx:04d}.csv"
        if args.dry_run:
            print(f"would write {out} rows={len(g)}")
        else:
            g.to_csv(out, index=False)
            print(f"wrote {out} rows={len(g)}")
        written += 1
    print(f"shards={written} / config_pockets={len(pockets)}")
    missing = [pid for pid in pockets if pid not in set(df["pocket_id"].astype(str))]
    print(f"not_yet_run={len(missing)}: {','.join(missing[:20])}{'…' if len(missing)>20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
