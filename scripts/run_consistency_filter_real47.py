#!/usr/bin/env python3
"""
Apply population consistency criteria to DiffSBDD real47 metrics (original vs atom_shuffle).

Uses per-pocket summary means from metrics_per_condition CSV (not raw RDKit mols).
Outputs ``paper/consistency_filter_results.csv`` with pass flags at each (qed/sa) threshold.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_REPO = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__run1778615375.csv",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO / "paper" / "consistency_filter_results.csv",
    )
    ap.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=(0.05, 0.10, 0.15),
    )
    args = ap.parse_args()

    df = pd.read_csv(args.metrics)
    df = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]
    rows_out: list[dict] = []
    for pid in sorted(df["pocket_id"].astype(str).unique()):
        sub = df[df["pocket_id"].astype(str) == str(pid)]
        o = sub[sub["perturbation_tag"].astype(str) == "original"]
        s = sub[sub["perturbation_tag"].astype(str) == "atom_shuffle"]
        if o.empty or s.empty:
            continue
        o = o.iloc[0]
        s = s.iloc[0]
        oq = float(pd.to_numeric(o.get("mean_qed"), errors="coerce"))
        sq = float(pd.to_numeric(s.get("mean_qed"), errors="coerce"))
        oa = float(pd.to_numeric(o.get("mean_sa"), errors="coerce"))
        sa = float(pd.to_numeric(s.get("mean_sa"), errors="coerce"))
        ovi = o.get("vina_score_mean")
        svi = s.get("vina_score_mean")
        if ovi is not None and pd.notna(ovi) and pd.notna(svi):
            ov = float(ovi)
            sv = float(svi)
            vd = abs(ov - sv)
        else:
            ov = sv = float("nan")
            vd = float("nan")

        dq = abs(oq - sq)
        dsa = abs(oa - sa)

        for thr in args.thresholds:
            pq = dq < thr
            psa = dsa < thr
            if pd.notna(vd):
                pv = vd < 0.5
            else:
                pv = True
            rows_out.append(
                {
                    "threshold": thr,
                    "pocket_id": pid,
                    "original_mean_qed": oq,
                    "shuffle_mean_qed": sq,
                    "qed_delta": dq,
                    "passed_qed_filter": pq,
                    "original_mean_sa": oa,
                    "shuffle_mean_sa": sa,
                    "sa_delta": dsa,
                    "passed_sa_filter": psa,
                    "original_mean_vina": ov,
                    "shuffle_mean_vina": sv,
                    "vina_delta": vd,
                    "passed_vina_filter": pv,
                    "passed_combined": bool(pq and psa and pv),
                }
            )

    out_df = pd.DataFrame(rows_out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print("Wrote", args.out.resolve(), "rows=", len(out_df))

    for thr in args.thresholds:
        sel = out_df[out_df["threshold"] == thr]
        n = len(sel)
        if n == 0:
            continue
        c = int(sel["passed_combined"].sum())
        print(f"threshold={thr}: {c}/{n} pockets pass combined filter")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
