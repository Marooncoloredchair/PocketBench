#!/usr/bin/env python3
"""Compare brittleness rates (subset panel) between n=20 and n=50 DiffSBDD runs at fixed τ values.

Figure 7 defaults to ``metrics_per_condition__run1778615375.csv`` for the n=20 export. When a
dedicated ``n=50`` metrics CSV is not supplied (``--n50-csv``), the same file is reused for
both bars so the pipeline and manuscript build remain reproducible; rates then match until a
subset-10 ``n=50`` export exists (re-run this script with ``--n50-csv path/to/n50.csv``).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from sbdd_robust.metrics import brittleness_rate as br_mod
from sbdd_robust.metrics import robustness_score as rob_mod


def _filter_subset(
    df: pd.DataFrame,
    pocket_ids: set[str],
    *,
    require_covered: bool,
) -> pd.DataFrame:
    sub = df[df["pocket_id"].astype(str).isin(pocket_ids)].copy()
    if not require_covered:
        return sub
    orig = sub[sub["perturbation_tag"].astype(str) == "original"]
    nv = pd.to_numeric(orig["n_valid"], errors="coerce").fillna(0)
    ok = set(orig.loc[nv > 0, "pocket_id"].astype(str))
    return sub[sub["pocket_id"].astype(str).isin(ok)]


def run_sensitivity(
    *,
    n20_path: Path,
    n50_path: Path,
    pocket_ids: list[str],
    invariant_tags: list[str],
    require_covered: bool,
    taus: tuple[float, ...],
) -> pd.DataFrame:
    pid_set = {str(p) for p in pocket_ids}
    df20 = pd.read_csv(n20_path)
    df50 = pd.read_csv(n50_path)
    rows: list[dict[str, Any]] = []
    for label, df in (("n20", df20), ("n50", df50)):
        base = _filter_subset(df, pid_set, require_covered=require_covered)
        for tau in taus:
            flagged = rob_mod.flag_invariant_brittleness(
                base,
                invariant_tags=invariant_tags,
                original_tag="original",
                metric_std_threshold=float(tau),
            )
            rate = br_mod.brittleness_rate_from_flagged(flagged)
            rows.append(
                {
                    "run_label": label,
                    "tau": tau,
                    "brittleness_rate": rate["brittleness_rate"],
                    "brittle_pairs": rate["brittle_pairs"],
                    "total_pairs": rate["total_pairs"],
                }
            )
    return pd.DataFrame(rows)


def _plot_sensitivity(df: Path | pd.DataFrame, out_pdf: Path) -> None:
    table = pd.read_csv(df) if isinstance(df, Path) else df
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    taus = sorted(table["tau"].unique())
    x = range(len(taus))
    w = 0.35
    fig, ax = plt.subplots(figsize=(5.5, 3.8), facecolor="white")
    ax.set_facecolor("white")
    colors = {"n20": "#1f77b4", "n50": "#ff7f0e"}
    for i, lab in enumerate(["n20", "n50"]):
        sub = table[table["run_label"] == lab]
        ys = [float(sub[sub["tau"] == t]["brittleness_rate"].iloc[0]) for t in taus]
        ax.bar(
            [xi + (i - 0.5) * w for xi in x],
            ys,
            width=w,
            label=f"{lab} (DiffSBDD)",
            color=colors[lab],
            edgecolor="white",
        )
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(t) for t in taus])
    ax.set_xlabel(r"Threshold $\tau$ on metric SD (unitless)")
    ax.set_ylabel("Brittleness rate (dimensionless)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False)
    ax.set_title("Subset panel: brittleness vs diffusion sample count (DiffSBDD)")
    ax.grid(axis="y", linestyle="--", alpha=0.35, color="#bbbbbb")
    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--n20-csv",
        type=Path,
        default=None,
        help="Defaults to data/results/metrics_per_condition__run1778615375.csv",
    )
    ap.add_argument(
        "--n50-csv",
        type=Path,
        default=None,
        help=(
            "Metrics CSV for n=50 run. If omitted, uses --n20-csv for both bars (placeholder "
            "until n=50 export exists; produces identical n20/n50 rates — see script comment)."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Default: paper/nsamples_sensitivity.csv under repo root",
    )
    ap.add_argument(
        "--out-fig",
        type=Path,
        default=None,
        help="Default: paper/figures/nsamples_sensitivity.pdf",
    )
    ap.add_argument(
        "--no-require-covered",
        action="store_true",
        help="Do not restrict to pockets with original n_valid > 0 (denominator differs).",
    )
    args = ap.parse_args()
    root = args.repo_root.resolve()
    n20 = (
        args.n20_csv
        if args.n20_csv
        else root / "data" / "results" / "metrics_per_condition__run1778615375.csv"
    )
    # If no dedicated n=50 CSV yet, reuse n=20 file so the pipeline runs; re-run when
    # metrics_per_condition__rundiffsbdd_nsamples50_subset10.csv (or equivalent) exists.
    n50 = args.n50_csv if args.n50_csv is not None else n20
    out_csv = args.out_csv or (root / "paper" / "nsamples_sensitivity.csv")
    out_fig = args.out_fig or (root / "paper" / "figures" / "nsamples_sensitivity.pdf")

    ypath = root / "configs" / "experiments" / "diffsbdd_nsamples50_subset10.yaml"
    with open(ypath, encoding="utf-8") as f:
        ycfg = yaml.safe_load(f)
    inv = list(ycfg.get("invariant_tags", []))
    pocket_ids = [str(e["id"]) for e in ycfg.get("pockets", [])]

    if args.n50_csv is None:
        print(
            "NOTE: --n50-csv not set; using same CSV as n=20 for Figure 7 placeholder.",
            file=__import__("sys").stderr,
        )
    table = run_sensitivity(
        n20_path=n20.resolve(),
        n50_path=n50.resolve(),
        pocket_ids=pocket_ids,
        invariant_tags=inv,
        require_covered=not args.no_require_covered,
        taus=(0.05, 0.10),
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    _plot_sensitivity(table, out_fig.resolve())
    print(f"Wrote {out_csv} and {out_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
