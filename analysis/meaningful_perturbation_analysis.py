#!/usr/bin/env python3
"""
Summarize meaningful (Ala / Trp) perturbations vs. original per pocket: metric deltas, heatmap, responsiveness.

Use ``--no-vina`` before docking finishes; omit it for full analysis including Vina deltas and docking-based checks.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

MUTATION_TAGS = ("meaningful_ala", "meaningful_trp")
RESPONSIVE_THRESHOLD = 0.10
INTERP_SIGN_EPS = 0.02


def _f(x: object) -> float:
    v = pd.to_numeric(x, errors="coerce")
    if hasattr(v, "item"):
        try:
            v = v.item()
        except (ValueError, TypeError):
            v = float(v.iloc[0]) if hasattr(v, "iloc") else float(v)
    return float(v) if pd.notna(v) else float("nan")


def _interpretable_qed_validity(dqed: float, dval: float) -> bool:
    """
    Directionality from QED and validity alone: strong opposite signs => not interpretable.
    """
    if not (np.isfinite(dqed) and np.isfinite(dval)):
        return False
    if abs(dqed) < INTERP_SIGN_EPS and abs(dval) < INTERP_SIGN_EPS:
        return True
    if abs(dqed) >= INTERP_SIGN_EPS and abs(dval) >= INTERP_SIGN_EPS:
        if dqed > 0 and dval < 0:
            return False
        if dqed < 0 and dval > 0:
            return False
    return True


def _vina_interpretable(dqed: float, dvina: float) -> bool:
    """Loose concordance: conflicting strong signs between QED and Vina delta flagged."""
    if not (np.isfinite(dqed) and np.isfinite(dvina)):
        return False
    if abs(dqed) < INTERP_SIGN_EPS and abs(dvina) < INTERP_SIGN_EPS:
        return True
    if abs(dqed) >= INTERP_SIGN_EPS and abs(dvina) >= INTERP_SIGN_EPS:
        if dqed > 0 > dvina or dqed < 0 < dvina:
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=_REPO / "data/results/metrics_per_condition__rundiffsbdd_meaningful_real20.csv",
    )
    ap.add_argument(
        "--summary-out",
        type=Path,
        default=_REPO / "paper/meaningful_perturbation_summary.csv",
    )
    ap.add_argument(
        "--heatmap-out",
        type=Path,
        default=_REPO / "paper/figures/meaningful_deltas_heatmap.pdf",
    )
    ap.add_argument(
        "--no-vina",
        action="store_true",
        help="Skip Vina deltas and docking-dependent interpretability checks.",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats and print summary only; do not write CSV or PDF.",
    )
    args = ap.parse_args()

    path = Path(args.metrics_csv).resolve()
    if not path.is_file():
        print(f"Missing metrics CSV: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path)
    need = {"pocket_id", "perturbation_tag", "validity", "mean_qed", "mean_sa"}
    miss = need - set(df.columns)
    if miss:
        print(f"CSV missing columns: {miss}", file=sys.stderr)
        return 1

    use_vina = not args.no_vina
    if use_vina and "vina_score_mean" not in df.columns:
        print(
            "Vina column missing; use --no-vina for pre-docking CSV.",
            file=sys.stderr,
        )
        return 1

    orig = df[df["perturbation_tag"].astype(str) == "original"].copy()
    orig = orig.drop_duplicates(subset=["pocket_id"]).set_index("pocket_id")

    rows: list[dict[str, object]] = []
    for pid in sorted(orig.index.astype(str).unique()):
        ro = orig.loc[pid]
        v0 = _f(ro["validity"])
        q0 = _f(ro["mean_qed"])
        s0 = _f(ro["mean_sa"])
        vin0 = _f(ro["vina_score_mean"]) if use_vina else float("nan")

        for tag in MUTATION_TAGS:
            sub = df[
                (df["pocket_id"].astype(str) == pid)
                & (df["perturbation_tag"].astype(str) == tag)
            ]
            if sub.empty:
                continue
            r = sub.iloc[0]
            v1, q1, s1 = _f(r["validity"]), _f(r["mean_qed"]), _f(r["mean_sa"])
            dv = float(v1 - v0) if np.isfinite(v1) and np.isfinite(v0) else float("nan")
            dq = float(q1 - q0) if np.isfinite(q1) and np.isfinite(q0) else float("nan")
            ds = float(s1 - s0) if np.isfinite(s1) and np.isfinite(s0) else float("nan")
            dvin = float("nan")
            if use_vina:
                v1b = _f(r["vina_score_mean"])
                if np.isfinite(v1b) and np.isfinite(vin0):
                    dvin = float(v1b - vin0)

            comp = [dv, dq, ds]
            if use_vina:
                comp.append(dvin)
            mag = [abs(x) for x in comp if np.isfinite(x)]
            responsive = bool(mag) and max(mag) > RESPONSIVE_THRESHOLD
            interp_qv = _interpretable_qed_validity(dq, dv)
            row: dict[str, object] = {
                "pocket_id": pid,
                "mutation_tag": tag,
                "delta_validity": dv,
                "delta_mean_qed": dq,
                "delta_mean_sa": ds,
                "responsive_any_gt_zero10": responsive,
                "interpretable_qed_validity": interp_qv,
            }
            if use_vina:
                row["delta_vina"] = dvin
                row["interpretable_vina_qed"] = (
                    _vina_interpretable(dq, dvin)
                    if np.isfinite(dvin) and np.isfinite(dq)
                    else False
                )
            rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No mutation rows built (check CSV).", file=sys.stderr)
        return 1

    heat_rows: list[dict[str, float]] = []
    for pid in sorted(summary["pocket_id"].unique()):
        drow: dict[str, float] = {}
        for tag in MUTATION_TAGS:
            sub = summary[
                (summary["pocket_id"] == pid)
                & (summary["mutation_tag"] == tag)
            ]
            if sub.empty:
                continue
            r0 = sub.iloc[0]
            prefix = "ala_" if "ala" in tag else "trp_"
            drow[f"{prefix}d_validity"] = float(r0["delta_validity"])
            drow[f"{prefix}d_qed"] = float(r0["delta_mean_qed"])
            drow[f"{prefix}d_sa"] = float(r0["delta_mean_sa"])
            if use_vina and "delta_vina" in summary.columns:
                drow[f"{prefix}d_vina"] = float(r0["delta_vina"])
        if drow:
            drow["_pocket"] = pid
            heat_rows.append(drow)
    heat_df = pd.DataFrame(heat_rows).set_index("_pocket").sort_index()
    heat_df = heat_df.astype(float)

    n_pairs = len(summary)
    n_resp = int(summary["responsive_any_gt_zero10"].sum())
    n_non = n_pairs - n_resp
    resp_df = summary[summary["responsive_any_gt_zero10"]]
    n_interp = int(resp_df["interpretable_qed_validity"].sum())
    frac = (n_interp / len(resp_df)) if len(resp_df) else float("nan")

    print(f"Pairs (pocket x mutation): {n_pairs}")
    print(
        "Responsive (max(|d validity|, |d QED|, |d SA|"
        + (", |d Vina|" if use_vina else "")
        + f") > {RESPONSIVE_THRESHOLD}): {n_resp}"
    )
    print(f"Non-responsive: {n_non}")
    print(
        "Of responsive pairs, interpretable (QED-validity directionality): "
        f"{n_interp} / {len(resp_df) if len(resp_df) else 0} "
        f"({frac:.1%})"
    )
    if use_vina and len(resp_df) and "interpretable_vina_qed" in resp_df.columns:
        iv = int(resp_df["interpretable_vina_qed"].sum())
        print(
            "Of responsive pairs, Vina-QED interpretability (docking-aided): "
            f"{iv} / {len(resp_df)} ({iv / len(resp_df):.1%})"
        )

    if not args.dry_run:
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.heatmap_out.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.summary_out, index=False)
        sns.set_theme(style="white")
        sns.set_context("paper", font_scale=1.35)
        plt.rcParams.update(
            {
                "axes.titlesize": 13,
                "axes.labelsize": 12,
                "xtick.labelsize": 9.5,
                "ytick.labelsize": 10,
            }
        )
        fig_h = max(5.2, 0.42 * len(heat_df))
        fig_w = max(11.5, min(14.0, 7.5 + 0.12 * heat_df.shape[1]))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        arr = heat_df.values
        finite = arr[np.isfinite(arr)]
        vmax = float(np.nanpercentile(np.abs(finite), 95)) if finite.size else 0.15
        vmax = max(vmax, 0.15)
        sns.heatmap(
            heat_df,
            ax=ax,
            cmap="RdBu_r",
            center=0.0,
            vmin=-vmax,
            vmax=vmax,
            linewidths=0.55,
            linecolor="0.88",
            cbar_kws={"label": r"$\Delta$ vs original (same units as metric)"},
        )
        cb = ax.collections[0].colorbar
        if cb is not None:
            cb.ax.tick_params(labelsize=10)
        ax.set_title(
            "Meaningful pocket mutations: metric deltas vs original (DiffSBDD)"
            + (" — Vina omitted" if args.no_vina else "")
        )
        ax.tick_params(axis="y", which="major", labelsize=10)
        plt.xticks(rotation=35, ha="right")
        plt.tight_layout()
        fig.savefig(args.heatmap_out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote {args.summary_out}")
        print(f"Wrote {args.heatmap_out}")
    else:
        print("--dry-run: skipped writing CSV and PDF.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
