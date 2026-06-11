#!/usr/bin/env python3
"""
Compare DiffSBDD vs Pocket2Mol metrics: coverage, brittleness@τ on covered pockets, figure.

Usage (from repo root):
  python analysis/compare_models.py
  python analysis/compare_models.py \\
    --diffsbdd data/results/metrics_per_condition__run1778615375.csv \\
    --pocket2mol data/results/metrics_per_condition__runpocket2mol_real47_full.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo root: sbdd-robust/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "paper" / "figures"))
from nmi_style import COLOR_DIFFSBDD, COLOR_POCKET2MOL, figsize_tripanel, save_figure, setup_rc

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged
from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

# Same invariant panel as configs/experiments/pocket2mol_real47.yaml / DiffSBDD Colab runs.
DEFAULT_INVARIANT_TAGS = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)

THRESHOLDS = (0.05, 0.10, 0.15, 0.20)


def _norm_pid(s: pd.Series | np.ndarray) -> pd.Series:
    return s.astype(str).str.strip()


def _original_rows(df: pd.DataFrame) -> pd.DataFrame:
    tag_col = "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"
    return df[_norm_pid(df[tag_col]) == "original"].copy()


def _excluded_original_zero_count(df: pd.DataFrame) -> int:
    """Pockets whose *original* condition has n_valid <= 0 (or missing)."""
    o = _original_rows(df)
    if o.empty:
        return 0
    nv = pd.to_numeric(o["n_valid"], errors="coerce").fillna(0)
    bad = o.loc[nv <= 0, "pocket_id"]
    return int(bad.nunique())


def _good_original_pocket_ids(df: pd.DataFrame) -> set[str]:
    o = _original_rows(df)
    nv = pd.to_numeric(o["n_valid"], errors="coerce").fillna(0)
    o = o.assign(_nv=nv)
    return set(_norm_pid(o.loc[o["_nv"] > 0, "pocket_id"]))


def _total_pocket_count(df: pd.DataFrame) -> int:
    return int(df["pocket_id"].nunique())


def _coverage_stats(df: pd.DataFrame) -> tuple[int, int, float]:
    """(n_total pockets, n_covered with original n_valid>0, coverage_rate)."""
    n_tot = _total_pocket_count(df)
    good = _good_original_pocket_ids(df)
    n_cov = len(good)
    rate = float(n_cov / n_tot) if n_tot else 0.0
    return n_tot, n_cov, rate


def _filter_pockets(df: pd.DataFrame, pocket_ids: set[str]) -> pd.DataFrame:
    return df[_norm_pid(df["pocket_id"]).isin(pocket_ids)].copy()


def _mean_validity_original(df: pd.DataFrame) -> float:
    o = _original_rows(df)
    if o.empty:
        return float("nan")
    v = pd.to_numeric(o["validity"], errors="coerce")
    return float(v.mean())


def _perturbation_extra_failure_fraction(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Among non-original rows, fraction with n_valid==0 (Pocket2Mol stress story).
    Returns (n_zero, n_rows, rate).
    """
    tag_col = "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"
    sub = df[_norm_pid(df[tag_col]) != "original"].copy()
    if sub.empty:
        return 0, 0, float("nan")
    nv = pd.to_numeric(sub["n_valid"], errors="coerce").fillna(0)
    n_zero = int((nv <= 0).sum())
    n_rows = int(sub.shape[0])
    return n_zero, n_rows, float(n_zero / n_rows) if n_rows else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--diffsbdd",
        type=Path,
        default=_REPO_ROOT / "data/results/metrics_per_condition__run1778615375.csv",
        help="DiffSBDD metrics_per_condition CSV",
    )
    ap.add_argument(
        "--pocket2mol",
        type=Path,
        default=_REPO_ROOT / "data/results/metrics_per_condition__runpocket2mol_real47_full.csv",
        help="Pocket2Mol metrics_per_condition CSV",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=_REPO_ROOT / "paper/model_comparison_brittleness.csv",
        help="Brittleness comparison table",
    )
    ap.add_argument(
        "--out-pdf",
        type=Path,
        default=_REPO_ROOT / "paper/figures/model_comparison.pdf",
        help="Side-by-side figure (PDF)",
    )
    ap.add_argument(
        "--invariant-tags",
        nargs="+",
        default=list(DEFAULT_INVARIANT_TAGS),
        help="Invariant perturbation tags (perturbation_tag column)",
    )
    args = ap.parse_args()

    diffsbdd_path = args.diffsbdd.resolve()
    p2m_path = args.pocket2mol.resolve()
    if not diffsbdd_path.is_file():
        raise FileNotFoundError(f"DiffSBDD metrics not found: {diffsbdd_path}")
    if not p2m_path.is_file():
        raise FileNotFoundError(f"Pocket2Mol metrics not found: {p2m_path}")

    df_d = pd.read_csv(diffsbdd_path)
    df_p = pd.read_csv(p2m_path)

    n_tot_d, n_cov_d, cov_rate_d = _coverage_stats(df_d)
    n_tot_p, n_cov_p, cov_rate_p = _coverage_stats(df_p)

    n_excl_d = _excluded_original_zero_count(df_d)
    n_excl_p = _excluded_original_zero_count(df_p)
    print(
        f"Generation coverage (original n_valid > 0): "
        f"DiffSBDD {n_cov_d}/{n_tot_d} ({cov_rate_d:.4f}), "
        f"Pocket2Mol {n_cov_p}/{n_tot_p} ({cov_rate_p:.4f})"
    )
    print(
        f"Pockets excluded (original n_valid == 0): DiffSBDD={n_excl_d}, "
        f"Pocket2Mol={n_excl_p}"
    )

    good_d = _good_original_pocket_ids(df_d)
    good_p = _good_original_pocket_ids(df_p)
    shared = good_d & good_p
    print(f"Shared covered pockets (intersection): {len(shared)}")

    n_p2m_inv_zero, n_p2m_inv_rows, p2m_inv_fail_r = _perturbation_extra_failure_fraction(df_p)
    print(
        "Pocket2Mol invariant conditions with n_valid==0: "
        f"{n_p2m_inv_zero}/{n_p2m_inv_rows} ({p2m_inv_fail_r:.4f} of non-original rows)"
    )

    # Per-model brittleness on *covered* pockets only (denominator = covered count).
    df_d_cov = _filter_pockets(df_d, good_d)
    df_p_cov = _filter_pockets(df_p, good_p)

    rows_out = []
    rates_d_cov: list[float] = []
    rates_p_cov: list[float] = []
    for thr in THRESHOLDS:
        fl_d = flag_invariant_brittleness(
            df_d_cov,
            invariant_tags=args.invariant_tags,
            metric_std_threshold=float(thr),
        )
        fl_p = flag_invariant_brittleness(
            df_p_cov,
            invariant_tags=args.invariant_tags,
            metric_std_threshold=float(thr),
        )
        st_d = brittleness_rate_from_flagged(fl_d)
        st_p = brittleness_rate_from_flagged(fl_p)
        rows_out.append(
            {
                "metric_std_threshold": thr,
                "n_pockets_total_diffsbdd": n_tot_d,
                "n_covered_diffsbdd": n_cov_d,
                "coverage_rate_diffsbdd": cov_rate_d,
                "n_pockets_total_pocket2mol": n_tot_p,
                "n_covered_pocket2mol": n_cov_p,
                "coverage_rate_pocket2mol": cov_rate_p,
                "brittleness_rate_on_covered_diffsbdd": st_d["brittleness_rate"],
                "brittleness_rate_on_covered_pocket2mol": st_p["brittleness_rate"],
                "brittle_pairs_on_covered_diffsbdd": st_d["brittle_pairs"],
                "brittle_pairs_on_covered_pocket2mol": st_p["brittle_pairs"],
                "denominator_covered_diffsbdd": st_d["total_pairs"],
                "denominator_covered_pocket2mol": st_p["total_pairs"],
                "n_shared_covered_pockets": len(shared),
            }
        )
        rates_d_cov.append(float(st_d["brittleness_rate"]))
        rates_p_cov.append(float(st_p["brittleness_rate"]))

    out_csv = args.out_csv.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    # Mean original validity: shared pockets if any, else report per-model on covered.
    if shared:
        df_d_s = _filter_pockets(df_d, shared)
        df_p_s = _filter_pockets(df_p, shared)
        mean_v_d = _mean_validity_original(df_d_s)
        mean_v_p = _mean_validity_original(df_p_s)
        validity_note = f"{len(shared)} shared covered pockets"
    else:
        mean_v_d = _mean_validity_original(df_d_cov)
        mean_v_p = _mean_validity_original(df_p_cov)
        validity_note = "no shared covered pockets — means are on each model's covered set"

    out_pdf = args.out_pdf.resolve()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    setup_rc(False)
    fig, (ax0, ax1, ax2) = plt.subplots(
        1,
        3,
        figsize=figsize_tripanel(),
        constrained_layout=True,
        facecolor="white",
    )
    for ax in (ax0, ax1, ax2):
        ax.set_facecolor("white")
    w = 0.36
    x = np.arange(len(THRESHOLDS), dtype=float)
    ax0.bar(x - w / 2, rates_d_cov, width=w, label="DiffSBDD", color=COLOR_DIFFSBDD)
    ax0.bar(x + w / 2, rates_p_cov, width=w, label="Pocket2Mol", color=COLOR_POCKET2MOL)
    ax0.set_xticks(x)
    ax0.set_xticklabels([str(t) for t in THRESHOLDS])
    ax0.set_xlabel(r"Featurization metric-SD threshold $\tau$ (unitless)")
    ax0.set_ylabel("Brittleness rate (dimensionless)")
    ax0.set_title(
        "Brittleness on covered pockets (original n_valid>0)",
        fontweight="normal",
    )
    ax0.set_ylim(
        0,
        max(1.0, max(rates_d_cov + rates_p_cov) * 1.08) if (rates_d_cov + rates_p_cov) else 1.0,
    )
    ax0.legend(loc="upper left")
    ax0.axhline(0, color="0.5", linewidth=0.6)

    ax1.bar(
        [0 - w / 2, 0 + w / 2],
        [mean_v_d, mean_v_p],
        width=w,
        color=[COLOR_DIFFSBDD, COLOR_POCKET2MOL],
    )
    ax1.set_xticks([0 - w / 2, 0 + w / 2])
    ax1.set_xticklabels(["DiffSBDD", "Pocket2Mol"], fontsize=11)
    ax1.set_ylabel("Validity (fraction)")
    ax1.set_title(f"Original-condition validity (mean; {validity_note})", fontweight="normal")
    ax1.set_xlim(-0.55, 0.55)
    ax1.set_ylim(0, 1.12)
    for xpos, val in zip([0 - w / 2, 0 + w / 2], [mean_v_d, mean_v_p]):
        if val >= 0.55:
            y_text = val * 0.52
            color = "white"
            va = "center"
        else:
            y_text = val + 0.035
            color = "#222"
            va = "bottom"
        ax1.text(
            xpos,
            y_text,
            f"{val:.3f}",
            ha="center",
            va=va,
            fontsize=10,
            color=color,
            fontweight="bold",
        )

    ax2.bar(
        [0 - w / 2, 0 + w / 2],
        [cov_rate_d, cov_rate_p],
        width=w,
        color=[COLOR_DIFFSBDD, COLOR_POCKET2MOL],
    )
    ax2.set_xticks([0 - w / 2, 0 + w / 2])
    ax2.set_xticklabels(["DiffSBDD", "Pocket2Mol"], fontsize=11)
    ax2.set_ylabel("Coverage (dimensionless)")
    ax2.set_title("Coverage: original condition produced at least one valid molecule", fontweight="normal")
    ax2.set_xlim(-0.55, 0.55)
    ax2.set_ylim(0, 1.12)
    for xi, rate, n_cov, n_tot in zip(
        [0 - w / 2, 0 + w / 2],
        [cov_rate_d, cov_rate_p],
        [n_cov_d, n_cov_p],
        [n_tot_d, n_tot_p],
    ):
        label = f"{n_cov}/{n_tot}\n({rate:.2f})"
        if rate >= 0.55:
            y_text = rate * 0.45
            color = "white"
            va = "center"
        else:
            y_text = rate + 0.04
            color = "#222"
            va = "bottom"
        ax2.text(
            xi,
            y_text,
            label,
            ha="center",
            va=va,
            fontsize=9,
            color=color,
            linespacing=1.1,
        )

    pdf, png = save_figure(fig, out_pdf.parent / out_pdf.stem)
    plt.close(fig)
    print(f"Wrote {pdf}")
    print(f"Wrote {png}")
    print(
        f"Mean validity (original, {validity_note}): "
        f"DiffSBDD={mean_v_d:.4f}, Pocket2Mol={mean_v_p:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
