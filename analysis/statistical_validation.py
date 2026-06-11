#!/usr/bin/env python3
"""
Statistical validation from ``metrics_per_condition`` CSV: paired Wilcoxon on validity
deltas, mean absolute jitter validity deviation (one-sample *t*-test + CI for the mean),
Wilcoxon on cross-condition spread of validity among the four stresses, bootstrap
brittleness rate at fixed ``tau``, and an IID pooled-resampling permutation check on validity dispersion.

Loads a ``metrics_per_condition`` CSV filtered to one ``model_name`` (default DiffSBDD).
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

DEFAULT_INV = (
    "atom_shuffle",
    "coordinate_jitter",
    "crop_radius_plus_1.5",
    "crop_radius_minus_1.5",
)


def pocket_brittle_flags(df: pd.DataFrame, tau: float, invariant_tags: tuple[str, ...]) -> tuple[np.ndarray, int]:
    flagged = flag_invariant_brittleness(
        df,
        invariant_tags=list(invariant_tags),
        original_tag="original",
        metric_std_threshold=float(tau),
    )
    sub = flagged.drop_duplicates(subset=["pocket_id", "model_name"])
    return sub["brittle_invariant"].to_numpy(dtype=bool), int(sub.shape[0])


def _mean_ci_t(vals: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Classic two-sided *t*-interval for E[X] given i.i.d.-ish sample."""
    vals = vals[np.isfinite(vals)]
    n = int(vals.size)
    if n == 0:
        return float("nan"), float("nan")
    m = float(np.mean(vals))
    if n == 1:
        return m, m
    sem = float(np.std(vals, ddof=1) / math.sqrt(n))
    df_deg = n - 1
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df_deg))
    delta = t_crit * sem
    return m - delta, m + delta


def invariant_validity_matrix(
    df: pd.DataFrame, pockets: list[str], model_lower: str, invariant_tags: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Rows = pockets order; cols = invariant tags. NaN if a cell is missing."""
    n_p = len(pockets)
    mat = np.full((n_p, len(invariant_tags)), np.nan, dtype=np.float64)
    dsub = df[df["model_name"].astype(str).str.lower() == model_lower.lower()]
    for i, pid in enumerate(pockets):
        for j, tag in enumerate(invariant_tags):
            sel = (dsub["pocket_id"].astype(str) == pid) & (dsub["perturbation_tag"] == tag)
            if sel.any():
                mat[i, j] = float(pd.to_numeric(dsub.loc[sel].iloc[0]["validity"], errors="coerce"))
    complete = np.all(np.isfinite(mat), axis=1)
    return mat, complete


def pocket_sigma_population(vals: np.ndarray) -> np.ndarray:
    """Per-row sigma with ddof=0 (matches brittle-column ``numpy.std`` defaults)."""
    return np.asarray(np.std(vals, axis=1, ddof=0), dtype=np.float64)


def iid_draw_sigma_permutation_exceedances(
    mat_complete: np.ndarray,
    *,
    rng: np.random.Generator,
    n_perm: int,
    percentile: float = 95.0,
) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Unconditional-exchangeability envelope: flatten the observed validity surface,
    and for each pocket row independently redraw four IID samples with replacement
    from that pooled multiset, computing sigma (=population std ddof 0).

    Repeated ``n_perm`` times builds pocket-specific null envelopes; compares each
    observed pocket sigma against the percentile of its envelope.
    """
    n_p, n_c = mat_complete.shape
    pool = mat_complete.ravel(order="C")
    null_sigmas = np.zeros((n_p, n_perm), dtype=np.float64)
    for b in range(n_perm):
        draws = rng.choice(pool, size=(n_p, n_c), replace=True)
        null_sigmas[:, b] = pocket_sigma_population(draws)
    obs_sigma = pocket_sigma_population(mat_complete)
    thr = np.percentile(null_sigmas, percentile, axis=1).astype(np.float64)
    frac = float(np.mean(obs_sigma > thr))
    return frac, obs_sigma, thr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=_ROOT / "paper" / "statistical_validation.csv")
    ap.add_argument("--model", type=str, default="diffsbdd")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--perms", type=int, default=1000, help="Permutation repeats for dispersion null.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model_lower = args.model.lower()
    df = pd.read_csv(args.metrics)
    df = df[df["model_name"].astype(str).str.lower() == model_lower]

    pockets = sorted(df["pocket_id"].astype(str).unique())
    inv_mat, inv_complete = invariant_validity_matrix(df, pockets, model_lower, DEFAULT_INV)
    mat_complete = inv_mat[inv_complete]
    sigma_per_pocket_complete = pocket_sigma_population(mat_complete)

    rows: list[dict[str, float | str | int]] = []

    # (1) Paired Wilcoxon: pocket-level validity original vs each invariant tag
    for tag in DEFAULT_INV:
        vo: list[float] = []
        vp: list[float] = []
        for pid in pockets:
            o = df[(df["pocket_id"].astype(str) == pid) & (df["perturbation_tag"] == "original")]
            p = df[(df["pocket_id"].astype(str) == pid) & (df["perturbation_tag"] == tag)]
            if o.empty or p.empty:
                continue
            vo.append(float(pd.to_numeric(o.iloc[0]["validity"], errors="coerce")))
            vp.append(float(pd.to_numeric(p.iloc[0]["validity"], errors="coerce")))
        npairs = len(vo)
        if npairs < 3:
            stat_w, p_w = float("nan"), float("nan")
        else:
            diff_v = np.asarray(vp) - np.asarray(vo)
            if np.all(diff_v == 0):
                p_w = 1.0
                stat_w = 0.0
            else:
                stat_w, p_w = stats.wilcoxon(diff_v, zero_method="wilcox", mode="auto")
        rows.append(
            {
                "test": "paired_wilcoxon_validity_delta",
                "contrast": f"original_vs_{tag}",
                "n_paired_pockets": npairs,
                "statistic": stat_w,
                "p_value": p_w,
                "note": "Wilcoxon signed-rank on (validity_pert - validity_orig) per pocket.",
            }
        )

    # (2) Absolute validity deviation original vs jitter: one-sample t on mean MAD > 0
    mad_vals: list[float] = []
    for pid in pockets:
        o = df[(df["pocket_id"].astype(str) == pid) & (df["perturbation_tag"] == "original")]
        p = df[(df["pocket_id"].astype(str) == pid) & (df["perturbation_tag"] == "coordinate_jitter")]
        if o.empty or p.empty:
            continue
        vo = float(pd.to_numeric(o.iloc[0]["validity"], errors="coerce"))
        vj = float(pd.to_numeric(p.iloc[0]["validity"], errors="coerce"))
        if not math.isfinite(vo) or not math.isfinite(vj):
            continue
        mad_vals.append(abs(vj - vo))
    mad_arr = np.asarray(mad_vals, dtype=np.float64)
    n_mad = int(mad_arr.size)
    if n_mad >= 2:
        res_t = stats.ttest_1samp(mad_arr, popmean=0.0, alternative="greater")
        t_stat = float(res_t.statistic)
        p_t = float(res_t.pvalue)
        mean_mad = float(np.mean(mad_arr))
        mad_lo, mad_hi = _mean_ci_t(mad_arr)
    else:
        t_stat = float("nan")
        p_t = float("nan")
        mean_mad = float(np.nanmean(mad_arr)) if n_mad else float("nan")
        mad_lo, mad_hi = float("nan"), float("nan")

    rows.append(
        {
            "test": "onesample_t_mean_abs_validity_delta_jitter",
            "contrast": "coordinate_jitter_vs_original",
            "n_paired_pockets": n_mad,
            "statistic": t_stat if math.isfinite(t_stat) else float("nan"),
            "p_value": p_t if math.isfinite(p_t) else float("nan"),
            "note": (
                f"H0 mean|valid_jitter-valid_orig|=0 alternative greater; "
                f"mean_MAD={mean_mad:.6g}; CI95(mean_MAD)=({mad_lo:.6g},{mad_hi:.6g})"
            ),
        }
    )

    # (3) Variance: sigma of validity over four invariant stresses - Wilcoxon vs 0
    n_sigma = int(sigma_per_pocket_complete.size)
    median_sigma = float(np.median(sigma_per_pocket_complete)) if n_sigma else float("nan")
    wil_stat = float("nan")
    wil_p = float("nan")
    if n_sigma > 0:
        sig = sigma_per_pocket_complete
        if np.all(sig <= 1e-14):
            wil_stat, wil_p = 0.0, 1.0
        elif np.any(sig > 0):
            try:
                wres = stats.wilcoxon(sig, zero_method="wilcox", alternative="greater", mode="auto")
                wil_stat = float(wres.statistic)
                wil_p = float(wres.pvalue)
            except ValueError:
                wil_stat = float("nan")
                wil_p = float("nan")

    rows.append(
        {
            "test": "wilcoxon_validity_sigma_across_four_stresses",
            "contrast": "atom_shuffle+jitter+crop+/crop- only",
            "n_paired_pockets": n_sigma,
            "statistic": wil_stat if math.isfinite(wil_stat) else float("nan"),
            "p_value": wil_p if math.isfinite(wil_p) else float("nan"),
            "note": (
                "Wilcoxon signed-rank on pocket sigma(validity across four invariant stresses); "
                "alternative median sigma > 0; sigma ddof=0 matches brittle-flag code; "
                f"median_sigma={median_sigma:.6g}"
            ),
        }
    )

    # (4) Bootstrap brittleness rate at tau (unchanged)
    tau = 0.10
    mask, n_tot = pocket_brittle_flags(df, tau, DEFAULT_INV)
    n_b = int(mask.sum())
    rate = n_b / n_tot if n_tot else float("nan")
    boots: list[float] = []
    ix = np.arange(len(mask))
    for _ in range(args.bootstrap):
        samp = rng.choice(ix, size=len(ix), replace=True)
        boots.append(float(mask[samp].mean()))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    rows.append(
        {
            "test": "bootstrap_brittleness_rate",
            "contrast": f"tau={tau}",
            "n_paired_pockets": n_tot,
            "statistic": rate if math.isfinite(rate) else float("nan"),
            "p_value": float("nan"),
            "note": f"Bootstrap 95% CI on brittle fraction: [{lo:.4f}, {hi:.4f}]",
        }
    )
    rows.append(
        {
            "test": "bootstrap_ci_low",
            "contrast": f"tau={tau}",
            "n_paired_pockets": n_tot,
            "statistic": lo,
            "p_value": float("nan"),
            "note": "2.5th percentile brittle-fraction bootstrap",
        }
    )
    rows.append(
        {
            "test": "bootstrap_ci_high",
            "contrast": f"tau={tau}",
            "n_paired_pockets": n_tot,
            "statistic": hi,
            "p_value": float("nan"),
            "note": "97.5th percentile brittle-fraction bootstrap",
        }
    )

    # (5) Permutation dispersion test
    frac_exceed, _, _thr = iid_draw_sigma_permutation_exceedances(
        mat_complete, rng=rng, n_perm=args.perms, percentile=95.0
    )
    rows.append(
        {
            "test": "permutation_iid_pool_validity_sigma_vs_p95_null",
            "contrast": f"n_perm={args.perms}; four_IID_draws_from_pooled_validities",
            "n_paired_pockets": n_sigma,
            "statistic": frac_exceed,
            "p_value": float("nan"),
            "note": (
                f"IID with-replacement quartet per pseudopocket-row from pooled {mat_complete.shape[0]} x {mat_complete.shape[1]} "
                f"({args.perms} reps); upper-tail fraction counts obs_sigma > row p95 of null. Values near "
                f"zero indicate cohesion (typical IID draws are MORE dispersed than curated pocket rows)."
            ),
        }
    )

    out = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print("Wrote", args.out.resolve())
    print(out.to_string(index=False))

    print()
    print("----- Plain-English summary -----")
    print(
        "Paired Wilcoxon on signed validity deltas (perturbation minus original): "
        "see four rows paired_wilcoxon_validity_delta."
    )

    if n_mad >= 2 and math.isfinite(p_t):
        sig_mad = (
            "The mean absolute validity shift is statistically above zero (one-sided alpha=0.05)."
            if float(p_t) < 0.05
            else "We do not reject a mean absolute deviation of zero at one-sided alpha=0.05."
        )
        print(
            f"Jitter sanity check: Across {n_mad} pockets the average |valid_jitter-valid_orig| is "
            f"{mean_mad:.4g} (two-sided 95% CI for that mean [{mad_lo:.4g}, {mad_hi:.4g}]). "
            f"One-sample t-test versus mean=0 yields p={p_t:.4g}; {sig_mad}"
        )
    else:
        print("Insufficient paired jitter-vs-original validity rows for the t-test on mean MAD.")

    if math.isfinite(wil_p):
        sig_wil = (
            "Pocket-wise validity dispersion tends to lie above Wilcoxon's null reference at zero (p < 0.05)."
            if wil_p < 0.05
            else "Wilcoxon's evidence against a zero-median dispersion is not significant at alpha=0.05."
        )
        print(
            f"Across {n_sigma} pockets with completeness across the four invariant stresses, per-pocket "
            f"dispersion sigma(validity), ddof=0, has median {median_sigma:.4g}. Signed-rank p={wil_p:.4g}: "
            f"{sig_wil}"
        )

    print(
        f"Brittle fraction at tau={tau} is {rate:.4f}; bootstrap 95% interval "
        f"[{lo:.4f}, {hi:.4f}] summarises sampling uncertainty among covered pockets."
    )
    pct_explain = frac_exceed * 100.0 if math.isfinite(frac_exceed) else float("nan")
    print(
        "IID quartet resampling from the pooled multiset simulates unstructured draws "
        "(\"label-scrambled\" cohesion missing). Fraction with obs sigma above each pocket-row's "
        f"perm p95 benchmark is {frac_exceed:.4f} (~{pct_explain:.2f}%); values near zero here mean observed "
        f"pockets generally look LESS dispersed than the upper tail of unstructured quartets, not evidence of invariance."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
