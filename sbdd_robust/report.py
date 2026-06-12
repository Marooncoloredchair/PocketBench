"""Reusable reliability-report metrics for PocketBench.

This module is the **single source of truth** for the three headline analyses, so
they can be computed identically from the benchmark CLI, the standalone scripts in
``analysis/``, and — most importantly — by external users on *their own model's*
per-condition metrics CSV, with no GPU and no PocketBench generation run required.

Standardized input
------------------
A "metrics" DataFrame has one row per (pocket, perturbation condition) with columns:

    pocket_id            str    target identifier
    perturbation_tag     str    one of: original, atom_shuffle, coordinate_jitter,
                                 crop_radius_plus_<x>, crop_radius_minus_<x>, ...
                                 (``perturbation_type`` accepted as a fallback)
    model_name           str    optional; if present you can filter to one model
    validity             float  [0,1]
    uniqueness           float  [0,1]
    mean_qed, std_qed    float  [0,1]
    mean_sa, std_sa      float  RDKit SA (~1-10)   (optional)

Bring-your-own-model: produce this CSV from your generator's outputs (RDKit QED/SA on
the molecules sampled for each perturbed pocket) and PocketBench will report
brittleness, the paired crop-radius test, and the Pocket Boundary Sensitivity Index.
"""

from __future__ import annotations

import re
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from sbdd_robust.metrics.brittleness_rate import brittleness_rate_from_flagged
from sbdd_robust.metrics.robustness_score import flag_invariant_brittleness

NORMALIZED_METRICS = ["validity", "uniqueness", "mean_qed", "std_qed"]
INVARIANT_TAGS = ["atom_shuffle", "coordinate_jitter", "crop_radius_plus_1.5", "crop_radius_minus_1.5"]
NOISE_TAGS = ["atom_shuffle", "coordinate_jitter"]
DEFAULT_TAUS = [0.05, 0.10, 0.15, 0.20]

_CROP_RE = re.compile(r"crop_radius_(plus|minus)_([0-9]+(?:\.[0-9]+)?)")


def tag_col(df: pd.DataFrame) -> str:
    return "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"


def tag_to_offset(tag: str) -> float | None:
    """Crop-radius offset (Å) encoded by a perturbation tag; None if not a radius condition."""
    t = str(tag)
    if t == "original":
        return 0.0
    m = _CROP_RE.fullmatch(t)
    if not m:
        return None
    return (1.0 if m.group(1) == "plus" else -1.0) * float(m.group(2))


def available_crop_minus_tags(df: pd.DataFrame) -> list[str]:
    """Crop-minus tags present, ordered by ascending offset magnitude."""
    tags = set(df[tag_col(df)].astype(str))
    minus = [(abs(tag_to_offset(t)), t) for t in tags if t.startswith("crop_radius_minus_")]
    return [t for _, t in sorted(minus)]


def resolve_crop_tag(df: pd.DataFrame, requested: str) -> str | None:
    """Return the requested crop tag if present, else the available crop-minus tag
    closest to 1.5 Å, else None."""
    if requested in set(df[tag_col(df)].astype(str)):
        return requested
    avail = available_crop_minus_tags(df)
    if not avail:
        return None
    return min(avail, key=lambda t: abs(abs(tag_to_offset(t)) - 1.5))


def filter_model(df: pd.DataFrame, model: str | None) -> pd.DataFrame:
    if model and "model_name" in df.columns:
        return df[df["model_name"].astype(str).str.lower() == model.lower()].copy()
    return df.copy()


# --------------------------------------------------------------------------- #
# Normalized brittleness                                                       #
# --------------------------------------------------------------------------- #
def normalized_brittleness(
    df: pd.DataFrame,
    taus: Iterable[float] = DEFAULT_TAUS,
    metrics: list[str] | None = None,
    invariant_tags: list[str] | None = None,
    dataset: str = "dataset",
) -> pd.DataFrame:
    """Brittleness rate per τ using the normalized metric subset only."""
    metrics = [m for m in (metrics or NORMALIZED_METRICS) if m in df.columns]
    inv = invariant_tags or INVARIANT_TAGS
    rows: list[dict] = []
    for tau in taus:
        flagged = flag_invariant_brittleness(
            df, invariant_tags=inv, original_tag="original",
            metric_std_threshold=float(tau), metrics_subset=metrics,
        )
        st = brittleness_rate_from_flagged(flagged)
        rows.append({
            "dataset": dataset,
            "tau": float(tau),
            "metrics": ",".join(metrics),
            "n_brittle": int(st["brittle_pairs"]),
            "total": int(st["total_pairs"]),
            "brittleness_rate": round(float(st["brittleness_rate"]), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Paired crop-radius Wilcoxon test                                             #
# --------------------------------------------------------------------------- #
def _paired_arrays(df: pd.DataFrame, metric: str, crop_tag: str) -> tuple[np.ndarray, np.ndarray]:
    df = df.copy()
    df["pocket_id"] = df["pocket_id"].astype(str)
    tcol = tag_col(df)
    orig = df[df[tcol].astype(str) == "original"].drop_duplicates("pocket_id").set_index("pocket_id")
    crop = df[df[tcol].astype(str) == crop_tag].drop_duplicates("pocket_id").set_index("pocket_id")
    common = sorted(set(orig.index) & set(crop.index))
    ov, cv = [], []
    for pid in common:
        o = pd.to_numeric(orig.loc[pid, metric], errors="coerce")
        c = pd.to_numeric(crop.loc[pid, metric], errors="coerce")
        if pd.isna(o) or pd.isna(c):
            continue
        ov.append(float(o))
        cv.append(float(c))
    return np.asarray(cv), np.asarray(ov)


def _rank_biserial(delta: np.ndarray) -> float:
    d = delta[delta != 0]
    if d.size == 0:
        return float("nan")
    ranks = stats.rankdata(np.abs(d))
    r_pos = ranks[d > 0].sum()
    r_neg = ranks[d < 0].sum()
    total = r_pos + r_neg
    return float((r_pos - r_neg) / total) if total else float("nan")


def crop_paired_wilcoxon(
    df: pd.DataFrame,
    crop_tag: str = "crop_radius_minus_1.5",
    metrics: tuple[tuple[str, str], ...] = (("mean_qed", "dQED"), ("mean_sa", "dSA")),
    n_boot: int = 10000,
    seed: int = 20260612,
    dataset: str = "dataset",
) -> pd.DataFrame:
    """Paired two-sided Wilcoxon signed-rank test of (crop_tag − original) per metric."""
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for metric, label in metrics:
        if metric not in df.columns:
            continue
        crop, orig = _paired_arrays(df, metric, crop_tag)
        if crop.size < 1:
            continue
        delta = crop - orig
        n = int(delta.size)
        n_nz = int(np.sum(delta != 0))
        if n_nz == 0:
            W, p, z, r_z = float("nan"), 1.0, float("nan"), float("nan")
        else:
            res = stats.wilcoxon(crop, orig, zero_method="wilcox", alternative="two-sided")
            W, p = float(res.statistic), float(res.pvalue)
            ranks = stats.rankdata(np.abs(delta[delta != 0]))
            r_pos = ranks[delta[delta != 0] > 0].sum()
            mean_w = n_nz * (n_nz + 1) / 4.0
            std_w = np.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0)
            z = (r_pos - mean_w) / std_w if std_w > 0 else float("nan")
            r_z = abs(z) / np.sqrt(n_nz)
        rbc = _rank_biserial(delta)
        if n_boot and n >= 2:
            idx = rng.integers(0, n, size=(n_boot, n))
            med = np.median(delta[idx], axis=1)
            mean = np.mean(delta[idx], axis=1)
            med_lo, med_hi = np.percentile(med, [2.5, 97.5])
            mean_lo, mean_hi = np.percentile(mean, [2.5, 97.5])
        else:
            med_lo = med_hi = mean_lo = mean_hi = float("nan")
        rows.append({
            "dataset": dataset, "metric": label, "comparison": f"{crop_tag} vs original",
            "n_pairs": n, "n_zero_diff": int(np.sum(delta == 0)),
            "mean_delta": round(float(np.mean(delta)), 4), "median_delta": round(float(np.median(delta)), 4),
            "wilcoxon_W": round(W, 2) if not np.isnan(W) else np.nan, "p_value": p,
            "rank_biserial_r": round(rbc, 4), "z": round(float(z), 4) if not np.isnan(z) else np.nan,
            "effect_size_r_z": round(float(r_z), 4) if not np.isnan(r_z) else np.nan,
            "median_ci95_lo": round(float(med_lo), 4), "median_ci95_hi": round(float(med_hi), 4),
            "mean_ci95_lo": round(float(mean_lo), 4), "mean_ci95_hi": round(float(mean_hi), 4),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Pocket Boundary Sensitivity Index (PBSI)                                     #
# --------------------------------------------------------------------------- #
def pocket_boundary_sensitivity(
    df: pd.DataFrame,
    metric: str = "mean_qed",
    noise_tags: list[str] | None = None,
    dataset: str = "dataset",
) -> tuple[pd.DataFrame, dict]:
    """Per-pocket QED-vs-crop-radius slope (PBSI) + boundary-to-noise SNR; returns (per_pocket, summary)."""
    noise_tags = noise_tags or NOISE_TAGS
    df = df.copy()
    df["pocket_id"] = df["pocket_id"].astype(str)
    tcol = tag_col(df)
    df["_tag"] = df[tcol].astype(str)
    df["_offset"] = df["_tag"].map(tag_to_offset)
    df[metric] = pd.to_numeric(df[metric], errors="coerce")

    recs: list[dict] = []
    for pid, grp in df.groupby("pocket_id", sort=True):
        radius_rows = grp[grp["_offset"].notna()].drop_duplicates("_offset")
        radius_rows = radius_rows[radius_rows[metric].notna()]
        if radius_rows["_offset"].nunique() < 2:
            continue
        x = radius_rows["_offset"].to_numpy(dtype=float)
        y = radius_rows[metric].to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, y, 1)
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        by_off = dict(zip(np.round(x, 3), y))
        curvature = float("nan")
        pos = sorted(o for o in by_off if o > 0)
        neg = sorted((o for o in by_off if o < 0), reverse=True)
        if 0.0 in by_off and pos and neg and abs(pos[0]) == abs(neg[0]):
            d = pos[0]
            curvature = float(by_off[d] + by_off[-d] - 2 * by_off[0.0])
        boundary_range = float(np.max(y) - np.min(y))

        noise_rows = grp[grp["_tag"].isin(noise_tags)]
        nvals = pd.to_numeric(noise_rows[metric], errors="coerce").dropna().to_numpy()
        noise_std = float(np.std(nvals)) if nvals.size >= 2 else float("nan")
        swing = abs(slope) * 1.5
        snr_1p5 = swing / noise_std if noise_std and noise_std > 0 else float("nan")
        snr_range = boundary_range / noise_std if noise_std and noise_std > 0 else float("nan")

        recs.append({
            "dataset": dataset, "pocket_id": pid, "n_radii": int(radius_rows["_offset"].nunique()),
            "qed_at_minus": float(by_off.get(-1.5, np.nan)), "qed_at_orig": float(by_off.get(0.0, np.nan)),
            "qed_at_plus": float(by_off.get(1.5, np.nan)),
            "pbsi_slope_qed_per_A": round(float(slope), 5), "abs_slope": round(abs(float(slope)), 5),
            "fit_r2": round(float(r2), 4),
            "curvature": round(curvature, 5) if not np.isnan(curvature) else np.nan,
            "boundary_range": round(boundary_range, 5),
            "noise_std": round(noise_std, 5) if not np.isnan(noise_std) else np.nan,
            "snr_1.5A": round(snr_1p5, 3) if not np.isnan(snr_1p5) else np.nan,
            "snr_range": round(snr_range, 3) if not np.isnan(snr_range) else np.nan,
        })
    pp = pd.DataFrame(recs)
    summary = _pbsi_summary(pp, dataset)
    return pp, summary


# --------------------------------------------------------------------------- #
# Initialization Sensitivity Ratio (ISR)                                        #
# --------------------------------------------------------------------------- #
# The ISR metric lives in ``sbdd_robust.metrics.initialization_sensitivity`` (kept
# self-contained so it can be reused without ``report``). It is re-exported here so the
# CLI and analysis scripts keep a single ``report.initialization_sensitivity`` entry
# point alongside the other reliability analyses.
from sbdd_robust.metrics.initialization_sensitivity import (  # noqa: E402
    FEATURIZATION_TAGS,
    FRAME_TAG_PREFIXES,
    initialization_sensitivity,
)


def _pbsi_summary(pp: pd.DataFrame, dataset: str) -> dict:
    if pp.empty:
        return {"dataset": dataset, "n_pockets": 0}
    s = pp["pbsi_slope_qed_per_A"].to_numpy()
    asl = pp["abs_slope"].to_numpy()
    snr = pp["snr_1.5A"].dropna().to_numpy()
    return {
        "dataset": dataset,
        "n_pockets": int(len(pp)),
        "PBSI_median_abs_slope_qed_per_A": round(float(np.median(asl)), 5),
        "median_signed_slope": round(float(np.median(s)), 5),
        "iqr_abs_slope": round(float(np.percentile(asl, 75) - np.percentile(asl, 25)), 5),
        "frac_shrinking_lowers_qed": round(float(np.mean(s > 0)), 4),
        "median_boundary_range_qed": round(float(np.median(pp["boundary_range"])), 5),
        "median_noise_std_qed": round(float(np.nanmedian(pp["noise_std"])), 5) if pp["noise_std"].notna().any() else float("nan"),
        "median_snr_1.5A": round(float(np.median(snr)), 3) if snr.size else float("nan"),
        "frac_boundary_exceeds_noise": round(float(np.mean(snr > 1.0)), 4) if snr.size else float("nan"),
    }
