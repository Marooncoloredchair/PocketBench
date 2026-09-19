#!/usr/bin/env python3
"""Pre-registered permutation test on PBSI (runreal100 / diffsbdd).

STEP 1 — DECLARE (frozen before execution; do not edit after hash):
-----------------------------------------------------------------
DATA: data/results/metrics_per_condition__runreal100.csv, diffsbdd, 99 pockets.
BASELINE (fixed; do not recompute differently):
  PBSI_median_abs_slope_qed_per_A = 0.0226
  median_signed_slope = 0.01858

NULL:
  Within each pocket, permute mean_qed across the three radius-bearing rows
  (original, crop_radius_plus_1.5, crop_radius_minus_1.5). Nothing else changes.
  N = 200 draws.
  Seeds: exactly range(200). Any seed set is iterated as sorted(set(...)),
  never bare set() iteration.

REPORT (both statistics):
  null median, null p5/p50/p95, baseline z-score, empirical p =
  fraction of draws at least as extreme as baseline.
  - For PBSI_median_abs_slope_qed_per_A (unsigned): extreme = null_value >= baseline
  - For median_signed_slope: extreme = abs(null_value) >= abs(baseline)

INTERPRETATION RULE (headline uses abs PBSI only):
  |z| < 2.0  -> HEADLINE_NOT_SEPARABLE
  |z| >= 2.0 -> HEADLINE_SURVIVES
  median_signed_slope is reported but does not decide the headline verdict.

CONTROLS (STEP 2): both must behave or CONTROL_FAILED and stop.
  POSITIVE: mean_qed has known linear dependence on offset; baseline outside null.
  NEGATIVE: mean_qed independent of tag; baseline inside null.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sbdd_robust.report import pocket_boundary_sensitivity, tag_col, tag_to_offset

# --- frozen baseline (do not recompute differently) ---
BASELINE_ABS = 0.0226
BASELINE_SIGNED = 0.01858
N_DRAWS = 200
SEEDS = list(range(200))  # iterated via sorted(set(SEEDS))
RADIUS_TAGS = ("original", "crop_radius_plus_1.5", "crop_radius_minus_1.5")
SRC = Path("data/results/metrics_per_condition__runreal100.csv")


def _stats(null_vals: np.ndarray, baseline: float, unsigned: bool) -> dict:
    null_vals = np.asarray(null_vals, dtype=float)
    med = float(np.median(null_vals))
    p5, p50, p95 = [float(x) for x in np.percentile(null_vals, [5, 50, 95])]
    sd = float(np.std(null_vals, ddof=1))
    z = float((baseline - float(np.mean(null_vals))) / sd) if sd > 0 else float("nan")
    if unsigned:
        extreme = null_vals >= baseline
    else:
        extreme = np.abs(null_vals) >= abs(baseline)
    p = float(np.mean(extreme))
    return {
        "null_median": med,
        "null_p5": p5,
        "null_p50": p50,
        "null_p95": p95,
        "null_mean": float(np.mean(null_vals)),
        "null_sd": sd,
        "z": z,
        "empirical_p": p,
        "n": int(null_vals.size),
    }


def _permute_radius_qed(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    out = df.copy()
    tcol = tag_col(out)
    out["_tag"] = out[tcol].astype(str)
    rng = np.random.default_rng(seed)
    # sorted pocket ids for deterministic row-group order given seed
    for pid in sorted(out["pocket_id"].astype(str).unique()):
        sub = out[out["pocket_id"].astype(str) == pid]
        ridx = sub.index[sub["_tag"].isin(RADIUS_TAGS)].tolist()
        if len(ridx) < 2:
            continue
        vals = out.loc[ridx, "mean_qed"].to_numpy(dtype=float).copy()
        rng.shuffle(vals)
        out.loc[ridx, "mean_qed"] = vals
    return out.drop(columns=["_tag"])


def _null_distribution(df: pd.DataFrame, seeds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    abs_vals = []
    signed_vals = []
    for seed in sorted(set(seeds)):
        dfp = _permute_radius_qed(df, seed)
        _, summary = pocket_boundary_sensitivity(dfp, dataset=f"perm_{seed}")
        abs_vals.append(summary["PBSI_median_abs_slope_qed_per_A"])
        signed_vals.append(summary["median_signed_slope"])
    return np.asarray(abs_vals, float), np.asarray(signed_vals, float)


def _synthesize_positive(template: pd.DataFrame) -> pd.DataFrame:
    """Known linear dependence: mean_qed = 0.5 + 0.04 * offset_A (+ tiny noise)."""
    df = template.copy()
    tcol = tag_col(df)
    rng = np.random.default_rng(99901)
    offs = df[tcol].astype(str).map(tag_to_offset)
    # only set radius rows; leave noise tags as in template (unused by slope)
    mask = offs.notna()
    df.loc[mask, "mean_qed"] = 0.5 + 0.04 * offs[mask].to_numpy(dtype=float) + rng.normal(
        0.0, 1e-6, size=int(mask.sum())
    )
    return df


def _synthesize_negative(template: pd.DataFrame) -> pd.DataFrame:
    """mean_qed independent of tag: i.i.d. draws per radius row."""
    df = template.copy()
    tcol = tag_col(df)
    rng = np.random.default_rng(99902)
    offs = df[tcol].astype(str).map(tag_to_offset)
    mask = offs.notna()
    df.loc[mask, "mean_qed"] = rng.uniform(0.2, 0.8, size=int(mask.sum()))
    return df


def _control_baseline_and_null(df: pd.DataFrame, label: str) -> tuple[dict, dict, dict]:
    _, s = pocket_boundary_sensitivity(df, dataset=label)
    abs_null, signed_null = _null_distribution(df, SEEDS)
    abs_base = s["PBSI_median_abs_slope_qed_per_A"]
    signed_base = s["median_signed_slope"]
    return (
        s,
        _stats(abs_null, abs_base, unsigned=True),
        _stats(signed_null, signed_base, unsigned=False),
    )


def main() -> int:
    here = Path(__file__).resolve()
    digest = hashlib.sha256(here.read_bytes()).hexdigest()
    print(f"SCRIPT_SHA256={digest}")
    print(f"SCRIPT_PATH={here}")

    df = pd.read_csv(SRC)
    assert df["model_name"].str.lower().eq("diffsbdd").all()
    assert df["pocket_id"].nunique() == 99

    # --- STEP 2: controls ---
    pos = _synthesize_positive(df)
    pos_sum, pos_abs, pos_signed = _control_baseline_and_null(pos, "control_positive")
    # "far outside null": |z| >= 2 AND empirical_p <= 0.05 on abs headline
    pos_ok = abs(pos_abs["z"]) >= 2.0 and pos_abs["empirical_p"] <= 0.05

    neg = _synthesize_negative(df)
    neg_sum, neg_abs, neg_signed = _control_baseline_and_null(neg, "control_negative")
    # "inside null": |z| < 2 OR empirical_p > 0.05 on abs headline
    neg_ok = abs(neg_abs["z"]) < 2.0 or neg_abs["empirical_p"] > 0.05

    print("CONTROL_POSITIVE_BASELINE", pos_sum)
    print("CONTROL_POSITIVE_ABS", pos_abs)
    print("CONTROL_POSITIVE_SIGNED", pos_signed)
    print("CONTROL_POSITIVE_OK", pos_ok)
    print("CONTROL_NEGATIVE_BASELINE", neg_sum)
    print("CONTROL_NEGATIVE_ABS", neg_abs)
    print("CONTROL_NEGATIVE_SIGNED", neg_signed)
    print("CONTROL_NEGATIVE_OK", neg_ok)

    if not (pos_ok and neg_ok):
        print("CONTROL_FAILED")
        return 2

    # --- STEP 3: real data null ---
    abs_null, signed_null = _null_distribution(df, SEEDS)
    real_abs = _stats(abs_null, BASELINE_ABS, unsigned=True)
    real_signed = _stats(signed_null, BASELINE_SIGNED, unsigned=False)
    print("REAL_ABS", real_abs)
    print("REAL_SIGNED", real_signed)

    # --- STEP 4: verdict (abs only) ---
    z = real_abs["z"]
    if abs(z) < 2.0:
        verdict = "HEADLINE_NOT_SEPARABLE"
    else:
        verdict = "HEADLINE_SURVIVES"
    print(f"VERDICT={verdict}")
    print(f"HEADLINE_Z={z}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
