"""Compare metrics across original vs perturbed pockets; flag brittleness."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np
import pandas as pd


def _metric_columns(df: pd.DataFrame) -> List[str]:
    skip = {
        "pocket_id",
        "perturbation_type",
        "perturbation_tag",
        "model_name",
        "run_id",
        "brittle_invariant",
        "brittleness_note",
    }
    numeric = []
    for c in df.columns:
        if c in skip:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric.append(c)
    return numeric


def flag_invariant_brittleness(
    df: pd.DataFrame,
    invariant_tags: Iterable[str],
    original_tag: str = "original",
    metric_std_threshold: float = 0.1,
    metrics_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each (pocket_id, model_name), compare numeric metrics on the original pocket
    to the distribution across **invariant** perturbation tags.

    Rows inherit ``brittle_invariant`` True if, for any numeric metric, the standard
    deviation across invariant perturbations exceeds ``metric_std_threshold``.
    """
    inv = set(invariant_tags)
    out = df.copy()
    tag_col = "perturbation_tag" if "perturbation_tag" in out.columns else "perturbation_type"
    out["_ptag"] = out[tag_col].astype(str)

    metrics = metrics_subset or _metric_columns(out)
    key_brittle: dict[tuple[Any, Any], bool] = {}
    key_note: dict[tuple[Any, Any], str] = {}

    for (pid, model), grp in out.groupby(["pocket_id", "model_name"], sort=False):
        inv_rows = grp[grp["_ptag"].isin(inv)]
        note_parts: list[str] = []
        is_brittle = False
        if inv_rows.shape[0] < 2:
            key_brittle[(pid, model)] = False
            key_note[(pid, model)] = "insufficient_invariant_rows"
            continue
        for m in metrics:
            if m not in inv_rows.columns:
                continue
            vals = inv_rows[m].astype(float).values
            vals = vals[~np.isnan(vals)]
            if vals.size < 2:
                continue
            std = float(np.std(vals))
            if std > metric_std_threshold:
                is_brittle = True
                note_parts.append(f"{m}:std={std:.4f}")
        key_brittle[(pid, model)] = is_brittle
        key_note[(pid, model)] = ";".join(note_parts) if note_parts else ""

    out["brittle_invariant"] = [
        key_brittle.get((r["pocket_id"], r["model_name"]), False) for _, r in out.iterrows()
    ]
    out["brittleness_note"] = [
        key_note.get((r["pocket_id"], r["model_name"]), "") for _, r in out.iterrows()
    ]
    out.drop(columns=["_ptag"], inplace=True, errors="ignore")
    return out


def summarize_robustness_vs_original(
    df: pd.DataFrame,
    invariant_tags: Iterable[str],
    original_tag: str = "original",
    metrics_subset: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    For each pocket and model, report mean ± std of each metric across invariant
    perturbations, and delta vs the original row.
    """
    inv = set(invariant_tags)
    metrics = metrics_subset or _metric_columns(df)
    rows: list[dict[str, Any]] = []
    df = df.copy()
    tag_col = "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"
    df["_ptag"] = df[tag_col].astype(str)

    for (pid, model), grp in df.groupby(["pocket_id", "model_name"], sort=False):
        base = grp[grp["_ptag"] == original_tag]
        invr = grp[grp["_ptag"].isin(inv)]
        if base.empty or invr.empty:
            continue
        base = base.iloc[0]
        rec: dict[str, Any] = {"pocket_id": pid, "model_name": model}
        for m in metrics:
            if m not in invr.columns:
                continue
            vals = invr[m].astype(float).values
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            rec[f"{m}_original"] = float(base[m]) if m in base.index and pd.notna(base[m]) else float("nan")
            rec[f"{m}_inv_mean"] = float(np.mean(vals))
            rec[f"{m}_inv_std"] = float(np.std(vals))
            bo = rec[f"{m}_original"]
            if not np.isnan(bo):
                rec[f"{m}_mean_delta"] = rec[f"{m}_inv_mean"] - bo
        rows.append(rec)
    return pd.DataFrame(rows)
