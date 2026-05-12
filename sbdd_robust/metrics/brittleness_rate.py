"""Headline brittleness tally: fraction of pocket–model pairs flagged brittle on invariants."""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd


def brittleness_rate_from_flagged(flagged_df: pd.DataFrame) -> Dict[str, Any]:
    """
    ``brittleness_rate`` = (# distinct (pocket_id, model_name) with any ``brittle_invariant``)
    divided by (total distinct (pocket_id, model_name) in the table).
    """
    if flagged_df.empty or "pocket_id" not in flagged_df.columns:
        return {"brittleness_rate": 0.0, "brittle_pairs": 0, "total_pairs": 0}

    model_col = "model_name" if "model_name" in flagged_df.columns else None
    if model_col is None:
        pairs = flagged_df[["pocket_id"]].drop_duplicates()
        brittle_mask = flagged_df["brittle_invariant"] == True  # noqa: E712
        brittle_pairs = flagged_df.loc[brittle_mask, ["pocket_id"]].drop_duplicates()
    else:
        cols = ["pocket_id", "model_name"]
        pairs = flagged_df[cols].drop_duplicates()
        brittle_mask = flagged_df["brittle_invariant"] == True  # noqa: E712
        brittle_pairs = flagged_df.loc[brittle_mask, cols].drop_duplicates()

    n_total = int(pairs.shape[0])
    n_brittle = int(brittle_pairs.shape[0])
    rate = float(n_brittle / n_total) if n_total else 0.0
    return {
        "brittleness_rate": rate,
        "brittle_pairs": n_brittle,
        "total_pairs": n_total,
    }
