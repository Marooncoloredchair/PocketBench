"""Merge per-run outputs into tidy tables under ``data/results/``."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def load_metrics_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def save_merged(df: pd.DataFrame, out_dir: Path, name: str = "metrics_merged.csv") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / name
    df.to_csv(p, index=False)
    return p


def append_metrics_row(row: dict, csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row])
    if csv_path.is_file():
        old = pd.read_csv(csv_path)
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(csv_path, index=False)


def merge_existing_results(results_dir: Path, pattern: str = "*.csv") -> Optional[pd.DataFrame]:
    paths = sorted(results_dir.glob(pattern))
    if not paths:
        return None
    dfs = [pd.read_csv(p) for p in paths if p.name != "metrics_merged.csv"]
    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)
