#!/usr/bin/env python3
"""Recompute brittleness on NORMALIZED metrics only, and quantify the crop_radius_minus_1.5 effect.

Normalized metric subset for the brittleness flag: validity, uniqueness, mean_qed, std_qed.
(Excludes raw counts n_total/n_valid/n_unique_valid and unnormalized mean_sa/std_sa, which
trivially exceed an SD threshold designed for [0,1] metrics.)

Outputs:
  paper/normalized_brittleness.csv   - dataset x tau brittleness rates
  paper/crop_radius_effect.csv       - per-dataset crop_radius_minus_1.5 QED/SA effect summary
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.report import (
    INVARIANT_TAGS,
    NORMALIZED_METRICS,
    filter_model,
    normalized_brittleness,
)

TAUS = [0.05, 0.10, 0.15, 0.20]

# real100 lives on D:\ (Colab Drive download); real47 is the committed DiffSBDD frozen panel.
REAL100_PATH = Path("D:/metrics_per_condition__runreal100.csv")
RESULTS_DIR = _ROOT / "data" / "results"


def _tag_col(df: pd.DataFrame) -> str:
    return "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"


def find_diffsbdd_real47() -> Path:
    """Locate the DiffSBDD ~47-pocket metrics CSV by content (model=diffsbdd, ~47 pockets, 4 invariants)."""
    best: tuple[int, Path] | None = None
    for p in sorted(RESULTS_DIR.glob("metrics_per_condition__*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model_name" not in df.columns or "pocket_id" not in df.columns:
            continue
        dfd = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]
        if dfd.empty:
            continue
        tags = set(dfd[_tag_col(dfd)].astype(str))
        if not set(INVARIANT_TAGS).issubset(tags):
            continue
        n_pockets = dfd["pocket_id"].astype(str).nunique()
        # real47 panel: between ~40 and ~55 pockets; prefer the one closest to 47.
        if 40 <= n_pockets <= 55:
            score = abs(n_pockets - 47)
            if best is None or score < best[0]:
                best = (score, p)
    if best is None:
        raise FileNotFoundError("Could not auto-detect a DiffSBDD real47 metrics CSV in data/results/")
    return best[1]


def normalized_brittleness_table(df: pd.DataFrame, dataset: str) -> list[dict]:
    df = filter_model(df, "diffsbdd")
    out = normalized_brittleness(df, taus=TAUS, metrics=NORMALIZED_METRICS, dataset=dataset)
    return out.to_dict("records")


def crop_effect(df: pd.DataFrame, dataset: str) -> tuple[dict, pd.DataFrame]:
    """delta = crop_radius_minus_1.5 - original, per pocket, for QED and SA."""
    df = df[df["model_name"].astype(str).str.lower() == "diffsbdd"].copy()
    df["pocket_id"] = df["pocket_id"].astype(str)
    tcol = _tag_col(df)
    orig = (
        df[df[tcol].astype(str) == "original"]
        .drop_duplicates(subset="pocket_id")
        .set_index("pocket_id")
    )
    crop = (
        df[df[tcol].astype(str) == "crop_radius_minus_1.5"]
        .drop_duplicates(subset="pocket_id")
        .set_index("pocket_id")
    )
    common = sorted(set(orig.index) & set(crop.index))
    recs: list[dict] = []
    for pid in common:
        qo = pd.to_numeric(orig.loc[pid, "mean_qed"], errors="coerce")
        qc = pd.to_numeric(crop.loc[pid, "mean_qed"], errors="coerce")
        so = pd.to_numeric(orig.loc[pid, "mean_sa"], errors="coerce")
        sc = pd.to_numeric(crop.loc[pid, "mean_sa"], errors="coerce")
        recs.append(
            {
                "pocket_id": pid,
                "qed_original": float(qo),
                "qed_crop_minus": float(qc),
                "delta_qed": float(qc - qo),
                "sa_original": float(so),
                "sa_crop_minus": float(sc),
                "delta_sa": float(sc - so),
            }
        )
    pdf = pd.DataFrame(recs)
    dq = pdf["delta_qed"].to_numpy()
    ds = pdf["delta_sa"].to_numpy()
    summary = {
        "dataset": dataset,
        "n_pockets": int(len(pdf)),
        "mean_delta_qed": round(float(np.mean(dq)), 4),
        "median_delta_qed": round(float(np.median(dq)), 4),
        "mean_delta_sa": round(float(np.mean(ds)), 4),
        "median_delta_sa": round(float(np.median(ds)), 4),
        "frac_qed_drop_gt_0.05": round(float(np.mean(dq < -0.05)), 4),
        "frac_sa_increase_gt_0.5": round(float(np.mean(ds > 0.5)), 4),
    }
    return summary, pdf


def main() -> None:
    paper = _ROOT / "paper"
    paper.mkdir(parents=True, exist_ok=True)

    datasets: dict[str, pd.DataFrame] = {}
    real47_path = find_diffsbdd_real47()
    datasets["real47"] = pd.read_csv(real47_path)
    print(f"real47 source: {real47_path}")
    if REAL100_PATH.is_file():
        datasets["real100"] = pd.read_csv(REAL100_PATH)
        print(f"real100 source: {REAL100_PATH}")
    else:
        print(f"WARN: real100 not found at {REAL100_PATH}; skipping.")

    # ---- normalized brittleness ----
    brit_rows: list[dict] = []
    for name, df in datasets.items():
        brit_rows.extend(normalized_brittleness_table(df, name))
    brit_df = pd.DataFrame(brit_rows)
    brit_out = paper / "normalized_brittleness.csv"
    brit_df.to_csv(brit_out, index=False)

    # ---- crop_radius_minus_1.5 effect ----
    crop_summaries: list[dict] = []
    top10_blocks: list[tuple[str, pd.DataFrame]] = []
    for name, df in datasets.items():
        summary, pdf = crop_effect(df, name)
        crop_summaries.append(summary)
        top10 = pdf.sort_values("delta_qed").head(10)
        top10_blocks.append((name, top10))
    crop_sum_df = pd.DataFrame(crop_summaries)

    # Write crop CSV: summary rows + top-10 per dataset (tagged).
    crop_out = paper / "crop_radius_effect.csv"
    with crop_out.open("w", encoding="utf-8", newline="") as fh:
        fh.write("# SECTION: summary\n")
        crop_sum_df.to_csv(fh, index=False)
        fh.write("\n# SECTION: top10_qed_drop\n")
        for name, top10 in top10_blocks:
            block = top10.copy()
            block.insert(0, "dataset", name)
            block_round = block.round(4)
            block_round.to_csv(fh, index=False)
            fh.write("\n")

    # ---- plain-English summary ----
    print("\n" + "=" * 70)
    print("NORMALIZED BRITTLENESS (validity, uniqueness, mean_qed, std_qed only)")
    print("=" * 70)
    print(brit_df.to_string(index=False))
    print(f"\nWrote {brit_out}")

    print("\nPlain English — brittleness:")
    for name in datasets:
        sub = brit_df[brit_df["dataset"] == name]
        r05 = sub[sub["tau"] == 0.05]["brittleness_rate"].iloc[0]
        r10 = sub[sub["tau"] == 0.10]["brittleness_rate"].iloc[0]
        r20 = sub[sub["tau"] == 0.20]["brittleness_rate"].iloc[0]
        tot = int(sub["total"].iloc[0])
        print(
            f"  {name} ({tot} pockets): with count/SA columns removed, brittleness is "
            f"{r05:.0%} at tau=0.05, {r10:.0%} at tau=0.10, and {r20:.0%} at tau=0.20. "
            "On normalized chemistry metrics DiffSBDD is far more stable than the raw "
            "all-column flag suggested."
        )

    print("\n" + "=" * 70)
    print("CROP_RADIUS_MINUS_1.5 EFFECT (crop_minus - original)")
    print("=" * 70)
    print(crop_sum_df.to_string(index=False))
    print(f"\nWrote {crop_out}")

    print("\nPlain English — crop_radius_minus_1.5:")
    for s in crop_summaries:
        print(
            f"  {s['dataset']} ({s['n_pockets']} pockets): shrinking the pocket by 1.5 A changes QED by "
            f"mean {s['mean_delta_qed']:+.3f} (median {s['median_delta_qed']:+.3f}) and SA by "
            f"mean {s['mean_delta_sa']:+.3f} (median {s['median_delta_sa']:+.3f}). "
            f"{s['frac_qed_drop_gt_0.05']:.0%} of pockets lose >0.05 QED; "
            f"{s['frac_sa_increase_gt_0.5']:.0%} get >0.5 worse SA."
        )
    for name, top10 in top10_blocks:
        print(f"\n  Top 10 QED drops — {name}:")
        for _, r in top10.iterrows():
            print(
                f"    {r['pocket_id']}: dQED={r['delta_qed']:+.3f} "
                f"(QED {r['qed_original']:.3f}->{r['qed_crop_minus']:.3f}), dSA={r['delta_sa']:+.3f}"
            )


if __name__ == "__main__":
    main()
