#!/usr/bin/env python3
"""Compare ISR on a minimal stress panel (Pocket2Mol vs DiffSBDD, optional).

Designed for cheap mechanism sniff tests (3-5 pockets, 7 conditions) when a full
real100 stress sweep is too expensive. Prints:

  * auto ISR (all frame tags present)
  * matched ISR (crop + face_peel only — same tags on both architectures)
  * per-tag median |delta mean_qed| vs original (shows anchor_offset in isolation)

Example (after a local Pocket2Mol smoke run):

    python analysis/isr_minimal_compare.py \\
        --pocket2mol-metrics data/results/metrics_per_condition__runpocket2mol_isr_smoke5.csv

Optional DiffSBDD arm (same pocket IDs, filtered from a larger CSV):

    python analysis/isr_minimal_compare.py \\
        --pocket2mol-metrics data/results/metrics_per_condition__runpocket2mol_isr_smoke5.csv \\
        --diffsbdd-metrics /path/to/metrics_per_condition__runreal100_stress.csv \\
        --pocket-ids 1AO7 1B0R 1DIZ 1HXC 1I4F
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from sbdd_robust.metrics.initialization_sensitivity import (
    FEATURIZATION_TAGS,
    initialization_sensitivity,
    tag_column,
)

MATCHED_FRAME_TAGS = [
    "crop_radius_minus_1.5",
    "face_peel_0.25",
]


def _load(path: str, model: str | None, pocket_ids: set[str] | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model and "model_name" in df.columns:
        df = df[df["model_name"].astype(str).str.lower() == model.lower()].copy()
    if pocket_ids is not None:
        df = df[df["pocket_id"].astype(str).str.upper().isin(pocket_ids)].copy()
    return df


def _median_abs_delta(df: pd.DataFrame, tag: str, metric: str = "mean_qed") -> float:
    tcol = tag_column(df)
    work = df.copy()
    work["pocket_id"] = work["pocket_id"].astype(str)
    orig = (
        work[work[tcol].astype(str) == "original"]
        .drop_duplicates("pocket_id")
        .set_index("pocket_id")
    )
    cond = (
        work[work[tcol].astype(str) == tag]
        .drop_duplicates("pocket_id")
        .set_index("pocket_id")
    )
    common = sorted(set(orig.index) & set(cond.index))
    deltas: list[float] = []
    for pid in common:
        o = pd.to_numeric(orig.loc[pid, metric], errors="coerce")
        c = pd.to_numeric(cond.loc[pid, metric], errors="coerce")
        if pd.notna(o) and pd.notna(c):
            deltas.append(abs(float(c) - float(o)))
    return float(pd.Series(deltas).median()) if deltas else float("nan")


def _summarize(label: str, df: pd.DataFrame, metric: str, dataset: str) -> dict:
    present = set(df[tag_column(df)].astype(str))
    matched = [t for t in MATCHED_FRAME_TAGS if t in present]
    auto = initialization_sensitivity(df, metric=metric, dataset=dataset, model=label)
    matched_row = initialization_sensitivity(
        df,
        metric=metric,
        frame_tags=matched or None,
        featurization_tags=[t for t in FEATURIZATION_TAGS if t in present],
        dataset=dataset,
        model=label,
    )
    auto["view"] = "auto"
    matched_row["view"] = "matched"
    return {"auto": auto, "matched": matched_row, "df": df}


def main() -> None:
    ap = argparse.ArgumentParser(description="ISR comparison for minimal stress panels.")
    ap.add_argument("--pocket2mol-metrics", required=True, help="Pocket2Mol per-condition CSV.")
    ap.add_argument("--diffsbdd-metrics", default=None, help="Optional DiffSBDD CSV (filtered to --pocket-ids).")
    ap.add_argument(
        "--pocket-ids",
        nargs="+",
        default=None,
        help="Restrict both CSVs to these pocket IDs (recommended for cross-model compare).",
    )
    ap.add_argument("--metric", default="mean_qed")
    ap.add_argument("--out", default=None, help="Optional output CSV for stacked ISR rows.")
    args = ap.parse_args()

    pocket_ids = {str(x).upper() for x in args.pocket_ids} if args.pocket_ids else None

    arms: list[tuple[str, pd.DataFrame]] = []
    p2m = _load(args.pocket2mol_metrics, "pocket2mol", pocket_ids)
    if p2m.empty:
        raise SystemExit("Pocket2Mol metrics empty after filtering.")
    arms.append(("pocket2mol", p2m))

    if args.diffsbdd_metrics:
        dsb = _load(args.diffsbdd_metrics, "diffsbdd", pocket_ids)
        if dsb.empty:
            print("WARN: DiffSBDD metrics empty after pocket filter — skipping DiffSBDD arm.")
        else:
            arms.append(("diffsbdd", dsb))

    rows: list[dict] = []
    print()
    print("=" * 72)
    print("ISR minimal panel comparison")
    print("=" * 72)
    for label, df in arms:
        n_pockets = df["pocket_id"].astype(str).nunique()
        n_rows = len(df)
        print(f"\n--- {label} ({n_pockets} pockets, {n_rows} condition rows) ---")
        summary = _summarize(label, df, args.metric, dataset=Path(args.pocket2mol_metrics).stem)
        for view in ("auto", "matched"):
            r = summary[view]
            print(
                f"  {view:7s} ISR={r['ISR']} [{r['ISR_ci95_lo']}, {r['ISR_ci95_hi']}]  "
                f"frame={r['frame_median_abs_delta']}  feat={r['featurization_median_abs_delta']}"
            )
            rows.append({**r, "architecture": label})

        print("  per-tag median |delta mean_qed| vs original:")
        tags = sorted(set(df[tag_column(df)].astype(str)) - {"original"})
        for tag in tags:
            med = _median_abs_delta(df, tag, args.metric)
            kind = "frame" if any(tag.startswith(p) for p in ("face_peel", "anchor_offset", "crop_radius_minus")) else "feat"
            print(f"    {tag:28s}  {med:.5f}  ({kind})")

    if len(arms) == 2:
        p2m_auto = next(r for r in rows if r["architecture"] == "pocket2mol" and r["view"] == "auto")
        dsb_auto = next(r for r in rows if r["architecture"] == "diffsbdd" and r["view"] == "auto")
        print()
        print("--- mechanism readout ---")
        print(
            f"  Pocket2Mol auto ISR / DiffSBDD auto ISR = "
            f"{p2m_auto['ISR']} / {dsb_auto['ISR']} "
            f"(ratio ~ {float(p2m_auto['ISR']) / float(dsb_auto['ISR']):.2f}x"
            f" if both finite)"
        )
        print("  Mechanism holds if Pocket2Mol ISR clearly exceeds DiffSBDD (~1.95 on your full run)")
        print("  AND anchor_offset median |delta| >> featurization floor on Pocket2Mol.")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
