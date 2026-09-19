#!/usr/bin/env python3
"""Matched ISR on a fixed pocket panel across DiffSBDD, Pocket2Mol, and TargetDiff.

Filters all metrics CSVs to the same pocket IDs and computes **matched** ISR using
``crop_radius_minus_1.5`` + ``face_peel_0.25`` frame tags (plus the featurization
floor ``atom_shuffle`` / ``coordinate_jitter``).

Example:

    python analysis/matched_isr_3pocket.py \\
        --out paper/matched_isr_3pocket.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from sbdd_robust.metrics.initialization_sensitivity import (
    FEATURIZATION_TAGS,
    initialization_sensitivity,
    tag_column,
)

_ROOT = Path(__file__).resolve().parents[1]
MATCHED_FRAME_TAGS = ["crop_radius_minus_1.5", "face_peel_0.25"]
DEFAULT_POCKETS = ["1B0R", "1HXC", "1I4F"]


def _load(path: Path, model: str | None, pocket_ids: set[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if model and "model_name" in df.columns:
        df = df[df["model_name"].astype(str).str.lower() == model.lower()].copy()
    df["pocket_id"] = df["pocket_id"].astype(str).str.upper()
    return df[df["pocket_id"].isin(pocket_ids)].copy()


def _coverage(df: pd.DataFrame, tags: list[str], metric: str = "mean_qed") -> dict:
    """Per-pocket: does original + tag have finite metric?"""
    tcol = tag_column(df)
    orig = df[df[tcol].astype(str) == "original"].drop_duplicates("pocket_id").set_index("pocket_id")
    out: dict[str, dict[str, bool]] = {}
    for pid in sorted(df["pocket_id"].unique()):
        out[pid] = {}
        o = pd.to_numeric(orig.loc[pid, metric], errors="coerce") if pid in orig.index else float("nan")
        out[pid]["original"] = pd.notna(o)
        for tag in tags:
            sub = df[(df["pocket_id"] == pid) & (df[tcol].astype(str) == tag)]
            if sub.empty:
                out[pid][tag] = False
                continue
            v = pd.to_numeric(sub.iloc[0][metric], errors="coerce")
            out[pid][tag] = pd.notna(v)
    return out


def _summarize_arm(
    label: str,
    path: Path,
    df: pd.DataFrame,
    metric: str,
    pocket_ids: list[str],
) -> dict:
    present = set(df[tag_column(df)].astype(str))
    frame = [t for t in MATCHED_FRAME_TAGS if t in present]
    feat = [t for t in FEATURIZATION_TAGS if t in present]
    cov = _coverage(df, ["original", *frame, *feat], metric=metric)
    n_pockets_full = sum(
        1
        for pid in pocket_ids
        if pid in cov and cov[pid].get("original") and all(cov[pid].get(t, False) for t in frame)
    )
    row = initialization_sensitivity(
        df,
        metric=metric,
        frame_tags=frame or None,
        featurization_tags=feat or None,
        dataset=path.stem,
        model=label,
    )
    row.update(
        {
            "architecture": label,
            "view": "matched",
            "metrics_path": str(path),
            "pocket_ids": ",".join(pocket_ids),
            "n_pockets_panel": len(pocket_ids),
            "n_pockets_with_matched_frame_data": n_pockets_full,
            "frame_tags_requested": ",".join(MATCHED_FRAME_TAGS),
            "frame_tags_used": ",".join(frame),
            "featurization_tags_used": ",".join(feat),
            "coverage_json": json.dumps(cov),
            "notes": _notes(label, frame, n_pockets_full, len(pocket_ids)),
        }
    )
    return row


def _notes(label: str, frame_used: list[str], n_full: int, n_panel: int) -> str:
    parts: list[str] = []
    if set(frame_used) != set(MATCHED_FRAME_TAGS):
        missing = [t for t in MATCHED_FRAME_TAGS if t not in frame_used]
        if missing:
            parts.append(f"missing frame tags: {', '.join(missing)}")
    if n_full < n_panel:
        parts.append(f"only {n_full}/{n_panel} pockets have both frame tags + original QED")
    if not parts:
        parts.append("full matched panel")
    return "; ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="3-pocket matched ISR across SBDD models.")
    ap.add_argument(
        "--pocket-ids",
        nargs="+",
        default=DEFAULT_POCKETS,
        help="Pocket panel (default: 1B0R 1HXC 1I4F).",
    )
    ap.add_argument(
        "--diffsbdd-metrics",
        default=str(_ROOT / "data/results/metrics_per_condition__run1778615375.csv"),
        help="DiffSBDD per-condition CSV (real47 panel; may lack face_peel).",
    )
    ap.add_argument(
        "--pocket2mol-metrics",
        default=str(_ROOT / "data/results/metrics_per_condition__runpocket2mol_isr_smoke5_anchor510.csv"),
    )
    ap.add_argument(
        "--targetdiff-metrics",
        default=str(_ROOT / "data/results/metrics_per_condition__runtargetdiff_isr_smoke5.csv"),
    )
    ap.add_argument(
        "--diffsbdd-stress-reference",
        default=None,
        help="Optional full-panel DiffSBDD stress CSV (e.g. Colab real100_stress) for a reference row.",
    )
    ap.add_argument("--metric", default="mean_qed")
    ap.add_argument("--out", default=str(_ROOT / "paper/matched_isr_3pocket.csv"))
    args = ap.parse_args()

    pocket_ids = [str(x).upper() for x in args.pocket_ids]
    pocket_set = set(pocket_ids)

    arms = [
        ("diffsbdd", Path(args.diffsbdd_metrics), "diffsbdd"),
        ("pocket2mol", Path(args.pocket2mol_metrics), "pocket2mol"),
        ("targetdiff", Path(args.targetdiff_metrics), "targetdiff"),
    ]

    rows: list[dict] = []
    print("Matched ISR — panel:", ", ".join(pocket_ids))
    print("Frame tags:", ", ".join(MATCHED_FRAME_TAGS))
    print()

    for label, path, model in arms:
        if not path.is_file():
            print(f"WARN: missing {path} — skip {label}")
            continue
        df = _load(path, model, pocket_set)
        if df.empty:
            print(f"WARN: {label} empty after pocket filter — skip")
            continue
        row = _summarize_arm(label, path, df, args.metric, pocket_ids)
        rows.append(row)
        print(
            f"{label:12s}  ISR={row['ISR']:>8} [{row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]  "
            f"frame={row['frame_median_abs_delta']:.5f}  feat={row['featurization_median_abs_delta']:.5f}  "
            f"pockets w/ frame data={row['n_pockets_with_matched_frame_data']}/{row['n_pockets_panel']}"
        )
        if row["frame_tags_used"] != ",".join(MATCHED_FRAME_TAGS):
            print(f"             NOTE: frame tags used = {row['frame_tags_used']}")

    if args.diffsbdd_stress_reference:
        ref_path = Path(args.diffsbdd_stress_reference)
        if ref_path.is_file():
            ref_df = _load(ref_path, "diffsbdd", pocket_set)
            if not ref_df.empty:
                row = _summarize_arm("diffsbdd_stress_ref", ref_path, ref_df, args.metric, pocket_ids)
                row["architecture"] = "diffsbdd (stress ref)"
                rows.append(row)
                print()
                print(
                    f"diffsbdd ref  ISR={row['ISR']:>8} [{row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]  "
                    f"(from {ref_path.name})"
                )

    if not rows:
        raise SystemExit("No model arms produced ISR rows.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
