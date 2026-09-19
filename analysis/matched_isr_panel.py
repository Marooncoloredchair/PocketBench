#!/usr/bin/env python3
"""Matched ISR panel for Monday cross-model comparison.

Computes matched ISR (crop_radius_minus_1.5 + face_peel_0.25) on a fixed pocket
panel for DiffSBDD, Pocket2Mol, and TargetDiff, plus sanity variants for TargetDiff.

Outputs:
  paper/matched_isr_3pocket.csv   (legacy 3-pocket panel)
  paper/matched_isr_4pocket.csv   (1AO7, 1B0R, 1HXC, 1I4F)
  paper/matched_isr_sanity.csv    (TargetDiff ISR sensitivity checks)
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
MATCHED_FRAME = ["crop_radius_minus_1.5", "face_peel_0.25"]
PANEL_3 = ["1B0R", "1HXC", "1I4F"]
PANEL_4 = ["1AO7", "1B0R", "1HXC", "1I4F"]
PANEL_10 = PANEL_4 + ["1JF1", "1YFW", "1YFY", "2GJ6", "2P5E", "2P5W"]


def _latest_metrics(glob_pat: str) -> Path | None:
    hits = sorted(_ROOT.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0] if hits else None


def _load(path: Path, model: str, pockets: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "model_name" in df.columns:
        df = df[df["model_name"].astype(str).str.lower() == model.lower()]
    df["pocket_id"] = df["pocket_id"].astype(str).str.upper()
    return df[df["pocket_id"].isin(pockets)].copy()


def _coverage(df: pd.DataFrame, pockets: list[str], tags: list[str]) -> dict:
    tcol = tag_column(df)
    out: dict[str, dict[str, bool]] = {}
    for pid in pockets:
        sub = df[df["pocket_id"] == pid]
        out[pid] = {}
        for tag in tags:
            rows = sub[sub[tcol].astype(str) == tag]
            if rows.empty:
                out[pid][tag] = False
                continue
            v = pd.to_numeric(rows.iloc[0]["mean_qed"], errors="coerce")
            out[pid][tag] = pd.notna(v)
    return out


def _isr_row(
    label: str,
    path: Path,
    df: pd.DataFrame,
    pockets: list[str],
    frame_tags: list[str],
    variant: str = "matched",
) -> dict:
    present = set(df[tag_column(df)].astype(str))
    frame = [t for t in frame_tags if t in present]
    feat = [t for t in FEATURIZATION_TAGS if t in present]
    cov = _coverage(df, pockets, ["original", *frame, *feat])
    n_full = sum(
        1
        for pid in pockets
        if cov.get(pid, {}).get("original")
        and all(cov.get(pid, {}).get(t, False) for t in frame)
    )
    row = initialization_sensitivity(
        df,
        metric="mean_qed",
        frame_tags=frame,
        featurization_tags=feat,
        dataset=f"{path.stem}_{variant}",
        model=label,
    )
    row.update(
        {
            "architecture": label,
            "view": variant,
            "metrics_path": str(path),
            "pocket_ids": ",".join(pockets),
            "n_pockets_panel": len(pockets),
            "n_pockets_with_both_frame_tags": n_full,
            "coverage_json": json.dumps(cov),
        }
    )
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--diffsbdd-metrics",
        default=str(
            _latest_metrics("data/results/metrics_per_condition__rundiffsbdd_isr_matched4_v2.csv")
            or _latest_metrics("data/results/metrics_per_condition__rundiffsbdd_isr_matched4_panel.csv")
            or (_ROOT / "data/results/metrics_per_condition__run1778615375.csv")
        ),
    )
    ap.add_argument(
        "--pocket2mol-metrics",
        default=str(
            _latest_metrics("data/results/metrics_per_condition__runpocket2mol_isr_matched4_v3.csv")
            or _latest_metrics("data/results/metrics_per_condition__runpocket2mol_isr_matched4_v2.csv")
            or (_ROOT / "data/results/metrics_per_condition__runpocket2mol_isr_matched4.csv")
        ),
    )
    ap.add_argument(
        "--targetdiff-metrics",
        default=str(
            _latest_metrics("data/results/metrics_per_condition__runtargetdiff_isr_matched4.csv")
            or (_ROOT / "data/results/metrics_per_condition__runtargetdiff_isr_smoke5.csv")
        ),
    )
    ap.add_argument("--out-dir", default=str(_ROOT / "paper"))
    args = ap.parse_args()

    arms = [
        ("diffsbdd", Path(args.diffsbdd_metrics)),
        ("pocket2mol", Path(args.pocket2mol_metrics)),
        ("targetdiff", Path(args.targetdiff_metrics)),
    ]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for panel_name, pockets in [
        ("3pocket", PANEL_3),
        ("4pocket", PANEL_4),
        ("10pocket", PANEL_10),
    ]:
        rows: list[dict] = []
        print(f"\n=== Matched ISR panel: {panel_name} ({', '.join(pockets)}) ===")
        for label, path in arms:
            if not path.is_file():
                print(f"  SKIP {label}: missing {path.name}")
                continue
            df = _load(path, label, pockets)
            if df.empty:
                print(f"  SKIP {label}: no rows after filter")
                continue
            row = _isr_row(label, path, df, pockets, MATCHED_FRAME)
            rows.append(row)
            print(
                f"  {label:12s}  ISR={row['ISR']} [{row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]  "
                f"frame={row['frame_median_abs_delta']}  feat={row['featurization_median_abs_delta']}  "
                f"both frame tags={row['n_pockets_with_both_frame_tags']}/{row['n_pockets_panel']}  "
                f"tags={row['frame_tags']}"
            )
        if rows:
            out = out_dir / f"matched_isr_{panel_name}.csv"
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"  -> {out}")

    # TargetDiff sanity variants on 4-pocket panel
    td_path = Path(args.targetdiff_metrics)
    if td_path.is_file():
        td = _load(td_path, "targetdiff", PANEL_4)
        sanity_specs = [
            ("full", td, MATCHED_FRAME),
            ("crop_only", td, ["crop_radius_minus_1.5"]),
            ("exclude_1I4F", td[td["pocket_id"] != "1I4F"], MATCHED_FRAME),
            ("exclude_1I4F_peel_row", td[~((td["pocket_id"] == "1I4F") & (td[tag_column(td)] == "face_peel_0.25"))], MATCHED_FRAME),
        ]
        print("\n=== TargetDiff ISR sanity (4-pocket panel) ===")
        srows: list[dict] = []
        for name, sub, ftags in sanity_specs:
            if sub.empty:
                continue
            row = _isr_row("targetdiff", td_path, sub, PANEL_4 if name != "exclude_1I4F" else ["1AO7", "1B0R", "1HXC"], ftags, variant=name)
            srows.append(row)
            print(f"  {name:22s}  ISR={row['ISR']} [{row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]  n_frame={row['n_frame_pairs']}")
        if srows:
            out = out_dir / "matched_isr_sanity.csv"
            pd.DataFrame(srows).to_csv(out, index=False)
            print(f"  -> {out}")


if __name__ == "__main__":
    main()
