#!/usr/bin/env python3
"""ISR stability vs panel size: large-N reference vs small matched slices.

Uses locally archived DiffSBDD CSVs. The 99-pocket Colab export (run1778615375
companion / paper aggregates) has crop ±1.5 Å but **no face_peel**; matched4 v2
has face_peel on n=4 only.

Outputs:
  paper/isr_sample_size_stability.csv
  paper/isr_sample_size_stability.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sbdd_robust.metrics.initialization_sensitivity import (
    FEATURIZATION_TAGS,
    initialization_sensitivity,
)

_ROOT = Path(__file__).resolve().parents[1]
MATCHED4 = ["1AO7", "1B0R", "1HXC", "1I4F"]
FRAME_CROP = ["crop_radius_minus_1.5"]
FRAME_MATCHED = ["crop_radius_minus_1.5", "face_peel_0.25"]


def _load(path: Path, pockets: list[str] | None = None) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["pocket_id"] = df["pocket_id"].astype(str).str.upper()
    if pockets:
        df = df[df["pocket_id"].isin(pockets)].copy()
    return df


def _subsampling_distribution(
    df: pd.DataFrame,
    *,
    n_pockets: int,
    frame_tags: list[str],
    n_draws: int,
    seed: int,
    label: str,
) -> pd.DataFrame:
    """Bootstrap ISR over random pocket subsets (resample pockets with replacement)."""
    pids = sorted(df["pocket_id"].unique())
    if n_pockets > len(pids):
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for i in range(n_draws):
        pick = rng.choice(pids, size=n_pockets, replace=True)
        sub = df[df["pocket_id"].isin(pick)]
        row = initialization_sensitivity(
            sub,
            metric="mean_qed",
            frame_tags=frame_tags,
            featurization_tags=FEATURIZATION_TAGS,
            n_boot=2000,
            seed=seed + i,
            dataset=f"{label}_subsample_n{n_pockets}",
            model="diffsbdd",
        )
        row["draw"] = i
        row["n_pockets_requested"] = n_pockets
        row["n_pockets_unique"] = len(set(pick))
        rows.append(row)
    return pd.DataFrame(rows)


def _summarize_subsample(dist: pd.DataFrame, label: str) -> dict:
    isr = dist["ISR"].dropna().astype(float)
    return {
        "analysis": label,
        "n_draws": len(isr),
        "ISR_median": round(float(isr.median()), 3) if len(isr) else np.nan,
        "ISR_mean": round(float(isr.mean()), 3) if len(isr) else np.nan,
        "ISR_p2.5": round(float(isr.quantile(0.025)), 3) if len(isr) else np.nan,
        "ISR_p97.5": round(float(isr.quantile(0.975)), 3) if len(isr) else np.nan,
        "ISR_iqr": round(float(isr.quantile(0.75) - isr.quantile(0.25)), 3) if len(isr) else np.nan,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--colab47",
        default=str(_ROOT / "data/results/metrics_per_condition__run1778615375.csv"),
    )
    ap.add_argument(
        "--matched4-v2",
        default=str(_ROOT / "data/results/metrics_per_condition__rundiffsbdd_isr_matched4_v2.csv"),
    )
    ap.add_argument("--out-dir", default=str(_ROOT / "paper"))
    ap.add_argument("--n-draws", type=int, default=2000)
    args = ap.parse_args()

    colab47 = _load(Path(args.colab47))
    matched4 = _load(Path(args.matched4_v2), MATCHED4)

    rows: list[dict] = []

    # Reference panels (single estimate + built-in bootstrap CI)
    specs = [
        ("colab47_full_crop", colab47, FRAME_CROP),
        ("colab47_matched4_slice_crop", _load(Path(args.colab47), MATCHED4), FRAME_CROP),
        ("matched4_v2_crop+peel", matched4, FRAME_MATCHED),
    ]
    for label, sub, frame in specs:
        row = initialization_sensitivity(
            sub,
            metric="mean_qed",
            frame_tags=frame,
            featurization_tags=FEATURIZATION_TAGS,
            n_boot=10000,
            dataset=label,
            model="diffsbdd",
        )
        row["analysis"] = label
        rows.append(row)
        print(
            f"{label:32s}  n_pockets={sub['pocket_id'].nunique():2d}  "
            f"ISR={row['ISR']} [{row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]  "
            f"frame={row['frame_median_abs_delta']}  feat={row['featurization_median_abs_delta']}"
        )

    # Subsample instability: draw n=4 / n=10 / n=20 from 47-pocket crop panel
    subsample_summaries: list[dict] = []
    dist_frames: list[pd.DataFrame] = []
    for n in (4, 10, 20):
        dist = _subsampling_distribution(
            colab47,
            n_pockets=n,
            frame_tags=FRAME_CROP,
            n_draws=args.n_draws,
            seed=20260619 + n,
            label="colab47_crop_subsample",
        )
        if dist.empty:
            continue
        dist_frames.append(dist)
        s = _summarize_subsample(dist, f"colab47_crop_subsample_n{n}")
        subsample_summaries.append(s)
        matched4_point = rows[1]["ISR"] if len(rows) > 1 else np.nan
        pct_below = float((dist["ISR"] < matched4_point).mean()) if matched4_point == matched4_point else np.nan
        print(
            f"subsample n={n:2d} from 47  median ISR={s['ISR_median']}  "
            f"95% spread [{s['ISR_p2.5']}, {s['ISR_p97.5']}]  "
            f"IQR={s['ISR_iqr']}"
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_df = pd.DataFrame(rows)
    ref_path = out_dir / "isr_sample_size_stability.csv"
    ref_df.to_csv(ref_path, index=False)

    sub_path = out_dir / "isr_subsample_draws_colab47.csv"
    if dist_frames:
        pd.concat(dist_frames, ignore_index=True).to_csv(sub_path, index=False)

    lines = [
        "ISR sample-size stability (DiffSBDD, mean QED)",
        "",
        "Large-N reference (47-pocket Colab, crop only):",
        f"  ISR = {rows[0]['ISR']} [{rows[0]['ISR_ci95_lo']}, {rows[0]['ISR_ci95_hi']}]  (n=47 pockets)",
        "",
        "Matched4 slice from same 47-pocket export (crop only, same 4 pockets):",
        f"  ISR = {rows[1]['ISR']} [{rows[1]['ISR_ci95_lo']}, {rows[1]['ISR_ci95_hi']}]",
        "",
        "Matched4 local v2 (crop + face_peel, n=4):",
        f"  ISR = {rows[2]['ISR']} [{rows[2]['ISR_ci95_lo']}, {rows[2]['ISR_ci95_hi']}]",
        "",
        "99-pocket face_peel: NOT in archived CSVs. Produce via:",
        "  configs/diffsbdd_real100_stress.yaml  (run_id diffsbdd_real100_stress, ~100 pockets × 7 tags)",
        "  or a Colab rerun of real100 with face_peel_0.25 added to perturbations.",
        "",
        "Subsample instability (random n-pocket draws from 47-panel, crop ISR):",
    ]
    for s in subsample_summaries:
        lines.append(
            f"  n={s['analysis'].split('_n')[-1]:>2s}: median ISR={s['ISR_median']}  "
            f"95% [{s['ISR_p2.5']}, {s['ISR_p97.5']}]  IQR={s['ISR_iqr']}"
        )
    lines.extend(
        [
            "",
            "Interpretation sketch:",
            "  At n=47 crop-only, ISR is stable (~2.0) with a tight CI.",
            "  At n=4, bootstrap over random 4-pocket subsets spans roughly an order of magnitude",
            "  (see subsample 95% interval above) — matched4 point estimates are not comparable to large-N.",
            "  face_peel adds another tag group; 99-pocket face_peel ISR requires diffsbdd_real100_stress run.",
            "",
            f"Reference rows: {ref_path}",
        ]
    )
    txt_path = out_dir / "isr_sample_size_stability.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n-> {ref_path}\n-> {txt_path}")


if __name__ == "__main__":
    main()
