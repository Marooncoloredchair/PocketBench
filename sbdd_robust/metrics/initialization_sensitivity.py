"""Initialization Sensitivity Ratio (ISR).

Mechanism under test (reported to us by a senior collaborator): autoregressive
structure-based generators such as Pocket2Mol place atoms one at a time, so the *first*
atom — seeded from the pocket representation — lacks molecular context and is prone to
mispositioning; a misplaced first atom mispositions the whole molecule. Diffusion-based
generators (e.g. TargetDiff, DiffSBDD) denoise all atoms jointly and have no comparable
"first atom".

ISR operationalizes this as a single, falsifiable number. Perturbations are grouped:

  * **frame-moving** — they change *where the pocket frame / first-atom seed sits*
    without (much) changing local chemistry: directional ``face_peel`` (shifts the
    pocket centroid), ``anchor_offset`` (shifts the sampling center directly), and the
    symmetric ``crop_radius_minus`` boundary dose.
  * **featurization** — label/coordinate noise that leaves the frame intact:
    ``atom_shuffle`` and ``coordinate_jitter``.

ISR = median |Δmetric| over frame-moving conditions ÷ median |Δmetric| over
featurization conditions. The mechanism predicts **ISR ≫ 1** for an autoregressive
model and **ISR ≈ 1** for a translation/scale-robust diffusion model. This module is
self-contained (no dependency on ``report``) so it can be reused anywhere.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

FRAME_TAG_PREFIXES = ("face_peel", "anchor_offset", "crop_radius_minus")
FEATURIZATION_TAGS = ["atom_shuffle", "coordinate_jitter"]


def tag_column(df: pd.DataFrame) -> str:
    return "perturbation_tag" if "perturbation_tag" in df.columns else "perturbation_type"


def is_frame_tag(tag: str) -> bool:
    return any(str(tag).startswith(p) for p in FRAME_TAG_PREFIXES)


def _abs_deltas_for_tags(df: pd.DataFrame, metric: str, tags: Iterable[str]) -> np.ndarray:
    """Pooled |metric(condition) − metric(original)| over all (pocket, tag) pairs."""
    tcol = tag_column(df)
    work = df.copy()
    work["pocket_id"] = work["pocket_id"].astype(str)
    orig = (
        work[work[tcol].astype(str) == "original"]
        .drop_duplicates("pocket_id")
        .set_index("pocket_id")
    )
    out: list[float] = []
    for tag in tags:
        cond = (
            work[work[tcol].astype(str) == str(tag)]
            .drop_duplicates("pocket_id")
            .set_index("pocket_id")
        )
        for pid in sorted(set(orig.index) & set(cond.index)):
            o = pd.to_numeric(orig.loc[pid, metric], errors="coerce")
            c = pd.to_numeric(cond.loc[pid, metric], errors="coerce")
            if pd.isna(o) or pd.isna(c):
                continue
            out.append(abs(float(c) - float(o)))
    return np.asarray(out, dtype=float)


def initialization_sensitivity(
    df: pd.DataFrame,
    metric: str = "mean_qed",
    frame_tags: list[str] | None = None,
    featurization_tags: list[str] | None = None,
    n_boot: int = 10000,
    seed: int = 20260612,
    dataset: str = "dataset",
    model: str | None = None,
) -> dict:
    """Return ISR (median frame |Δ| / median featurization |Δ|) with a bootstrap 95% CI.

    The returned dict is one stackable row (ratio, CI, both component effect sizes, the
    tags actually used), so callers can concatenate rows across models for a contrast.
    """
    tcol = tag_column(df)
    present = set(df[tcol].astype(str))
    if frame_tags is None:
        frame_tags = sorted(t for t in present if is_frame_tag(t))
    else:
        frame_tags = [t for t in frame_tags if t in present]
    feat = [t for t in (featurization_tags or FEATURIZATION_TAGS) if t in present]

    frame_d = _abs_deltas_for_tags(df, metric, frame_tags)
    feat_d = _abs_deltas_for_tags(df, metric, feat)

    def _med(a: np.ndarray) -> float:
        return float(np.median(a)) if a.size else float("nan")

    frame_med, feat_med = _med(frame_d), _med(feat_d)
    isr = frame_med / feat_med if feat_med and feat_med > 0 else float("nan")

    isr_lo = isr_hi = float("nan")
    if n_boot and frame_d.size >= 2 and feat_d.size >= 2:
        rng = np.random.default_rng(seed)
        fi = rng.integers(0, frame_d.size, size=(n_boot, frame_d.size))
        ni = rng.integers(0, feat_d.size, size=(n_boot, feat_d.size))
        fb = np.median(frame_d[fi], axis=1)
        nb = np.median(feat_d[ni], axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(nb > 0, fb / nb, np.nan)
        ratios = ratios[np.isfinite(ratios)]
        if ratios.size:
            isr_lo, isr_hi = (float(v) for v in np.percentile(ratios, [2.5, 97.5]))

    return {
        "dataset": dataset,
        "model": model or "all",
        "metric": metric,
        "ISR": round(isr, 3) if not np.isnan(isr) else np.nan,
        "ISR_ci95_lo": round(isr_lo, 3) if not np.isnan(isr_lo) else np.nan,
        "ISR_ci95_hi": round(isr_hi, 3) if not np.isnan(isr_hi) else np.nan,
        "frame_median_abs_delta": round(frame_med, 5) if not np.isnan(frame_med) else np.nan,
        "featurization_median_abs_delta": round(feat_med, 5) if not np.isnan(feat_med) else np.nan,
        "n_frame_pairs": int(frame_d.size),
        "n_featurization_pairs": int(feat_d.size),
        "frame_tags": ",".join(frame_tags),
        "featurization_tags": ",".join(feat),
    }
