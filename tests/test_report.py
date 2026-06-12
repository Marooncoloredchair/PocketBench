"""Regression tests for the shared reliability-report metrics (sbdd_robust.report)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sbdd_robust import report as R


def _synthetic(n_pockets: int = 6, qed_plus: float = 0.55, qed_orig: float = 0.50) -> pd.DataFrame:
    """QED rises with crop radius (shrinking lowers QED); validity/uniqueness flat."""
    rows = []
    for i in range(n_pockets):
        qed_minus = qed_orig - (0.05 + 0.001 * i)  # strictly below original -> negative delta
        conds = {
            "original": qed_orig,
            "crop_radius_minus_1.5": qed_minus,
            "crop_radius_plus_1.5": qed_plus,
            "atom_shuffle": qed_orig,
            "coordinate_jitter": qed_orig + 0.005,
        }
        for tag, qed in conds.items():
            rows.append({
                "pocket_id": f"P{i}", "perturbation_tag": tag, "model_name": "demo",
                "validity": 0.9, "uniqueness": 0.9, "mean_qed": qed, "std_qed": 0.05,
                "mean_sa": 3.0, "std_sa": 0.2,
            })
    return pd.DataFrame(rows)


def test_tag_to_offset():
    assert R.tag_to_offset("original") == 0.0
    assert R.tag_to_offset("crop_radius_plus_1.5") == 1.5
    assert R.tag_to_offset("crop_radius_minus_2") == -2.0
    assert R.tag_to_offset("atom_shuffle") is None


def test_resolve_crop_tag_fallback():
    df = _synthetic()
    assert R.resolve_crop_tag(df, "crop_radius_minus_1.5") == "crop_radius_minus_1.5"
    # absent tag -> closest available crop-minus
    assert R.resolve_crop_tag(df, "crop_radius_minus_3") == "crop_radius_minus_1.5"
    df_no_crop = df[~df["perturbation_tag"].str.startswith("crop_radius_minus")]
    assert R.resolve_crop_tag(df_no_crop, "crop_radius_minus_1.5") is None


def test_pbsi_positive_slope_and_snr():
    pp, summary = R.pocket_boundary_sensitivity(_synthetic())
    assert summary["n_pockets"] == 6
    # shrinking lowers QED for every pocket -> all slopes positive
    assert summary["frac_shrinking_lowers_qed"] == pytest.approx(1.0)
    assert summary["PBSI_median_abs_slope_qed_per_A"] > 0
    # boundary swing dwarfs the tiny featurization-noise std
    assert summary["median_snr_1.5A"] > 1.0
    assert summary["frac_boundary_exceeds_noise"] == pytest.approx(1.0)
    assert (pp["pbsi_slope_qed_per_A"] > 0).all()


def test_crop_paired_wilcoxon_direction_and_significance():
    res = R.crop_paired_wilcoxon(_synthetic(), crop_tag="crop_radius_minus_1.5")
    qed = res[res["metric"] == "dQED"].iloc[0]
    assert qed["n_pairs"] == 6
    assert qed["median_delta"] < 0           # contraction lowers QED
    assert qed["rank_biserial_r"] == pytest.approx(-1.0)  # all pockets move the same way
    assert qed["p_value"] < 0.05             # n=6 all-negative -> two-sided p = 0.03125
    assert qed["median_ci95_hi"] < 0         # bootstrap CI excludes zero


def test_normalized_brittleness_monotone_in_tau():
    df = _synthetic()
    out = R.normalized_brittleness(df, taus=[0.05, 0.10, 0.15, 0.20])
    assert list(out["tau"]) == [0.05, 0.10, 0.15, 0.20]
    rates = out["brittleness_rate"].to_numpy()
    assert np.all((rates >= 0) & (rates <= 1))
    assert np.all(np.diff(rates) <= 1e-9)    # non-increasing in tau


def _isr_synthetic(n_pockets: int = 8, frame_shift: float = 0.20, noise_shift: float = 0.01) -> pd.DataFrame:
    """Frame-moving perturbations move QED a lot; featurization noise barely moves it."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(n_pockets):
        base = 0.55
        conds = {
            "original": base,
            "atom_shuffle": base + noise_shift * rng.normal(),
            "coordinate_jitter": base + noise_shift * rng.normal(),
            "face_peel_0.25": base - frame_shift - 0.01 * i,
            "anchor_offset_2.0": base - frame_shift - 0.01 * i,
            "crop_radius_minus_1.5": base - frame_shift * 0.8,
        }
        for tag, qed in conds.items():
            rows.append({
                "pocket_id": f"P{i}", "perturbation_tag": tag, "model_name": "p2m_like",
                "validity": 0.9, "uniqueness": 0.9, "mean_qed": float(qed), "std_qed": 0.05,
            })
    return pd.DataFrame(rows)


def test_isr_high_for_frame_sensitive_model():
    row = R.initialization_sensitivity(_isr_synthetic(), metric="mean_qed")
    assert row["n_frame_pairs"] > 0 and row["n_featurization_pairs"] > 0
    assert row["frame_median_abs_delta"] > row["featurization_median_abs_delta"]
    assert row["ISR"] > 3.0  # frame effect dwarfs featurization noise
    assert "face_peel_0.25" in row["frame_tags"]
    assert "anchor_offset_2.0" in row["frame_tags"]


def test_isr_near_one_for_robust_model():
    # Deterministic: every perturbation moves QED by the same 0.03, so frame == featurization.
    rows = []
    for i in range(8):
        base = 0.55
        for tag in ("original", "atom_shuffle", "coordinate_jitter",
                    "face_peel_0.25", "anchor_offset_2.0", "crop_radius_minus_1.5"):
            qed = base if tag == "original" else base - 0.03
            rows.append({"pocket_id": f"P{i}", "perturbation_tag": tag, "model_name": "diffusion_like",
                         "validity": 0.9, "uniqueness": 0.9, "mean_qed": qed, "std_qed": 0.05})
    row = R.initialization_sensitivity(pd.DataFrame(rows), metric="mean_qed", n_boot=0)
    assert row["ISR"] == pytest.approx(1.0, abs=1e-6)  # not the >>1 autoregressive signature


def test_normalized_brittleness_flags_high_variance():
    # crop_minus QED far from the rest -> mean_qed std across invariants exceeds 0.10
    df = _synthetic(qed_orig=0.50)
    df.loc[df["perturbation_tag"] == "crop_radius_minus_1.5", "mean_qed"] = 0.10
    out = R.normalized_brittleness(df, taus=[0.10])
    assert out.iloc[0]["brittleness_rate"] == pytest.approx(1.0)

    flat = _synthetic()
    flat["mean_qed"] = 0.5  # no dispersion anywhere
    out_flat = R.normalized_brittleness(flat, taus=[0.10])
    assert out_flat.iloc[0]["brittleness_rate"] == pytest.approx(0.0)
