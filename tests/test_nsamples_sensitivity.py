from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.nsamples_sensitivity import run_sensitivity


def test_run_sensitivity_runs_on_minimal_tables(tmp_path: Path):
    cols = [
        "pocket_id",
        "perturbation_tag",
        "model_name",
        "n_valid",
        "validity",
    ]
    rows = []
    for pid, base_v in ("P1", 0.9), ("P2", 0.8):
        rows.append([pid, "original", "m", 5, base_v])
        rows.append([pid, "atom_shuffle", "m", 5, base_v + 0.2])
        rows.append([pid, "coordinate_jitter", "m", 5, base_v + 0.15])
        rows.append([pid, "crop_radius_plus_1.5", "m", 5, base_v + 0.12])
        rows.append([pid, "crop_radius_minus_1.5", "m", 5, base_v + 0.11])
    df = pd.DataFrame(rows, columns=cols)
    p20 = tmp_path / "n20.csv"
    p50 = tmp_path / "n50.csv"
    df.to_csv(p20, index=False)
    df.to_csv(p50, index=False)
    inv = [
        "atom_shuffle",
        "coordinate_jitter",
        "crop_radius_plus_1.5",
        "crop_radius_minus_1.5",
    ]
    out = run_sensitivity(
        n20_path=p20,
        n50_path=p50,
        pocket_ids=["P1", "P2"],
        invariant_tags=inv,
        require_covered=True,
        taus=(0.05, 0.10),
    )
    assert len(out) == 4
    assert set(out["run_label"]) == {"n20", "n50"}
