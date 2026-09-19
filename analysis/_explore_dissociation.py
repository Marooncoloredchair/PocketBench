#!/usr/bin/env python3
"""EXPLORATORY: is DiffSBDD vs Pocket2Mol brittleness a real double dissociation?

Tests the interaction, not two separate significance tests:
  contrast(pocket, model) = |dMETRIC_boundary| - mean|dMETRIC_noise|
  H0: contrast_diffsbdd - contrast_pocket2mol = 0  (paired Wilcoxon, shared pockets)

Also tests the metric-crossed pattern (QED vs SA) that would make any scalar
reliability ranking metric-dependent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.report import filter_model, tag_col

RESULTS = _ROOT / "data" / "results"
P2M = RESULTS / "metrics_per_condition__runpocket2mol_real47_full_v2.csv"
NOISE = ["atom_shuffle", "coordinate_jitter"]
BOUNDARY = "crop_radius_minus_1.5"


def find_diffsbdd_real47() -> Path:
    best = None
    for p in sorted(RESULTS.glob("metrics_per_condition__*.csv")):
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "model_name" not in df.columns or "pocket_id" not in df.columns:
            continue
        d = df[df["model_name"].astype(str).str.lower() == "diffsbdd"]
        if d.empty:
            continue
        tags = set(d[tag_col(d)].astype(str))
        if BOUNDARY not in tags or not set(NOISE).issubset(tags):
            continue
        n = d["pocket_id"].nunique()
        if 40 <= n <= 55:
            score = abs(n - 47)
            if best is None or score < best[0]:
                best = (score, p)
    if best is None:
        raise SystemExit("no DiffSBDD real47 panel found")
    return best[1]


def per_pocket(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-pocket boundary and noise movement magnitudes for one metric."""
    df = df.copy()
    df["pocket_id"] = df["pocket_id"].astype(str)
    tc = tag_col(df)
    df[tc] = df[tc].astype(str)
    wide = (
        df.drop_duplicates(["pocket_id", tc])
        .pivot(index="pocket_id", columns=tc, values=metric)
        .apply(pd.to_numeric, errors="coerce")
    )
    need = ["original", BOUNDARY] + NOISE
    if any(c not in wide.columns for c in need):
        raise SystemExit(f"missing tags for {metric}: have {list(wide.columns)}")
    out = pd.DataFrame(index=wide.index)
    out["signed_boundary"] = wide[BOUNDARY] - wide["original"]
    out["boundary"] = (wide[BOUNDARY] - wide["original"]).abs()
    out["noise"] = pd.concat(
        [(wide[t] - wide["original"]).abs() for t in NOISE], axis=1
    ).mean(axis=1)
    out["contrast"] = out["boundary"] - out["noise"]
    return out.dropna()


def wilcox(x: np.ndarray, label: str) -> dict:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    nz = x[x != 0]
    if nz.size == 0:
        return {"label": label, "n": int(x.size), "median": 0.0, "p": 1.0, "r": np.nan}
    res = stats.wilcoxon(nz, alternative="two-sided")
    ranks = stats.rankdata(np.abs(nz))
    rp, rn = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {
        "label": label,
        "n": int(x.size),
        "median": round(float(np.median(x)), 4),
        "p": float(res.pvalue),
        "r": round(float((rp - rn) / (rp + rn)), 3),
    }


def main() -> None:
    d_path = find_diffsbdd_real47()
    print(f"DiffSBDD panel: {d_path.name}")
    print(f"Pocket2Mol panel: {P2M.name}\n")
    dsb = filter_model(pd.read_csv(d_path), "diffsbdd")
    p2m = filter_model(pd.read_csv(P2M), "pocket2mol")

    for metric in ["mean_qed", "mean_sa"]:
        a = per_pocket(dsb, metric)
        b = per_pocket(p2m, metric)
        shared = sorted(set(a.index) & set(b.index))
        print("=" * 72)
        print(f"METRIC: {metric}   shared pockets with complete rows: n={len(shared)}")
        print("=" * 72)
        if len(shared) < 8:
            print("too few shared pockets; skipping\n")
            continue
        a, b = a.loc[shared], b.loc[shared]

        # within-model signed boundary effect (replicates known per-model results)
        for name, t in [("DiffSBDD", a), ("Pocket2Mol", b)]:
            r = wilcox(t["signed_boundary"].to_numpy(), f"{name} signed d{metric} boundary")
            print(f"  {r['label']:<46} n={r['n']:>3} median={r['median']:>8} p={r['p']:.3g} r={r['r']}")

        # within-model boundary-vs-noise contrast
        for name, t in [("DiffSBDD", a), ("Pocket2Mol", b)]:
            r = wilcox(t["contrast"].to_numpy(), f"{name} (boundary - noise)")
            print(f"  {r['label']:<46} n={r['n']:>3} median={r['median']:>8} p={r['p']:.3g} r={r['r']}")

        # THE INTERACTION: does the boundary-vs-noise contrast differ by model?
        inter = (a["contrast"] - b["contrast"]).to_numpy()
        r = wilcox(inter, "INTERACTION contrast_DiffSBDD - contrast_P2M")
        print(f"\n  >> {r['label']:<43} n={r['n']:>3} median={r['median']:>8} p={r['p']:.3g} r={r['r']}")

        # simple boundary-magnitude difference by model
        r2 = wilcox((a["boundary"] - b["boundary"]).to_numpy(), "boundary_DiffSBDD - boundary_P2M")
        print(f"  >> {r2['label']:<43} n={r2['n']:>3} median={r2['median']:>8} p={r2['p']:.3g} r={r2['r']}")
        r3 = wilcox((a["noise"] - b["noise"]).to_numpy(), "noise_DiffSBDD - noise_P2M")
        print(f"  >> {r3['label']:<43} n={r3['n']:>3} median={r3['median']:>8} p={r3['p']:.3g} r={r3['r']}")
        print()

    # ---- Vina receptor-vs-molecule dominance, bootstrapped ----
    vpath = _ROOT / "paper" / "vina_receptor_control_per_pocket.csv"
    if vpath.is_file():
        v = pd.read_csv(vpath)
        rec = v["dVina_receptor_effect"].abs().to_numpy(dtype=float)
        mol = v["dVina_molecule_effect"].abs().to_numpy(dtype=float)
        ok = ~(np.isnan(rec) | np.isnan(mol))
        rec, mol = rec[ok], mol[ok]
        frac = rec / (rec + mol)
        rng = np.random.default_rng(0)
        idx = rng.integers(0, frac.size, size=(10000, frac.size))
        med_boot = np.median(frac[idx], axis=1)
        lo, hi = np.percentile(med_boot, [2.5, 97.5])
        print("=" * 72)
        print("VINA: receptor artifact share of |dVina| under pocket contraction")
        print("=" * 72)
        print(f"  n pockets                     = {frac.size}")
        print(f"  median |dVina_receptor|       = {np.median(rec):.4f}")
        print(f"  median |dVina_molecule|       = {np.median(mol):.4f}")
        print(f"  receptor/(receptor+molecule)  = {np.median(frac):.4f}  95% CI [{lo:.4f}, {hi:.4f}]")
        print(f"  pockets where receptor > mol  = {int((rec > mol).sum())}/{frac.size} "
              f"({100 * (rec > mol).mean():.1f}%)")
        rr = wilcox(rec - mol, "|dVina_receptor| - |dVina_molecule|")
        print(f"  Wilcoxon rec vs mol: median={rr['median']} p={rr['p']:.3g} r={rr['r']}")


if __name__ == "__main__":
    main()
