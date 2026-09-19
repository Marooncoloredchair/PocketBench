"""EXPLORATORY PILOT (n<=10 pockets) — NOT A FINAL CONCLUSION.

Question: on the locally available DiffSBDD real100 SDF slice, does the per-pocket
crop-induced change in QED track the change in molecular weight?

  ΔQED   = mean_qed(crop_radius_minus_1.5) - mean_qed(original)
  ΔMW    = mean_mw(crop_radius_minus_1.5)  - mean_mw(original)

We read per-molecule QED + MW directly from the SDFs (the substrate the aggregated
CSV discarded), aggregate per condition, then regress ΔQED on ΔMW across pockets.

Caveats (why this is a pilot, not a result):
  * n is at most the 10 locally-present pockets, not the full 99-pocket panel.
  * Molecule sets are the model's raw outputs; no docking/relaxation filtering here.
  * Reported to decide whether fetching the full off-machine SDF set is worthwhile.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, QED

RUN_DIR = Path("data/generations/run_diffsbdd_real100")
CONDS = ("original", "crop_radius_minus_1.5", "crop_radius_plus_1.5")


def _mol_stats(sdf: Path) -> tuple[float, float, int]:
    if not sdf.exists():
        return float("nan"), float("nan"), 0
    qeds, mws = [], []
    suppl = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
    for m in suppl:
        if m is None:
            continue
        try:
            Chem.SanitizeMol(m)
        except Exception:
            continue
        try:
            qeds.append(float(QED.qed(m)))
            mws.append(float(Descriptors.MolWt(m)))
        except Exception:
            continue
    n = len(qeds)
    return (float(np.mean(qeds)) if n else float("nan"),
            float(np.mean(mws)) if n else float("nan"), n)


def main() -> None:
    rows = []
    for pk in sorted(p for p in RUN_DIR.iterdir() if p.is_dir()):
        pid = pk.name
        rec = {"pocket_id": pid}
        ok = True
        for c in CONDS:
            q, mw, n = _mol_stats(pk / c / f"{pid}_gen.sdf")
            rec[f"qed_{c}"] = q
            rec[f"mw_{c}"] = mw
            rec[f"n_{c}"] = n
            if c in ("original", "crop_radius_minus_1.5") and (np.isnan(q) or n == 0):
                ok = False
        if ok:
            rows.append(rec)

    df = pd.DataFrame(rows)
    df["dQED_minus"] = df["qed_crop_radius_minus_1.5"] - df["qed_original"]
    df["dMW_minus"] = df["mw_crop_radius_minus_1.5"] - df["mw_original"]

    print("=== EXPLORATORY PILOT: per-pocket crop(-1.5A) deltas (NOT FINAL) ===")
    cols = ["pocket_id", "n_original", "n_crop_radius_minus_1.5",
            "qed_original", "qed_crop_radius_minus_1.5", "dQED_minus",
            "mw_original", "mw_crop_radius_minus_1.5", "dMW_minus"]
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(df[cols].round(4).to_string(index=False))

    sub = df.dropna(subset=["dQED_minus", "dMW_minus"])
    n = len(sub)
    print(f"\nn_pockets_used = {n}")
    if n >= 3:
        x = sub["dMW_minus"].to_numpy(float)
        y = sub["dQED_minus"].to_numpy(float)
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else float("nan")
        slope, intercept = (np.polyfit(x, y, 1) if np.std(x) > 0 else (float("nan"), float("nan")))
        from scipy import stats as _st
        rho, prho = _st.spearmanr(x, y)
        print(f"Pearson r(dMW, dQED)   = {r:.4f}")
        print(f"Spearman rho           = {float(rho):.4f}  (p={float(prho):.4f})")
        print(f"OLS slope dQED/dMW     = {slope:.6f} QED per Da; intercept {intercept:.4f}")
        print(f"median dQED_minus      = {float(np.median(y)):+.4f}")
        print(f"median dMW_minus       = {float(np.median(x)):+.4f}")
        print(f"frac dQED<0            = {float(np.mean(y < 0)):.2f}")
    else:
        print("Too few pockets for a correlation; report raw deltas only.")

    out = Path("paper/pilot_dqed_vs_dmw_n10_EXPLORATORY.csv")
    df.to_csv(out, index=False)
    print(f"\nwrote {out}  (EXPLORATORY PILOT — do not cite as a final result)")


if __name__ == "__main__":
    main()
