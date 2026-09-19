"""Verify WHY a DiffSBDD real47 regeneration is needed before spending cluster hours.

Checks, with evidence:
  1. Does the DiffSBDD run1778615375 panel cover the same 47 pockets as
     Pocket2Mol real47 v2, or are the pocket sets disjoint?
  2. Do the two panels share conditions and n_samples?
  3. Are per-molecule artifacts (SDF / SMILES sidecars) archived on disk for
     each run, i.e. can MW be recovered without regenerating?
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "data" / "results"
GENS = ROOT / "data" / "generations"

DIFF = RESULTS / "metrics_per_condition__run1778615375.csv"
P2M = RESULTS / "metrics_per_condition__runpocket2mol_real47_full_v2.csv"
DIFF_ALT = RESULTS / "metrics_per_condition__rundiffsbdd_real100_vina.csv"


def describe(path: Path, label: str) -> pd.DataFrame | None:
    print(f"\n--- {label} ---")
    print(f"path: {path}")
    if not path.exists():
        print("  MISSING")
        return None
    df = pd.read_csv(path)
    print(f"  rows={len(df)}  cols={len(df.columns)}")
    pockets = sorted(df["pocket_id"].astype(str).unique()) if "pocket_id" in df else []
    print(f"  unique pockets: {len(pockets)}")
    print(f"  first 8 pockets: {pockets[:8]}")
    if "condition" in df:
        print(f"  conditions ({df['condition'].nunique()}): {sorted(df['condition'].unique())}")
    for col in ("n_samples", "n_requested", "n_generated", "n_valid"):
        if col in df.columns:
            print(f"  {col}: min={df[col].min()} max={df[col].max()} median={df[col].median()}")
    return df


def archive_probe(run_token: str) -> None:
    """Look for per-molecule artifacts belonging to a run."""
    print(f"\n--- molecule archive probe: '{run_token}' ---")
    if not GENS.exists():
        print(f"  {GENS} does not exist")
        return
    hits = [p for p in GENS.iterdir() if run_token in p.name]
    if not hits:
        print(f"  no directory under data/generations matching '{run_token}'")
        names = sorted(p.name for p in GENS.iterdir() if p.is_dir())
        print(f"  generations dirs present ({len(names)}): {names[:20]}")
        return
    for h in hits:
        sdf = list(h.rglob("*.sdf"))
        smi = list(h.rglob("*.smi")) + list(h.rglob("*smiles*"))
        csvs = list(h.rglob("*.csv"))
        print(f"  {h.name}: sdf={len(sdf)} smiles={len(smi)} csv={len(csvs)}")


def main() -> None:
    d = describe(DIFF, "DiffSBDD run1778615375 (claimed real47)")
    p = describe(P2M, "Pocket2Mol real47 full v2")
    a = describe(DIFF_ALT, "DiffSBDD real100_vina (used in MW analysis)")

    print("\n=== POCKET SET OVERLAP ===")
    sets = {}
    if d is not None and "pocket_id" in d:
        sets["diffsbdd_run1778615375"] = set(d["pocket_id"].astype(str))
    if p is not None and "pocket_id" in p:
        sets["pocket2mol_real47_v2"] = set(p["pocket_id"].astype(str))
    if a is not None and "pocket_id" in a:
        sets["diffsbdd_real100_vina"] = set(a["pocket_id"].astype(str))

    keys = list(sets)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            x, y = keys[i], keys[j]
            inter = sets[x] & sets[y]
            print(f"  {x}  vs  {y}")
            print(f"    |A|={len(sets[x])} |B|={len(sets[y])} overlap={len(inter)}")
            if 0 < len(inter) <= 10:
                print(f"    shared: {sorted(inter)}")

    archive_probe("1778615375")
    archive_probe("pocket2mol_real47_full_v2")

    print("\n=== MW CSVs already extracted ===")
    for f in sorted(RESULTS.glob("mw_*.csv")):
        try:
            m = pd.read_csv(f)
            npk = m["pocket_id"].nunique() if "pocket_id" in m else "?"
            print(f"  {f.name}: rows={len(m)} pockets={npk} cols={list(m.columns)[:8]}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {f.name}: unreadable ({exc})")


if __name__ == "__main__":
    main()
