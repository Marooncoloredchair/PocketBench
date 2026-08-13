#!/usr/bin/env python3
"""Local Vina smoke: DiffSBDD molecules for one pocket × one condition → non-empty scores.

Uses existing DiffSBDD SDFs when present (this machine's diffsbdd env is torch CPU-only).
Always exercises the same score_molecules() path as compute_docking: true in the CLI.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from rdkit import Chem

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.metrics import docking as dock_mod

POCKET_ID = "7HZW"
SDF = (
    _REPO
    / "data"
    / "generations"
    / "run_diffsbdd_real100"
    / POCKET_ID
    / "original"
    / f"{POCKET_ID}_gen.sdf"
)
PDB = _REPO / "data" / "raw" / "real100" / "7hzw.pdb"
N = 2  # keep smoke light on low-RAM machines; Job A still uses n=20


def main() -> int:
    os.environ.setdefault("VINA_EXE", r"D:\obsfu\tools\vina\vina.exe")
    os.environ["SBDD_DOCKING_VERBOSE"] = "1"

    if not SDF.is_file():
        raise SystemExit(f"Missing DiffSBDD SDF for smoke: {SDF}")
    if not Path(os.environ["VINA_EXE"]).is_file():
        raise SystemExit(f"VINA_EXE not found: {os.environ['VINA_EXE']}")

    pocket = load_pocket_from_complex(
        pdb_path=PDB,
        radius=8.0,
        pocket_id=POCKET_ID,
        ligand_chain="C",
        ligand_resseq=301,
        metadata={"full_pdb": str(PDB)},
    )
    print(f"pocket={POCKET_ID} centroid={pocket.ligand_centroid}")

    suppl = Chem.SDMolSupplier(str(SDF), sanitize=False, removeHs=False)
    mols = []
    for m in suppl:
        if m is None:
            continue
        mols.append(m)
        if len(mols) >= N:
            break
    print(f"loaded {len(mols)} molecules from {SDF}")
    if len(mols) < 1:
        raise SystemExit("No molecules in SDF")

    scores = dock_mod.score_molecules(mols, pocket, n_cpus=4, verbose=True)
    finite = [s for s in scores if s is not None and np.isfinite(s)]
    print("---")
    print(f"scores={scores}")
    print(f"n_finite={len(finite)}/{len(scores)}")
    if finite:
        print(f"vina_score_mean={float(np.mean(finite)):.4f}")
        print(f"vina_score_std={float(np.std(finite)):.4f}" if len(finite) > 1 else "vina_score_std=nan")
    if not finite:
        raise SystemExit("FAIL: all Vina scores empty/None — do not submit Job A")
    print("PASS: non-empty Vina scores")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
