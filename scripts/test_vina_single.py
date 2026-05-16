#!/usr/bin/env python3
"""Step-through docking debug: one SDF + one full PDB (with ligand anchor), to localize failures."""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import numpy as np

from rdkit import Chem

from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.metrics import docking as dock


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--sdf",
        type=Path,
        default=_REPO
        / "data/generations/run_diffsbdd_meaningful_real20/2GJ6/original/2GJ6_gen.sdf",
    )
    ap.add_argument(
        "--pdb",
        type=Path,
        default=_REPO / "data/raw/real50/2gj6.pdb",
    )
    ap.add_argument("--ligand-chain", default="C")
    ap.add_argument("--ligand-resseq", type=int, default=10)
    ap.add_argument("--radius", type=float, default=8.0)
    ap.add_argument("--mol-index", type=int, default=0, help="Which molecule in the SDF (0-based).")
    args = ap.parse_args()

    print("=== Environment ===")
    vina_env = os.environ.get("VINA_EXE", "")
    print(f"VINA_EXE: {vina_env!r}" if vina_env else "VINA_EXE: (not set)")
    print(f"shutil.which('vina'): {shutil.which('vina')!r}")
    print(f"shutil.which('vina.exe'): {shutil.which('vina.exe')!r}")
    resolved = dock._resolve_vina_executable()
    print(f"dock._resolve_vina_executable(): {resolved!r}")
    print()

    sdf = args.sdf.resolve()
    pdb = args.pdb.resolve()
    print(f"SDF: {sdf} exists={sdf.is_file()}")
    print(f"PDB: {pdb} exists={pdb.is_file()}")
    print()

    print("=== Load pocket (same path as CLI) ===")
    pocket = load_pocket_from_complex(
        pdb_path=pdb,
        sdf_path=None,
        radius=float(args.radius),
        pocket_id="debug",
        metadata={"full_pdb": str(pdb)},
        ligand_chain=str(args.ligand_chain),
        ligand_resseq=int(args.ligand_resseq),
    )
    lc = pocket.ligand_centroid
    print(f"ligand_centroid: {lc}")
    print(f"pocket N_atoms: {pocket.coords.shape[0]}")
    print()

    suppl = Chem.SDMolSupplier(str(sdf), sanitize=False, removeHs=False)
    mols: list[Chem.Mol | None] = []
    for m in suppl:
        mols.append(m)
    n = len(mols)
    print(f"SDF molecules count: {n}")
    idx = int(args.mol_index)
    if idx < 0 or idx >= n:
        print(f"ERROR: mol-index {idx} out of range [0, {n - 1}]")
        return 1
    mol = mols[idx]
    if mol is None:
        print(f"ERROR: molecule {idx} is None in supplier")
        return 1
    mol_h = Chem.AddHs(Chem.Mol(mol))
    print(f"Using mol_index={idx} atoms={mol_h.GetNumAtoms()} (with H)")
    print()

    print("=== Step 1: write receptor PDB from pocket ===")
    with tempfile.TemporaryDirectory(prefix="sbdd_vina_debug_") as tmp:
        tdir = Path(tmp)
        rec_pdb = tdir / "receptor.pdb"
        rec_pdbqt = tdir / "receptor.pdbqt"
        dock._write_pocket_pdb(pocket, rec_pdb)
        print(f"  wrote {rec_pdb} ({rec_pdb.stat().st_size} bytes)")

        print("=== Step 2: receptor PDB -> PDBQT ===")
        ok, reason = dock._prepare_receptor_pdbqt(rec_pdb, rec_pdbqt)
        print(f"  ok={ok}")
        if not ok:
            print(f"  REASON: {reason}")
            return 2
        print(f"  wrote {rec_pdbqt} ({rec_pdbqt.stat().st_size} bytes)")

        print("=== Step 3: embed ligand ===")
        emb, er = dock._embed_mol(mol_h)
        if emb is None:
            print(f"  FAILED: {er}")
            return 3
        print(f"  ok ({emb.GetNumAtoms()} atoms)")
        print()

        print("=== Step 4: ligand -> PDBQT ===")
        lig_pdbqt = tdir / "ligand.pdbqt"
        ok_l, lr = dock._mol_to_pdbqt(emb, lig_pdbqt)
        print(f"  ok={ok_l}")
        if not ok_l:
            print(f"  REASON: {lr}")
            return 4
        print(f"  wrote {lig_pdbqt} ({lig_pdbqt.stat().st_size} bytes)")
        print()

        print("=== Step 5: Vina score ===")
        assert lc is not None
        center = np.asarray(lc, dtype=float).reshape(3)
        score, sr = dock._score_one_vina_binding(
            rec_pdbqt, lig_pdbqt, center, n_cpus=1
        )
        print(f"  score={score!r}")
        print(f"  detail: {sr}" if sr else "  detail: (ok)")
        if score is None:
            return 5

        print()
        print("=== Full score_molecules(verbose=True) on one mol ===")
        scores = dock.score_molecules([emb], pocket, n_cpus=1, verbose=True)
        print(f"scores: {scores}")

    print()
    print("Done (all steps succeeded).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
