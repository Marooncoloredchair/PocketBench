"""PDB serialization and RDKit bond inference for pocket objects."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sbdd_robust.datasets.pocket import Pocket


def pocket_to_pdb_block(pocket: Pocket, title: str = "pocket") -> str:
    """Write ATOM records (heavy atoms as given)."""
    lines = [f"HEADER    {title[:40]:<40}"]
    serial = 1
    for i in range(pocket.coords.shape[0]):
        x, y, z = pocket.coords[i]
        elem = str(pocket.elements[i]).upper()
        elem = elem[:2] if len(elem) == 2 and elem[1].islower() else elem[0]
        atom_name = str(pocket.atom_names[i])[:4].ljust(4)
        resname = str(pocket.residue_names[i])[:3].ljust(3)
        chain = str(pocket.chain_ids[i])[0]
        resseq = int(pocket.residue_numbers[i]) % 10000
        lines.append(
            f"ATOM  {serial:5d} {atom_name} {resname} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )
        serial += 1
    lines.append("END")
    return "\n".join(lines) + "\n"


def write_pocket_pdb(pocket: Pocket, path: Path, title: Optional[str] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pocket_to_pdb_block(pocket, title or pocket.pocket_id))


def infer_bonds_from_pdb_block(block: str) -> Optional[np.ndarray]:
    """Return (B,2) bond indices using RDKit PDB parsing, or None if parsing fails."""
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
    except Exception:
        pass
    bonds: list[tuple[int, int]] = []
    for b in mol.GetBonds():
        bonds.append((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
    if not bonds:
        return np.zeros((0, 2), dtype=np.int64)
    return np.array(sorted((min(i, j), max(i, j)) for i, j in bonds), dtype=np.int64)


def attach_bonds_rdkit(pocket: Pocket) -> Pocket:
    """Copy pocket with bonds filled from RDKit PDB interpretation."""
    block = pocket_to_pdb_block(pocket)
    bonds = infer_bonds_from_pdb_block(block)
    p = pocket.copy()
    p.bonds = bonds
    return p


def pocket_to_rdkit_mol(pocket: Pocket) -> Optional[Chem.Mol]:
    block = pocket_to_pdb_block(pocket)
    mol = Chem.MolFromPDBBlock(block, sanitize=False, removeHs=False)
    if mol is None:
        return None
    try:
        AllChem.SanitizeMol(mol)
    except Exception:
        pass
    return mol
