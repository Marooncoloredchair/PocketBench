"""Find protein residue nearest a reference ligand in a PDB (for meaningful mutation sites)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, is_aa

from sbdd_robust.datasets.pocket_extraction import ligand_centroid_from_pdb_residue


def _residue_representative_coord(res) -> np.ndarray | None:
    ca = None
    heavies: list[np.ndarray] = []
    for atom in res.get_atoms():
        el = (atom.element or "").upper()
        if el in ("H", "D"):
            continue
        c = np.array(atom.coord, dtype=np.float64)
        heavies.append(c)
        if str(atom.get_name()).strip().upper() == "CA":
            ca = c
    if ca is not None:
        return ca
    if heavies:
        return np.stack(heavies, axis=0).mean(axis=0)
    return None


def find_nearest_residue_id(pdb_path: Path, ligand_chain: str, ligand_resseq: int) -> str:
    """Return ``chain:resseq`` for the closest amino-acid residue to the ligand centroid."""
    pdb_path = Path(pdb_path)
    lig_c = ligand_centroid_from_pdb_residue(pdb_path, ligand_chain, int(ligand_resseq))
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    lc = ligand_chain.strip()
    lr = int(ligand_resseq)
    best_d = float("inf")
    best: str | None = None
    for model in structure:
        for chain in model:
            cid = str(chain.id).strip()
            for res in chain:
                if not is_aa(res, standard=False):
                    continue
                if cid == lc and int(res.id[1]) == lr:
                    continue
                pos = _residue_representative_coord(res)
                if pos is None:
                    continue
                d = float(np.linalg.norm(pos - lig_c))
                if d < best_d:
                    best_d = d
                    best = f"{cid}:{int(res.id[1])}"
    if best is None:
        raise RuntimeError(f"No protein residues to compare in {pdb_path}")
    return best
