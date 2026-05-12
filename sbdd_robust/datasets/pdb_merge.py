"""Merge perturbed pocket coordinates into a full PDB for model inference."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from Bio.PDB import PDBIO, PDBParser


def build_coord_map(pocket) -> dict[tuple[str, int, str, str], np.ndarray]:
    m: dict[tuple[str, int, str, str], np.ndarray] = {}
    for i in range(pocket.coords.shape[0]):
        k = (
            str(pocket.chain_ids[i]).strip(),
            int(pocket.residue_numbers[i]),
            str(pocket.residue_names[i]).strip(),
            str(pocket.atom_names[i]).strip(),
        )
        m[k] = pocket.coords[i].copy()
    return m


def merge_pocket_into_full_pdb(pocket, full_pdb: Path) -> Path:
    """
    Write a temporary PDB: same topology as ``full_pdb`` but coordinates updated
    for atoms matching pocket keys.
    """
    full_pdb = Path(full_pdb)
    cmap = build_coord_map(pocket)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(full_pdb.stem, str(full_pdb))
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res.get_atoms():
                    k = (
                        str(chain.id).strip(),
                        int(res.id[1]),
                        str(res.get_resname()).strip(),
                        str(atom.get_name()).strip(),
                    )
                    if k in cmap:
                        atom.set_coord(cmap[k].astype(float))

    fd, tmp = tempfile.mkstemp(suffix="_merged.pdb", prefix="sbdd_robust_")
    import os

    os.close(fd)
    out = Path(tmp)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out))
    return out
