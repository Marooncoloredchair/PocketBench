"""Merge perturbed pocket coordinates into a full PDB for model inference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
from Bio.PDB import PDBIO, PDBParser

AtomKey = Tuple[str, int, str, str]


def build_coord_map(pocket) -> dict[AtomKey, np.ndarray]:
    m: dict[AtomKey, np.ndarray] = {}
    for i in range(pocket.coords.shape[0]):
        k = atom_key_from_pocket_row(pocket, i)
        m[k] = pocket.coords[i].copy()
    return m


def atom_key_from_pocket_row(pocket, i: int) -> AtomKey:
    return (
        str(pocket.chain_ids[i]).strip(),
        int(pocket.residue_numbers[i]),
        str(pocket.residue_names[i]).strip(),
        str(pocket.atom_names[i]).strip(),
    )


def atom_keys_from_pocket(pocket) -> set[AtomKey]:
    return {atom_key_from_pocket_row(pocket, i) for i in range(pocket.coords.shape[0])}


def ghost_atom_keys(reference, perturbed) -> set[AtomKey]:
    """Atoms present in the reference pocket tensor but removed by a frame perturbation."""
    return atom_keys_from_pocket(reference) - atom_keys_from_pocket(perturbed)


def _pocket2mol_center(pocket) -> np.ndarray:
    if pocket.ligand_centroid is not None:
        return np.asarray(pocket.ligand_centroid, dtype=np.float64)
    return pocket.coords.mean(axis=0)


def count_merged_protein_atoms(
    merged_pdb: Path,
    *,
    center: np.ndarray,
    half: float | None,
) -> dict[str, int]:
    """Count heavy protein atoms in merged PDB (optionally inside the Pocket2Mol box)."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(merged_pdb.stem, str(merged_pdb))
    total = 0
    in_box = 0
    for model in structure:
        for chain in model:
            for res in chain:
                for atom in res.get_atoms():
                    elem = str(atom.element).strip().upper()
                    if elem in ("H", "D"):
                        continue
                    total += 1
                    if half is not None:
                        pos = np.asarray(atom.get_coord(), dtype=np.float64)
                        if float(np.max(np.abs(pos - center))) <= half:
                            in_box += 1
    return {"total_heavy": total, "in_bbox": in_box}


def merge_pocket_into_full_pdb(
    pocket,
    full_pdb: Path,
    *,
    bbox_size: float | None = None,
    ghost_atom_keys: Iterable[AtomKey] | None = None,
) -> Path:
    """
    Write a temporary PDB: same topology as ``full_pdb`` but coordinates updated
    for atoms matching pocket keys.

    When ``ghost_atom_keys`` is set with ``bbox_size``, only those keys inside the
    cubic binding-site box are stripped from the full structure. This removes
    crop / face-peel ghosts without deleting unrelated protein context that
    Pocket2Mol still needs inside the box.

    Legacy behaviour (``bbox_size`` set, ``ghost_atom_keys`` omitted): strip every
    heavy atom inside the box that is not in the perturbed pocket tensor.
    """
    full_pdb = Path(full_pdb)
    cmap = build_coord_map(pocket)
    center = _pocket2mol_center(pocket)
    half = float(bbox_size) / 2.0 if bbox_size is not None else None
    ghosts: set[AtomKey] | None = set(ghost_atom_keys) if ghost_atom_keys is not None else None

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(full_pdb.stem, str(full_pdb))
    for model in structure:
        for chain in model:
            for res in list(chain):
                remove_atoms: list[str] = []
                for atom in res.get_atoms():
                    elem = str(atom.element).strip().upper()
                    if elem in ("H", "D"):
                        continue
                    k = (
                        str(chain.id).strip(),
                        int(res.id[1]),
                        str(res.get_resname()).strip(),
                        str(atom.get_name()).strip(),
                    )
                    if k in cmap:
                        atom.set_coord(cmap[k].astype(float))
                        continue
                    if half is not None:
                        pos = np.asarray(atom.get_coord(), dtype=np.float64)
                        if float(np.max(np.abs(pos - center))) > half:
                            continue
                        if ghosts is not None:
                            if k in ghosts:
                                remove_atoms.append(atom.get_id())
                        else:
                            remove_atoms.append(atom.get_id())
                for aid in remove_atoms:
                    res.detach_child(aid)

    fd, tmp = tempfile.mkstemp(suffix="_merged.pdb", prefix="sbdd_robust_")
    os.close(fd)
    out = Path(tmp)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out))
    return out
