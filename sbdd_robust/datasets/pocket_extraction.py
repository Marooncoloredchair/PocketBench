"""Extract binding-pocket atoms within a radius of the ligand."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import numpy as np
from Bio.PDB import is_aa
from rdkit import Chem

from sbdd_robust.datasets.pdb_io import attach_bonds_rdkit
from sbdd_robust.datasets.pocket import Pocket


def _ligand_centroid_from_sdf(sdf_path: Path) -> np.ndarray:
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
    mol = suppl[0]
    if mol is None:
        raise ValueError(f"Could not read ligand SDF: {sdf_path}")
    conf = mol.GetConformer()
    coords = np.array(
        [conf.GetAtomPosition(i) for i in range(mol.GetNumAtoms())],
        dtype=np.float64,
    )
    atomic_nums = np.array([mol.GetAtomWithIdx(i).GetAtomicNum() for i in range(mol.GetNumAtoms())])
    heavy = atomic_nums > 1
    if not np.any(heavy):
        heavy = np.ones_like(heavy, dtype=bool)
    return coords[heavy].mean(axis=0)


def _centroid_from_pdb_ligand_residue(pdb_path: Path, chain_id: str, resseq: int) -> np.ndarray:
    """Fallback: centroid of a HETATM residue (e.g. from merged PDB)."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))
    coords: list[np.ndarray] = []
    chain_id = chain_id.strip()
    for model in structure:
        for chain in model:
            if chain.id.strip() != chain_id:
                continue
            for res in chain:
                if res.id[1] != resseq:
                    continue
                for atom in res.get_atoms():
                    if atom.element and atom.element != "H":
                        coords.append(np.array(atom.coord, dtype=np.float64))
    if not coords:
        raise ValueError(
            f"No heavy atoms for ligand chain={chain_id!r} resseq={resseq} in {pdb_path}"
        )
    return np.stack(coords, axis=0).mean(axis=0)


def extract_pocket(
    pdb_path: Path,
    ligand_sdf_path: Optional[Path] = None,
    radius: float = 8.0,
    pocket_id: Optional[str] = None,
    ligand_chain: Optional[str] = None,
    ligand_resseq: Optional[int] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> Pocket:
    """
    Select protein amino-acid heavy atoms within ``radius`` Å of the ligand centroid.

    Ligand location: ``ligand_sdf_path`` (RDKit centroid), or ``ligand_chain`` +
    ``ligand_resseq`` for a residue in the same PDB.
    """
    pdb_path = Path(pdb_path)
    meta: dict[str, Any] = dict(metadata or {})

    if ligand_sdf_path is not None:
        centroid = _ligand_centroid_from_sdf(Path(ligand_sdf_path))
        meta["ligand_source"] = "sdf"
        meta["ligand_sdf"] = str(ligand_sdf_path)
    elif ligand_chain is not None and ligand_resseq is not None:
        centroid = _centroid_from_pdb_ligand_residue(pdb_path, ligand_chain, int(ligand_resseq))
        meta["ligand_source"] = "pdb_residue"
        meta["ligand_chain"] = ligand_chain
        meta["ligand_resseq"] = ligand_resseq
    else:
        raise ValueError("Provide ligand_sdf_path or (ligand_chain and ligand_resseq).")

    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, str(pdb_path))

    elems: list[str] = []
    names: list[str] = []
    resnames: list[str] = []
    resnums: list[int] = []
    chains: list[str] = []
    xyz: list[np.ndarray] = []

    rad2 = float(radius) ** 2
    for model in structure:
        for chain in model:
            for res in chain:
                # standard=False: include MSE, HIE, etc. Otherwise the binding shell can be
                # empty near the ligand while standard=True (common in deposited PDBs).
                if not is_aa(res, standard=False):
                    continue
                resname = res.get_resname().strip()
                resid = res.id[1]
                cid = chain.id
                for atom in res.get_atoms():
                    el = (atom.element or "").upper()
                    if el == "H" or el == "D":
                        continue
                    pos = np.array(atom.coord, dtype=np.float64)
                    d2 = float(np.sum((pos - centroid) ** 2))
                    if d2 > rad2:
                        continue
                    elems.append(el if el else "C")
                    names.append(str(atom.get_name()).strip())
                    resnames.append(resname)
                    resnums.append(int(resid))
                    chains.append(str(cid))
                    xyz.append(pos)

    if not xyz:
        raise RuntimeError(f"No pocket atoms within radius={radius} of ligand in {pdb_path}")

    resi_keys = sorted(
        {f"{str(cid).strip()}:{int(rn)}" for cid, rn in zip(chains, resnums, strict=True)}
    )

    rad_f = float(radius)
    meta["source_pdb"] = str(pdb_path.resolve())
    meta["extraction_radius"] = rad_f
    meta["base_extraction_radius"] = rad_f
    meta["ligand_centroid_xyz"] = [float(x) for x in centroid.tolist()]

    pocket = Pocket(
        pocket_id=pocket_id or pdb_path.stem,
        coords=np.stack(xyz, axis=0),
        elements=np.array(elems, dtype=object),
        atom_names=np.array(names, dtype=object),
        residue_names=np.array(resnames, dtype=object),
        residue_numbers=np.array(resnums, dtype=np.int64),
        chain_ids=np.array(chains, dtype=object),
        bonds=None,
        ligand_centroid=centroid,
        source_pdb=pdb_path,
        metadata={**meta, "perturbation_type": "original", "resi_list": resi_keys},
    )
    return attach_bonds_rdkit(pocket)
