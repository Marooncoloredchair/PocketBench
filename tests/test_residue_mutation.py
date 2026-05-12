from __future__ import annotations

import numpy as np

from sbdd_robust.datasets.pdb_io import attach_bonds_rdkit
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.perturbations.meaningful.residue_mutation import mutate_residue


def _backbone_heavy_count(p: Pocket, chain: str, resseq: int) -> int:
    bb = {"N", "CA", "C", "O", "OXT"}
    n = 0
    for i in range(p.coords.shape[0]):
        if str(p.chain_ids[i]).strip() != chain or int(p.residue_numbers[i]) != resseq:
            continue
        if str(p.atom_names[i]).strip().upper() in bb:
            n += 1
    return n


def _sidechain_heavy_count(p: Pocket, chain: str, resseq: int) -> int:
    bb = {"N", "CA", "C", "O", "OXT"}
    n = 0
    for i in range(p.coords.shape[0]):
        if str(p.chain_ids[i]).strip() != chain or int(p.residue_numbers[i]) != resseq:
            continue
        if str(p.atom_names[i]).strip().upper() not in bb:
            n += 1
    return n


def _residue_name_at(p: Pocket, chain: str, resseq: int) -> str:
    for i in range(p.coords.shape[0]):
        if str(p.chain_ids[i]).strip() == chain and int(p.residue_numbers[i]) == resseq:
            return str(p.residue_names[i]).strip()
    raise AssertionError("residue not found")


def _make_arg_like_pocket() -> Pocket:
    """Single non-glycine residue with backbone + CB + extra sidechain heavies."""
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.5, 0.0, 0.0],
            [2.0, 1.4, 0.0],
            [1.2, 2.2, 0.0],
            [2.5, -0.8, 1.0],
            [3.8, -0.2, 1.2],
            [4.5, 0.9, 0.5],
            [5.1, 2.0, 0.8],
        ],
        dtype=np.float64,
    )
    elements = np.array(["N", "C", "C", "O", "C", "C", "C", "N"], dtype=object)
    atom_names = np.array(["N", "CA", "C", "O", "CB", "CG", "CD", "NE"], dtype=object)
    residue_names = np.array(["ARG"] * 8, dtype=object)
    residue_numbers = np.array([50] * 8, dtype=np.int64)
    chain_ids = np.array(["A"] * 8, dtype=object)
    p = Pocket(
        pocket_id="synthetic_arg",
        coords=coords,
        elements=elements,
        atom_names=atom_names,
        residue_names=residue_names,
        residue_numbers=residue_numbers,
        chain_ids=chain_ids,
        bonds=None,
        ligand_centroid=np.array([0.0, 0.0, 0.0], dtype=np.float64),
        source_pdb=None,
        metadata={"ligand_source": "pdb_residue", "ligand_chain": "A", "ligand_resseq": 99},
    )
    return attach_bonds_rdkit(p)


def test_mutate_non_gly_to_alanine_truncates_sidechain_preserves_backbone():
    p0 = _make_arg_like_pocket()
    chain, resseq = "A", 50
    assert _backbone_heavy_count(p0, chain, resseq) == 4
    assert _sidechain_heavy_count(p0, chain, resseq) == 4

    p1 = mutate_residue(p0, "A:50", "A")
    assert p1.metadata["perturbation_type"] == "meaningful"
    assert p1.metadata["perturbation_tag"] == "mutate_A:50_A"

    assert _backbone_heavy_count(p1, chain, resseq) == 4
    assert _sidechain_heavy_count(p1, chain, resseq) == 1
    assert _sidechain_heavy_count(p1, chain, resseq) < _sidechain_heavy_count(p0, chain, resseq)

    assert _residue_name_at(p1, chain, resseq) == "ALA"
    assert _residue_name_at(p0, chain, resseq) == "ARG"

    assert p1.coords.shape[0] == p0.coords.shape[0] - 3
