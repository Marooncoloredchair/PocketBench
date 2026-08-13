"""Tests for selective ghost-atom masking in full-PDB merge."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.datasets.pdb_merge import (
    count_merged_protein_atoms,
    ghost_atom_keys,
    merge_pocket_into_full_pdb,
)
from sbdd_robust.models.pocket2mol_adapter import _ligand_centered_bbox
from sbdd_robust.models.pocket2mol_ghosts import ghost_keys_for_frame_perturbation
from sbdd_robust.perturbations.invariant import crop_radius, directional_crop


@pytest.fixture
def matched_pocket():
    pdb = Path("data/raw/real50/1ao7.pdb")
    if not pdb.is_file():
        pytest.skip("1ao7.pdb not in tree")
    base = load_pocket_from_complex(
        pdb,
        radius=8.0,
        pocket_id="1AO7",
        ligand_chain="B",
        ligand_resseq=100,
        metadata={"source_pdb": str(pdb), "base_extraction_radius": 8.0, "extraction_radius": 8.0},
    )
    base.metadata["base_extraction_radius"] = 8.0
    base.metadata["extraction_radius"] = 8.0
    base.metadata["ligand_source"] = "pdb_residue"
    base.metadata["ligand_chain"] = "B"
    base.metadata["ligand_resseq"] = 100
    return base


def test_selective_masking_keeps_more_atoms_than_aggressive(matched_pocket):
    cropped = crop_radius.crop_radius_minus(matched_pocket, delta_angstrom=1.5)
    cropped.metadata["perturbation_tag"] = "crop_radius_minus_1.5"
    full_pdb = Path("data/raw/real50/1ao7.pdb")
    bbox = _ligand_centered_bbox(cropped)
    center = cropped.ligand_centroid
    half = bbox / 2.0
    ghosts = ghost_keys_for_frame_perturbation(cropped)

    aggressive = merge_pocket_into_full_pdb(cropped, full_pdb, bbox_size=bbox)
    selective = merge_pocket_into_full_pdb(
        cropped, full_pdb, bbox_size=bbox, ghost_atom_keys=ghosts
    )
    try:
        aggr_n = count_merged_protein_atoms(aggressive, center=center, half=half)["in_bbox"]
        sel_n = count_merged_protein_atoms(selective, center=center, half=half)["in_bbox"]
        assert sel_n > aggr_n
        assert len(ghosts) > 0
    finally:
        aggressive.unlink(missing_ok=True)
        selective.unlink(missing_ok=True)


def test_face_peel_ghost_keys_are_subset_of_reference(matched_pocket):
    peeled = directional_crop.face_peel(matched_pocket, fraction=0.25, axis="pca")
    peeled.metadata["perturbation_tag"] = "face_peel_0.25"
    ref = load_pocket_from_complex(
        Path("data/raw/real50/1ao7.pdb"),
        radius=8.0,
        pocket_id="1AO7",
        ligand_chain="B",
        ligand_resseq=100,
        metadata=matched_pocket.metadata,
    )
    ghosts = ghost_atom_keys(ref, peeled)
    assert len(ghosts) > 0
    assert peeled.metadata.get("face_peel_removed_residue_keys")
