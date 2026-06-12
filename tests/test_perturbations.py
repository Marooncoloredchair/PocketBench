from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from sbdd_robust.datasets.pocket_extraction import extract_pocket
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.perturbations.invariant import atom_shuffle, coordinate_jitter, crop_radius
from sbdd_robust.perturbations.invariant import anchor_offset as anchor_offset_mod
from sbdd_robust.perturbations.invariant import directional_crop
from sbdd_robust.perturbations.invariant import metadata_rename as meta_rename


def _atom_identity_key(p: Pocket, i: int) -> tuple:
    return (
        str(p.chain_ids[i]).strip(),
        int(p.residue_numbers[i]),
        str(p.residue_names[i]).strip(),
        str(p.atom_names[i]).strip(),
    )


def _connectivity_multiset(p: Pocket) -> Counter:
    if p.bonds is None or p.bonds.size == 0:
        return Counter()
    c = Counter()
    for a, b in p.bonds:
        ka = _atom_identity_key(p, int(a))
        kb = _atom_identity_key(p, int(b))
        c[tuple(sorted((ka, kb)))] += 1
    return c


def _residue_identity_multiset(p: Pocket) -> Counter:
    keys = [
        (str(c).strip(), int(rn), str(rnme).strip())
        for c, rn, rnme in zip(p.chain_ids, p.residue_numbers, p.residue_names, strict=True)
    ]
    return Counter(keys)


@pytest.fixture
def extracted_pocket(smoke_paths):
    return extract_pocket(
        pdb_path=smoke_paths["pdb"],
        ligand_sdf_path=None,
        radius=12.0,
        pocket_id="test_pocket",
        ligand_chain=smoke_paths["ligand_chain"],
        ligand_resseq=int(smoke_paths["ligand_resseq"]),
        metadata={},
    )


def test_atom_shuffle_preserves_counts_topology_identity(extracted_pocket):
    rng = np.random.default_rng(42)
    p2 = atom_shuffle.shuffle_atom_order(extracted_pocket, rng=rng)
    assert p2.coords.shape[0] == extracted_pocket.coords.shape[0]
    assert _residue_identity_multiset(p2) == _residue_identity_multiset(extracted_pocket)
    if extracted_pocket.bonds is not None and extracted_pocket.bonds.size:
        assert _connectivity_multiset(p2) == _connectivity_multiset(extracted_pocket)


def test_coordinate_jitter_preserves_counts_bonds_identity(extracted_pocket):
    rng = np.random.default_rng(0)
    p2 = coordinate_jitter.jitter_coordinates(extracted_pocket, sigma=0.05, rng=rng)
    assert p2.coords.shape[0] == extracted_pocket.coords.shape[0]
    if extracted_pocket.bonds is None:
        assert p2.bonds is None or (p2.bonds is not None and p2.bonds.size == 0)
    else:
        assert np.array_equal(p2.bonds, extracted_pocket.bonds)
    assert _residue_identity_multiset(p2) == _residue_identity_multiset(extracted_pocket)
    assert not np.allclose(p2.coords, extracted_pocket.coords)


def test_metadata_rename_preserves_counts_and_geometry(extracted_pocket):
    p2 = meta_rename.rename_chains_cyclic(extracted_pocket, rng=np.random.default_rng(0))
    assert p2.coords.shape[0] == extracted_pocket.coords.shape[0]
    assert np.allclose(p2.coords, extracted_pocket.coords)
    assert np.array_equal(p2.bonds, extracted_pocket.bonds)
    assert Counter(p2.chain_ids.tolist()) != Counter(extracted_pocket.chain_ids.tolist()) or len(
        set(extracted_pocket.chain_ids.tolist())
    ) < 2


def _coord_map(p: Pocket) -> dict[tuple[str, int, str, str], np.ndarray]:
    m: dict[tuple[str, int, str, str], np.ndarray] = {}
    for i in range(p.coords.shape[0]):
        k = _atom_identity_key(p, i)
        m[k] = p.coords[i].copy()
    return m


def test_pocket_extraction_metadata_has_provenance(extracted_pocket):
    m = extracted_pocket.metadata
    assert "source_pdb" in m
    assert Path(m["source_pdb"]).is_file()
    assert m.get("base_extraction_radius") == pytest.approx(12.0)
    assert m.get("extraction_radius") == pytest.approx(12.0)
    assert "ligand_centroid_xyz" in m
    assert len(m["ligand_centroid_xyz"]) == 3
    assert extracted_pocket.ligand_centroid is not None
    assert np.allclose(
        np.array(m["ligand_centroid_xyz"], dtype=np.float64),
        extracted_pocket.ligand_centroid,
    )


def test_crop_radius_plus_more_atoms_coord_superset(extracted_pocket):
    p_plus = crop_radius.crop_radius_plus(extracted_pocket, delta_angstrom=2.0)
    assert p_plus.coords.shape[0] > extracted_pocket.coords.shape[0]
    assert p_plus.metadata["perturbation_type"] == "invariant"
    assert p_plus.metadata["perturbation_tag"] == "crop_radius_plus_2.0"
    mo, mp = _coord_map(extracted_pocket), _coord_map(p_plus)
    for k, v in mo.items():
        assert k in mp
        assert np.allclose(mp[k], v, atol=1e-4, rtol=0)


def test_crop_radius_minus_fewer_atoms_coord_subset(extracted_pocket):
    p_minus = crop_radius.crop_radius_minus(extracted_pocket, delta_angstrom=4.0)
    assert p_minus.coords.shape[0] < extracted_pocket.coords.shape[0]
    assert p_minus.metadata["perturbation_type"] == "invariant"
    assert p_minus.metadata["perturbation_tag"] == "crop_radius_minus_4.0"
    mo, mm = _coord_map(extracted_pocket), _coord_map(p_minus)
    for k, v in mm.items():
        assert k in mo
        assert np.allclose(mo[k], v, atol=1e-4, rtol=0)


def test_face_peel_removes_one_side_and_shifts_centroid(extracted_pocket):
    p = directional_crop.face_peel(extracted_pocket, fraction=0.25, axis="pca", direction="plus")
    # Subset of residues, atom coords preserved for kept atoms (no jitter).
    assert p.coords.shape[0] < extracted_pocket.coords.shape[0]
    assert _residue_identity_multiset(p).keys() <= _residue_identity_multiset(extracted_pocket).keys()
    mo, mp = _coord_map(extracted_pocket), _coord_map(p)
    for k, v in mp.items():
        assert k in mo and np.allclose(mo[k], v, atol=1e-4, rtol=0)
    # Centroid moves opposite the peeled (+pca) face.
    assert not np.allclose(p.coords.mean(axis=0), extracted_pocket.coords.mean(axis=0))
    # resi_list rebuilt to the surviving residues so DiffSBDD sees the peel.
    kept = {f"{str(c).strip()}:{int(rn)}" for c, rn in zip(p.chain_ids, p.residue_numbers)}
    assert set(p.metadata["resi_list"]) == kept
    assert p.metadata["perturbation_tag"] == "face_peel_0.25"
    assert p.metadata["face_peel_residues_removed"] >= 1


def test_face_peel_fraction_is_monotone(extracted_pocket):
    small = directional_crop.face_peel(extracted_pocket, fraction=0.15, axis="pca")
    big = directional_crop.face_peel(extracted_pocket, fraction=0.40, axis="pca")
    assert big.coords.shape[0] <= small.coords.shape[0]
    assert big.metadata["face_peel_residues_removed"] >= small.metadata["face_peel_residues_removed"]


def test_anchor_offset_keeps_coords_records_offset(extracted_pocket):
    p = anchor_offset_mod.anchor_offset(extracted_pocket, offset_angstrom=2.0, direction="pca")
    # Coordinates and atom set are untouched; only the seeding frame metadata changes.
    assert np.allclose(p.coords, extracted_pocket.coords)
    assert p.coords.shape[0] == extracted_pocket.coords.shape[0]
    off = np.asarray(p.metadata["center_offset"], dtype=float)
    assert off.shape == (3,)
    assert float(np.linalg.norm(off)) == pytest.approx(2.0, abs=1e-6)
    assert p.metadata["perturbation_tag"] == "anchor_offset_2.0"
