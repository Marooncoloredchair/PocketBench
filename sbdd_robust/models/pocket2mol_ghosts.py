"""Reference-pocket helpers for Pocket2Mol ghost-atom masking."""

from __future__ import annotations

from pathlib import Path

from sbdd_robust.datasets.pdb_merge import AtomKey, ghost_atom_keys
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.datasets.pocket_extraction import extract_pocket
from sbdd_robust.perturbations.invariant.crop_radius import (
    _base_radius,
    _ligand_kwargs_from_metadata,
    _reextraction_metadata,
)


def reference_pocket_for_masking(pocket: Pocket) -> Pocket:
    """Unperturbed pocket tensor at the baseline extraction radius."""
    tag = str(pocket.metadata.get("perturbation_tag", "original"))
    pdb = pocket.source_pdb or Path(str(pocket.metadata.get("source_pdb", "")))
    if not pdb.is_file():
        raise FileNotFoundError(f"source_pdb not found for ghost masking: {pdb}")

    lig_kw = _ligand_kwargs_from_metadata(pocket.metadata)
    carry = _reextraction_metadata(pocket)

    if tag.startswith("crop_radius"):
        radius = _base_radius(pocket)
    elif tag.startswith("face_peel"):
        radius = float(
            pocket.metadata.get(
                "extraction_radius",
                pocket.metadata.get("base_extraction_radius", 8.0),
            )
        )
    else:
        raise ValueError(f"no reference pocket for perturbation tag {tag!r}")

    return extract_pocket(
        pdb_path=pdb,
        radius=radius,
        pocket_id=pocket.pocket_id,
        metadata=carry,
        **lig_kw,
    )


def ghost_keys_for_frame_perturbation(pocket: Pocket) -> set[AtomKey]:
    ref = reference_pocket_for_masking(pocket)
    return ghost_atom_keys(ref, pocket)
