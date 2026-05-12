"""Re-extract pocket atoms with a larger or smaller radius around the same ligand centroid."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.datasets.pocket_extraction import extract_pocket


def _base_radius(pocket: Pocket) -> float:
    m = pocket.metadata
    if "base_extraction_radius" in m:
        return float(m["base_extraction_radius"])
    if "extraction_radius" in m:
        return float(m["extraction_radius"])
    raise ValueError(
        "Pocket metadata missing base_extraction_radius / extraction_radius; "
        "re-run pocket_extraction.extract_pocket or set before crop_radius."
    )


def _ligand_kwargs_from_metadata(meta: dict) -> dict:
    src = meta.get("ligand_source")
    if src == "sdf":
        sdf = meta.get("ligand_sdf")
        if not sdf:
            raise ValueError("metadata ligand_sdf missing for ligand_source=sdf")
        return {"ligand_sdf_path": Path(sdf), "ligand_chain": None, "ligand_resseq": None}
    if src == "pdb_residue":
        lc = meta.get("ligand_chain")
        lr = meta.get("ligand_resseq")
        if lc is None or lr is None:
            raise ValueError("metadata ligand_chain / ligand_resseq missing for pdb_residue")
        return {"ligand_sdf_path": None, "ligand_chain": str(lc), "ligand_resseq": int(lr)}
    raise ValueError(f"Unknown or missing ligand_source in metadata: {src!r}")


def _reextraction_metadata(pocket: Pocket) -> dict:
    """Fields needed by extract_pocket plus passthrough (e.g. full_pdb)."""
    m = dict(pocket.metadata)
    for k in (
        "perturbation_type",
        "perturbation_tag",
        "brittle_invariant",
        "brittleness_note",
        "resi_list",
    ):
        m.pop(k, None)
    return m


def _atom_key_row(p: Pocket, i: int) -> tuple[str, int, str, str]:
    return (
        str(p.chain_ids[i]).strip(),
        int(p.residue_numbers[i]),
        str(p.residue_names[i]).strip(),
        str(p.atom_names[i]).strip(),
    )


def _coord_map_by_key(p: Pocket) -> dict[tuple[str, int, str, str], np.ndarray]:
    return {_atom_key_row(p, i): p.coords[i].copy() for i in range(p.coords.shape[0])}


def crop_radius_reextract(
    pocket: Pocket,
    delta_angstrom: float = 1.5,
    direction: Literal["plus", "minus"] = "plus",
) -> Pocket:
    """
    Re-extract amino-acid heavy atoms from the same ``source_pdb`` around the same
    ligand definition, using ``base_extraction_radius ± delta_angstrom``.
    """
    pdb = pocket.source_pdb or Path(str(pocket.metadata.get("source_pdb", "")))
    if not pdb.is_file():
        raise FileNotFoundError(f"source_pdb not found for crop_radius: {pdb}")

    r0 = _base_radius(pocket)
    d = float(delta_angstrom)
    if direction == "plus":
        r_new = r0 + d
        tag = f"crop_radius_plus_{d}"
    elif direction == "minus":
        r_new = max(0.0, r0 - d)
        tag = f"crop_radius_minus_{d}"
    else:
        raise ValueError(f"direction must be 'plus' or 'minus', got {direction!r}")

    lig_kw = _ligand_kwargs_from_metadata(pocket.metadata)
    carry = _reextraction_metadata(pocket)

    new_p = extract_pocket(
        pdb_path=pdb,
        radius=r_new,
        pocket_id=pocket.pocket_id,
        metadata=carry,
        **lig_kw,
    )
    new_p.metadata = dict(new_p.metadata)
    new_p.metadata["base_extraction_radius"] = r0
    new_p.metadata["extraction_radius"] = r_new
    new_p.metadata["perturbation_type"] = "invariant"
    new_p.metadata["perturbation_tag"] = tag
    new_p.metadata["crop_radius_delta"] = d
    new_p.metadata["crop_radius_direction"] = direction
    new_p.metadata["base_extraction_radius"] = r0
    new_p.metadata["extraction_radius"] = r_new
    return new_p


def crop_radius_plus(pocket: Pocket, delta_angstrom: float = 1.5) -> Pocket:
    return crop_radius_reextract(pocket, delta_angstrom=delta_angstrom, direction="plus")


def crop_radius_minus(pocket: Pocket, delta_angstrom: float = 1.5) -> Pocket:
    return crop_radius_reextract(pocket, delta_angstrom=delta_angstrom, direction="minus")
