"""Mutate a single pocket residue to a new amino-acid type (backbone + CB retained)."""

from __future__ import annotations

import copy
import re
from typing import FrozenSet

import numpy as np
from Bio.Data import IUPACData

from sbdd_robust.datasets.pdb_io import attach_bonds_rdkit
from sbdd_robust.datasets.pocket import Pocket

ONE_TO_THREE: dict[str, str] = {
    one: three.upper()[:3]
    for three, one in IUPACData.protein_letters_3to1.items()
    if len(one) == 1 and one.isalpha()
}

_BACKBONE: FrozenSet[str] = frozenset({"N", "CA", "C", "O", "OXT"})


def parse_residue_id(residue_id: str) -> tuple[str, int]:
    """
    Parse ``chain:resseq`` (e.g. ``A:48``).

    Chain may be multiple characters in theory; we take the segment before the last colon
    if more than one colon appears, else ``parts[0]`` and ``parts[1]``.
    """
    s = residue_id.strip()
    if ":" not in s:
        raise ValueError(f"residue_id must look like 'A:48', got {residue_id!r}")
    chain, num = s.rsplit(":", 1)
    chain = chain.strip()
    if not chain:
        raise ValueError(f"Empty chain in residue_id={residue_id!r}")
    return chain, int(num)


def _norm_atom_name(name: str) -> str:
    return str(name).strip().upper()


def _keep_atom_for_mutation(atom_name: str, target_one: str) -> bool:
    an = _norm_atom_name(atom_name)
    if an in _BACKBONE:
        return True
    if target_one == "G":
        return False
    return an == "CB"


def mutate_residue(pocket: Pocket, residue_id: str, target_aa: str) -> Pocket:
    """
    Mutate one residue in the pocket atom table.

    Backbone atoms ``N``, ``CA``, ``C``, ``O`` (and ``OXT`` if present) are kept as-is.
    All other atoms for that residue are dropped except ``CB``, which is kept when the
    target amino acid is not glycine (standard Ala-sized truncation model).

    Glycine targets keep backbone only. Mutating a glycine to a residue that requires ``CB``
    without an existing ``CB`` atom raises ``ValueError`` (no idealized placement here).
    """
    chain, resseq = parse_residue_id(residue_id)
    t = target_aa.strip().upper()
    if len(t) != 1 or t not in ONE_TO_THREE:
        raise ValueError(f"target_aa must be a single valid one-letter code, got {target_aa!r}")
    target_three = ONE_TO_THREE[t]

    n = pocket.coords.shape[0]
    keep: list[int] = []
    had_target = False
    for i in range(n):
        c = str(pocket.chain_ids[i]).strip()
        rn = int(pocket.residue_numbers[i])
        if c != chain or rn != resseq:
            keep.append(i)
            continue
        had_target = True
        aname = pocket.atom_names[i]
        if _keep_atom_for_mutation(str(aname), t):
            keep.append(i)

    if not had_target:
        raise ValueError(f"No atoms found for residue {residue_id!r} in pocket {pocket.pocket_id!r}")

    if t != "G":
        has_cb = any(
            _norm_atom_name(pocket.atom_names[i]) == "CB"
            for i in range(n)
            if str(pocket.chain_ids[i]).strip() == chain and int(pocket.residue_numbers[i]) == resseq
        )
        if not has_cb:
            raise ValueError(
                f"Cannot mutate {residue_id} to {target_aa}: target requires CB but "
                "source residue has no CB atom (e.g. glycine); idealized CB placement is not implemented."
            )

    idx = np.array(keep, dtype=np.int64)
    new_coords = pocket.coords[idx].copy()
    new_elems = pocket.elements[idx].copy()
    new_names = pocket.atom_names[idx].copy()
    new_resnames = pocket.residue_names[idx].copy()
    new_resnums = pocket.residue_numbers[idx].copy()
    new_chains = pocket.chain_ids[idx].copy()

    for j in range(new_coords.shape[0]):
        if str(new_chains[j]).strip() == chain and int(new_resnums[j]) == resseq:
            new_resnames[j] = target_three

    new_bonds = None
    if pocket.bonds is not None and pocket.bonds.size:
        inv = -np.ones(n, dtype=np.int64)
        inv[idx] = np.arange(idx.shape[0], dtype=np.int64)
        rows: list[tuple[int, int]] = []
        for a, b in pocket.bonds:
            ia, ib = int(a), int(b)
            if inv[ia] >= 0 and inv[ib] >= 0:
                u, v = sorted((int(inv[ia]), int(inv[ib])))
                rows.append((u, v))
        if rows:
            arr = np.array(sorted(set(rows)), dtype=np.int64)
            new_bonds = arr

    meta = copy.deepcopy(pocket.metadata)
    rid_safe = re.sub(r"[^\w.\-]+", "_", residue_id)
    meta["perturbation_type"] = "meaningful"
    meta["perturbation_tag"] = f"mutate_{residue_id}_{t}"
    meta["mutation_residue_id"] = residue_id
    meta["mutation_target_aa"] = t
    meta["mutation_target_three"] = target_three
    meta["mutation_residue_id_safe"] = rid_safe

    new_p = Pocket(
        pocket_id=pocket.pocket_id,
        coords=new_coords,
        elements=new_elems,
        atom_names=new_names,
        residue_names=new_resnames,
        residue_numbers=new_resnums,
        chain_ids=new_chains,
        bonds=new_bonds,
        ligand_centroid=None if pocket.ligand_centroid is None else pocket.ligand_centroid.copy(),
        source_pdb=pocket.source_pdb,
        metadata=meta,
    )
    return attach_bonds_rdkit(new_p)


def mutate_active_site_residue(pocket: Pocket, residue_id: str, target_aa: str) -> Pocket:
    """Alias for :func:`mutate_residue` (historical name)."""
    return mutate_residue(pocket, residue_id, target_aa)
