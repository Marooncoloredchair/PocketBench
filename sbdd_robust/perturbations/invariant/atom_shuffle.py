"""Shuffle atom row ordering (metadata / file layout) without changing chemistry."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sbdd_robust.datasets.pocket import Pocket


def shuffle_atom_order(pocket: Pocket, rng: Optional[np.random.Generator] = None) -> Pocket:
    rng = rng or np.random.default_rng()
    n = pocket.coords.shape[0]
    perm = rng.permutation(n)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(n)

    p = pocket.copy()
    p.coords = pocket.coords[perm].copy()
    p.elements = pocket.elements[perm].copy()
    p.atom_names = pocket.atom_names[perm].copy()
    p.residue_names = pocket.residue_names[perm].copy()
    p.residue_numbers = pocket.residue_numbers[perm].copy()
    p.chain_ids = pocket.chain_ids[perm].copy()
    if pocket.bonds is not None and pocket.bonds.size:
        new_b = inv[pocket.bonds]
        # normalize (min,max) per row for consistency
        lo = np.minimum(new_b[:, 0], new_b[:, 1])
        hi = np.maximum(new_b[:, 0], new_b[:, 1])
        p.bonds = np.stack([lo, hi], axis=1)
        order = np.lexsort((p.bonds[:, 1], p.bonds[:, 0]))
        p.bonds = p.bonds[order]
    elif pocket.bonds is not None:
        p.bonds = pocket.bonds.copy()

    p.metadata = dict(pocket.metadata)
    p.metadata["perturbation_type"] = "invariant_atom_shuffle"
    p.metadata["perturbation_tag"] = "atom_shuffle"
    return p
