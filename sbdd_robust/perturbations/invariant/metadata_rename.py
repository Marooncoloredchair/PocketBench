"""Rename chain identifiers (label-only change)."""

from __future__ import annotations

from typing import Optional

import numpy as np

from sbdd_robust.datasets.pocket import Pocket


def rename_chains_cyclic(
    pocket: Pocket, rng: Optional[np.random.Generator] = None
) -> Pocket:
    """
    Apply a deterministic cyclic shift to chain id characters (A->B->...->Z->A).

    Preserves atom count and geometry; intended as a file-format / metadata stressor.
    """
    rng = rng or np.random.default_rng(0)
    p = pocket.copy()
    chains = [str(c)[0] if str(c) else "A" for c in p.chain_ids]
    uniq = sorted(set(chains))
    if len(uniq) < 2:
        shift = int(rng.integers(1, 26))
        mapping = {c: chr((ord(c) - ord("A") + shift) % 26 + ord("A")) for c in uniq or ["A"]}
    else:
        order = list(uniq)
        mapping = {order[i]: order[(i + 1) % len(order)] for i in range(len(order))}
    new_chains = np.array([mapping.get(c, c) for c in chains], dtype=object)
    p.chain_ids = new_chains
    p.metadata = dict(pocket.metadata)
    p.metadata["perturbation_type"] = "invariant_metadata_rename"
    p.metadata["perturbation_tag"] = "metadata_rename"
    return p
