"""Initialization-frame attack: offset the first-atom seeding region, chemistry fixed.

This is the most direct test of Prof. Ma's "first-atom context problem": it leaves the
pocket atoms and their relative geometry completely unchanged and only shifts the
*sampling frame* (the center used to seed the autoregressive first atom, and/or the
bounding-box scale). A model whose output depends on where atom 1 is initialized
(Pocket2Mol) should degrade sharply; a translation/scale-robust diffusion model should
be unaffected.

The offset is recorded in metadata only. ``Pocket2MolAdapter`` forwards
``--center_offset`` / ``--bbox_scale`` to the sampling bridge. For adapters that derive
their own pocket frame from residues (DiffSBDD), this perturbation is a deliberate
**no-op**, which is exactly the control the cross-architecture ISR contrast needs.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from sbdd_robust.datasets.pocket import Pocket


def anchor_offset(
    pocket: Pocket,
    offset_angstrom: float = 2.0,
    direction: str | Sequence[float] = "pca",
    bbox_scale: float = 1.0,
    tag: Optional[str] = None,
) -> Pocket:
    """
    Record a center offset of magnitude ``offset_angstrom`` along ``direction`` (and an
    optional ``bbox_scale``) in metadata, without touching coordinates.

    ``direction`` may be ``"pca"`` (first principal axis of the pocket), ``"ligand"``
    (pocket-centroid → ligand-centroid), or an explicit 3-vector.
    """
    mag = float(offset_angstrom)
    if isinstance(direction, str):
        key = direction.lower()
        if key == "pca":
            centered = pocket.coords - pocket.coords.mean(axis=0, keepdims=True)
            if centered.shape[0] >= 2:
                _, _, vt = np.linalg.svd(centered, full_matrices=False)
                unit = vt[0]
            else:
                unit = np.array([1.0, 0.0, 0.0])
        elif key == "ligand":
            if pocket.ligand_centroid is None:
                unit = np.array([1.0, 0.0, 0.0])
            else:
                unit = np.asarray(pocket.ligand_centroid, dtype=np.float64) - pocket.coords.mean(axis=0)
        else:
            raise ValueError(f"direction must be 'pca', 'ligand', or a 3-vector; got {direction!r}")
    else:
        unit = np.asarray(direction, dtype=np.float64).reshape(3)

    n = float(np.linalg.norm(unit))
    unit = unit / n if n > 0 else np.array([1.0, 0.0, 0.0])
    offset_vec = (mag * unit).astype(np.float64)

    p = pocket.copy()
    p.metadata = dict(pocket.metadata)
    p.metadata["perturbation_type"] = "invariant_anchor_offset"
    p.metadata["perturbation_tag"] = tag or f"anchor_offset_{round(mag, 3)}"
    p.metadata["center_offset"] = [float(x) for x in offset_vec.tolist()]
    p.metadata["anchor_offset_magnitude"] = mag
    p.metadata["bbox_scale"] = float(bbox_scale)
    return p
