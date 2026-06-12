"""Asymmetric "face-peel" crop: remove pocket residues on one side of the pocket.

Motivation (Prof. Ma, personal communication): Pocket2Mol places atoms
autoregressively, so the *first* atom — seeded from the pocket representation — is
error-prone, and a misplaced first atom mispositions the whole molecule. Diffusion
models (TargetDiff, DiffSBDD) denoise all atoms jointly and have no fragile first atom.

A symmetric crop-radius change removes context evenly and barely moves the pocket
centroid. A *directional* peel removes residues on one face, which (a) deletes context
and (b) **shifts the pocket centroid / initialization frame**. The mechanism predicts a
directional peel degrades an autoregressive model far more than a symmetric crop of
equal residue count, and far more than it degrades a diffusion model. This perturbation
operates purely in pocket space (residue granularity, ``resi_list`` rebuilt), so it
feeds every model adapter identically and supports a fair cross-architecture contrast.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from sbdd_robust.datasets.pocket import Pocket


def _principal_axis(points: np.ndarray) -> np.ndarray:
    centered = points - points.mean(axis=0, keepdims=True)
    if centered.shape[0] < 2:
        return np.array([1.0, 0.0, 0.0])
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    n = float(np.linalg.norm(axis))
    return axis / n if n > 0 else np.array([1.0, 0.0, 0.0])


def _resolve_axis(pocket: Pocket, axis: str | Sequence[float]) -> np.ndarray:
    if isinstance(axis, str):
        key = axis.lower()
        if key == "pca":
            return _principal_axis(pocket.coords)
        if key == "ligand":
            # Direction from pocket centroid toward the ligand centroid (the "mouth").
            if pocket.ligand_centroid is None:
                return _principal_axis(pocket.coords)
            v = np.asarray(pocket.ligand_centroid, dtype=np.float64) - pocket.coords.mean(axis=0)
            n = float(np.linalg.norm(v))
            return v / n if n > 0 else _principal_axis(pocket.coords)
        raise ValueError(f"axis must be 'pca', 'ligand', or a 3-vector; got {axis!r}")
    vec = np.asarray(axis, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(vec))
    if n == 0:
        raise ValueError("axis vector must be non-zero")
    return vec / n


def face_peel(
    pocket: Pocket,
    fraction: float = 0.25,
    axis: str | Sequence[float] = "pca",
    direction: str = "plus",
    min_residues_kept: int = 3,
    tag: Optional[str] = None,
) -> Pocket:
    """
    Remove the ``fraction`` of pocket residues whose centroids project farthest along
    ``axis`` (``direction="plus"``) or farthest in the opposite sense (``"minus"``).

    Residue granularity keeps residues intact and lets us rebuild ``resi_list`` so the
    peel reaches DiffSBDD (residue-defined pocket) as well as Pocket2Mol (atom set +
    derived bbox). The pocket centroid shifts toward the surviving face.
    """
    f = float(fraction)
    if not 0.0 < f < 1.0:
        raise ValueError(f"fraction must be in (0, 1); got {fraction!r}")
    if direction not in ("plus", "minus"):
        raise ValueError(f"direction must be 'plus' or 'minus'; got {direction!r}")

    unit = _resolve_axis(pocket, axis)
    if direction == "minus":
        unit = -unit

    # Per-residue grouping (chain, resnum) preserving first-seen order.
    keys = list(zip(pocket.chain_ids.tolist(), pocket.residue_numbers.tolist()))
    order: list[tuple] = []
    members: dict[tuple, list[int]] = {}
    for i, k in enumerate(keys):
        if k not in members:
            members[k] = []
            order.append(k)
        members[k].append(i)

    n_res = len(order)
    if n_res <= min_residues_kept:
        raise ValueError(
            f"pocket has only {n_res} residues; cannot peel and keep >= {min_residues_kept}"
        )

    ref = pocket.coords.mean(axis=0)
    proj = {k: float(np.dot(pocket.coords[idx].mean(axis=0) - ref, unit)) for k, idx in members.items()}

    n_remove = int(round(f * n_res))
    n_remove = max(1, min(n_remove, n_res - min_residues_kept))
    # Residues with the largest projection along +unit are on the peeled face.
    to_remove = set(sorted(order, key=lambda k: proj[k], reverse=True)[:n_remove])
    keep_idx = np.array(
        [i for i, k in enumerate(keys) if k not in to_remove], dtype=np.int64
    )
    if keep_idx.size == 0:
        raise ValueError("face_peel removed all atoms; lower fraction")

    p = pocket.copy()
    p.coords = pocket.coords[keep_idx]
    p.elements = pocket.elements[keep_idx]
    p.atom_names = pocket.atom_names[keep_idx]
    p.residue_names = pocket.residue_names[keep_idx]
    p.residue_numbers = pocket.residue_numbers[keep_idx]
    p.chain_ids = pocket.chain_ids[keep_idx]
    p.bonds = None  # atom indices changed; let the adapter re-derive bonds if needed

    resi_list = sorted(
        {f"{str(c).strip()}:{int(rn)}" for c, rn in zip(p.chain_ids, p.residue_numbers)}
    )
    p.metadata = dict(pocket.metadata)
    p.metadata["resi_list"] = resi_list
    p.metadata["perturbation_type"] = "invariant_directional_crop"
    p.metadata["perturbation_tag"] = tag or f"face_peel_{round(f, 3)}"
    p.metadata["face_peel_fraction"] = f
    p.metadata["face_peel_axis"] = (
        axis if isinstance(axis, str) else [float(x) for x in np.asarray(axis).reshape(3)]
    )
    p.metadata["face_peel_direction"] = direction
    p.metadata["face_peel_residues_removed"] = int(n_remove)
    p.metadata["face_peel_residues_kept"] = int(n_res - n_remove)
    return p
