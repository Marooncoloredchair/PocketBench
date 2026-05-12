from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np


@dataclass
class Pocket:
    """Binding pocket represented as heavy-atom arrays plus optional bond list."""

    pocket_id: str
    coords: np.ndarray  # (N, 3) float64
    elements: np.ndarray  # (N,) object — element symbols, e.g. "C", "N"
    atom_names: np.ndarray  # (N,) object — PDB atom names
    residue_names: np.ndarray  # (N,) object — three-letter codes
    residue_numbers: np.ndarray  # (N,) int — residue sequence numbers
    chain_ids: np.ndarray  # (N,) object — single-character chain ids
    bonds: Optional[np.ndarray] = None  # (B, 2) int64 — 0-based atom indices
    ligand_centroid: Optional[np.ndarray] = None  # (3,) for traceability
    source_pdb: Optional[Path] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = self.coords.shape[0]
        for name in (
            "elements",
            "atom_names",
            "residue_names",
            "residue_numbers",
            "chain_ids",
        ):
            arr = getattr(self, name)
            if arr.shape[0] != n:
                raise ValueError(f"{name} length {arr.shape[0]} != N={n}")
        if self.bonds is not None and self.bonds.ndim != 2:
            raise ValueError("bonds must be (B, 2)")

    def copy(self) -> Pocket:
        return Pocket(
            pocket_id=self.pocket_id,
            coords=np.array(self.coords, copy=True),
            elements=np.array(self.elements, copy=True),
            atom_names=np.array(self.atom_names, copy=True),
            residue_names=np.array(self.residue_names, copy=True),
            residue_numbers=np.array(self.residue_numbers, copy=True),
            chain_ids=np.array(self.chain_ids, copy=True),
            bonds=None if self.bonds is None else np.array(self.bonds, copy=True),
            ligand_centroid=None
            if self.ligand_centroid is None
            else np.array(self.ligand_centroid, copy=True),
            source_pdb=self.source_pdb,
            metadata=copy.deepcopy(self.metadata),
        )

    def with_metadata(self, **kwargs: Any) -> Pocket:
        p = self.copy()
        p.metadata.update(kwargs)
        return p

    def residue_keys(self) -> list[tuple[str, int, str]]:
        """(chain, residue_number, residue_name) per atom — order follows atom order."""
        return [
            (str(c), int(rn), str(rname))
            for c, rn, rname in zip(
                self.chain_ids, self.residue_numbers, self.residue_names, strict=True
            )
        ]
