"""Load protein–ligand complexes from PDB and SDF."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from Bio.PDB import PDBParser

from sbdd_robust.datasets.pocket_extraction import extract_pocket


@dataclass
class ComplexPaths:
    """Paths to a single PDB structure and optional cognate ligand SDF."""

    pdb_path: Path
    sdf_path: Optional[Path] = None
    pocket_id: Optional[str] = None


def load_complex_paths(pdb_path: Path, sdf_path: Optional[Path] = None) -> ComplexPaths:
    return ComplexPaths(
        pdb_path=Path(pdb_path),
        sdf_path=Path(sdf_path) if sdf_path else None,
        pocket_id=Path(pdb_path).stem,
    )


def load_structure_biopython(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    return parser.get_structure(pdb_path.stem, str(pdb_path))


def load_pocket_from_complex(
    pdb_path: Path,
    sdf_path: Optional[Path] = None,
    radius: float = 8.0,
    pocket_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    ligand_chain: Optional[str] = None,
    ligand_resseq: Optional[int] = None,
):
    """Convenience: load PDB and extract pocket within ``radius`` Å."""
    meta = dict(metadata or {})
    return extract_pocket(
        pdb_path=Path(pdb_path),
        ligand_sdf_path=Path(sdf_path) if sdf_path else None,
        radius=float(radius),
        pocket_id=pocket_id or Path(pdb_path).stem,
        metadata=meta,
        ligand_chain=ligand_chain,
        ligand_resseq=int(ligand_resseq) if ligand_resseq is not None else None,
    )
