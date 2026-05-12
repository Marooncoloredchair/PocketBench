"""Abstract interface: pocket in, RDKit molecules out."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List

from rdkit import Chem

from sbdd_robust.datasets.pocket import Pocket


class BaseSBDDAdapter(ABC):
    """Model wrapper used by the benchmark CLI."""

    name: str = "base"

    @abstractmethod
    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        """Generate ``n_samples`` ligand hypotheses for the given pocket."""

    def cleanup(self, workdir: Path) -> None:
        """Optional hook for temporary files (default: no-op)."""
        return
