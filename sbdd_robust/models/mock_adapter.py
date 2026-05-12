"""Fast deterministic fake generator for smoke tests and CI."""

from __future__ import annotations

from pathlib import Path
from typing import List

from rdkit import Chem
from rdkit.Chem import AllChem

from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.base_adapter import BaseSBDDAdapter


class MockSBDDAdapter(BaseSBDDAdapter):
    name = "mock"

    _POOL = (
        "CCO",
        "c1ccccc1",
        "CC(=O)OC1=CC=CC=C1",
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    )

    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        mols: List[Chem.Mol] = []
        for i in range(int(n_samples)):
            smi = self._POOL[i % len(self._POOL)]
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = Chem.AddHs(mol)
            coord_part = int(abs(float(pocket.coords.sum())) * 1e4) % 100000
            seed = 42 + i + (hash(pocket.pocket_id) % 10000) + coord_part
            AllChem.EmbedMolecule(mol, randomSeed=seed % (2**31 - 1))
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass
            mol = Chem.RemoveHs(mol)
            mols.append(mol)
        return mols
