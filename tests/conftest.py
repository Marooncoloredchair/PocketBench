from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def smoke_paths():
    return {
        "pdb": ROOT / "data" / "raw" / "smoke" / "shared_protein.pdb",
        # HET residue LIG A 99 in shared_protein.pdb (for tests without SDF parsing).
        "ligand_chain": "A",
        "ligand_resseq": 99,
    }
