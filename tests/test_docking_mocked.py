from __future__ import annotations

from pathlib import Path

import numpy as np
from rdkit import Chem

from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.metrics.docking import _best_affinity_from_vina_text
from sbdd_robust.metrics.docking import score_molecules


def _tiny_pocket() -> Pocket:
    coords = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
    return Pocket(
        pocket_id="dock_test",
        coords=coords,
        elements=np.array(["C"], dtype=object),
        atom_names=np.array(["CA"], dtype=object),
        residue_names=np.array(["ALA"], dtype=object),
        residue_numbers=np.array([1], dtype=np.int64),
        chain_ids=np.array(["A"], dtype=object),
        bonds=None,
        ligand_centroid=np.array([10.0, 10.0, 10.0], dtype=np.float64),
        source_pdb=Path("dummy.pdb"),
        metadata={},
    )


def test_best_affinity_from_vina_text_parses_mode_table():
    log = """
-----+------------+----------+----------
   1         -7.2      0.000      0.000
   2         -6.1      1.2        2.3
"""
    assert _best_affinity_from_vina_text(log) == -7.2


def test_score_molecules_returns_list_same_length(monkeypatch):
    monkeypatch.setattr(
        "sbdd_robust.metrics.docking._prepare_receptor_pdbqt",
        lambda *a, **k: (True, ""),
    )
    monkeypatch.setattr(
        "sbdd_robust.metrics.docking._mol_to_pdbqt", lambda *a, **k: (True, "")
    )
    monkeypatch.setattr(
        "sbdd_robust.metrics.docking._score_one_vina_binding",
        lambda *a, **k: (-6.0, ""),
    )
    monkeypatch.setattr(
        "sbdd_robust.metrics.docking._embed_mol", lambda m: (m, "")
    )

    p = _tiny_pocket()
    mols = [
        Chem.AddHs(Chem.MolFromSmiles("C")),
        Chem.MolFromSmiles("CC"),
        None,
    ]
    scores = score_molecules(mols, p, n_cpus=1)
    assert len(scores) == 3
    assert scores[0] == -6.0
    assert scores[1] == -6.0
    assert scores[2] is None
    assert all(s is None or isinstance(s, float) for s in scores)


def test_score_molecules_all_none_without_centroid(monkeypatch):
    monkeypatch.setattr(
        "sbdd_robust.metrics.docking._prepare_receptor_pdbqt",
        lambda *a, **k: (True, ""),
    )
    p = _tiny_pocket()
    p.ligand_centroid = None
    mol = Chem.MolFromSmiles("C")
    scores = score_molecules([mol], p)
    assert scores == [None]
