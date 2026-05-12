from rdkit import Chem

from sbdd_robust.metrics import chemistry as chem


def test_summarize_molecules_basic():
    mols = [
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("CCO"),
        Chem.MolFromSmiles("c1ccccc1"),
        None,
    ]
    s = chem.summarize_molecules(mols)
    assert s["n_total"] == 4
    assert 0.0 <= s["validity"] <= 1.0
    assert s["n_valid"] >= 2


def test_summarize_empty():
    s = chem.summarize_molecules([])
    assert s["n_total"] == 0
    assert s["validity"] == 0.0
