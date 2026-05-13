"""Tests for Pocket2Mol adapter subprocess wrapper and .pt parsing."""

from __future__ import annotations

from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import numpy as np
import pytest
import torch
from rdkit import Chem

from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.pocket2mol_adapter import (
    Pocket2MolAdapter,
    mol_from_atom_arrays,
    mols_from_pocket2mol_payload,
)


def _tiny_pocket() -> Pocket:
    return Pocket(
        pocket_id="t1",
        coords=np.zeros((1, 3), dtype=np.float64),
        elements=np.array(["C"], dtype=object),
        atom_names=np.array(["CA"], dtype=object),
        residue_names=np.array(["ALA"], dtype=object),
        residue_numbers=np.array([1], dtype=np.int64),
        chain_ids=np.array(["A"], dtype=object),
    )


@pytest.fixture
def fake_repo(tmp_path: Path) -> Path:
    (tmp_path / "sample_drug.py").write_text("# stub entrypoint\n", encoding="utf-8")
    return tmp_path


def test_mols_from_payload_smiles_list():
    raw = [{"smiles": "CCO"}, {"SMILES": "c1ccccc1"}]
    mols = mols_from_pocket2mol_payload(raw, sanitize=True)
    assert len(mols) == 2
    assert all(isinstance(m, Chem.Mol) for m in mols)


def test_mols_from_payload_dict_finished():
    raw = {"finished": [{"smiles": "CC"}, {"smiles": "invalid%%%"}]}
    mols = mols_from_pocket2mol_payload(raw, sanitize=True)
    assert len(mols) == 1
    assert Chem.MolToSmiles(mols[0]) == "CC"


def test_mol_from_atom_arrays_ethane_like():
    z = np.array([6, 6], dtype=np.int64)
    pos = np.array([[0.0, 0.0, 0.0], [1.54, 0.0, 0.0]], dtype=np.float64)
    mol = mol_from_atom_arrays(z, pos)
    assert mol is not None
    assert mol.GetNumAtoms() == 2


@patch("sbdd_robust.models.pocket2mol_adapter.subprocess.run")
def test_pocket2mol_adapter_mock_subprocess_writes_pt(mock_run, fake_repo: Path, tmp_path: Path):
    captured: dict = {}

    def _side_effect(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        rp = cmd.index("--result_path") + 1
        out_pt = Path(cmd[rp])
        torch.save(
            [
                {"smiles": "CCO"},
                {"pos": np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]]), "element": np.array([6, 6])},
            ],
            out_pt,
        )
        return CompletedProcess(cmd, 0, stdout="", stderr="")

    mock_run.side_effect = _side_effect

    adapter = Pocket2MolAdapter(
        repo_root=fake_repo,
        checkpoint=fake_repo / "dummy.ckpt",
        python_exe="python",
    )
    mols = adapter.generate(_tiny_pocket(), n_samples=2, workdir=tmp_path)

    assert len(mols) == 2
    assert all(isinstance(m, Chem.Mol) for m in mols)

    cmd = captured["cmd"]
    assert "--pdb_path" in cmd
    assert "--num_samples" in cmd and "2" in cmd
    assert "--result_path" in cmd
    assert "--checkpoint" in cmd

    env = captured["kwargs"]["env"]
    assert "PYTHONPATH" in env
    assert str(fake_repo.resolve()) in env["PYTHONPATH"]
    assert captured["kwargs"]["cwd"] == str(fake_repo.resolve())
