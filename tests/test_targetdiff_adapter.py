from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rdkit import Chem

from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.targetdiff_adapter import TargetDiffAdapter


def _dummy_pocket() -> Pocket:
    import numpy as np

    return Pocket(
        pocket_id="1ABC",
        coords=np.zeros((3, 3), dtype=np.float64),
        elements=np.array(["C", "C", "N"], dtype=object),
        atom_names=np.array(["CA", "CA", "N"], dtype=object),
        residue_names=np.array(["ALA", "ALA", "ALA"], dtype=object),
        residue_numbers=np.array([1, 1, 2], dtype=int),
        chain_ids=np.array(["A", "A", "A"], dtype=object),
        ligand_centroid=np.zeros(3, dtype=np.float64),
    )


def test_targetdiff_adapter_loads_sdf(tmp_path: Path) -> None:
    repo = tmp_path / "td"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "sample_for_pocket.py").write_text("# stub\n", encoding="utf-8")

    cfg = tmp_path / "sampling.yml"
    cfg.write_text("model:\n  checkpoint: dummy.pt\n", encoding="utf-8")

    def fake_run(*args, **kwargs):
        return MagicMock(returncode=0, stdout="", stderr="")

    ad = TargetDiffAdapter(repo_root=repo, config_yaml=cfg, python_exe=sys.executable)
    pock = _dummy_pocket()
    wdir = tmp_path / "work"
    sdf_dir = wdir / "targetdiff_out" / "sdf"
    sdf_dir.mkdir(parents=True)
    mol = Chem.MolFromSmiles("CCO")
    w = Chem.SDWriter(str(sdf_dir / "000.sdf"))
    w.write(mol)
    w.close()

    with patch("subprocess.run", fake_run):
        mols = ad.generate(pock, n_samples=5, workdir=wdir)
    assert len(mols) >= 1


def test_targetdiff_adapter_falls_back_to_sample_pt(tmp_path: Path) -> None:
    repo = tmp_path / "td"
    scripts = repo / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "sample_for_pocket.py").write_text("# stub\n", encoding="utf-8")

    cfg = tmp_path / "sampling.yml"
    cfg.write_text("model:\n  checkpoint: dummy.pt\n", encoding="utf-8")

    ad = TargetDiffAdapter(repo_root=repo, config_yaml=cfg, python_exe=sys.executable)
    pock = _dummy_pocket()
    wdir = tmp_path / "work"
    out = wdir / "targetdiff_out"
    out.mkdir(parents=True)

    import torch

    pos = torch.tensor([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.2, 0.0]])
    v = torch.tensor([0, 0, 0])
    torch.save({"pred_ligand_pos": [pos], "pred_ligand_v": [v]}, out / "sample.pt")

    td_root = Path(__file__).resolve().parents[2].parent / "targetdiff"
    if not (td_root / "utils" / "transforms.py").is_file():
        pytest.skip("targetdiff clone not present for sample.pt fallback test")

    with patch("subprocess.run", MagicMock(returncode=0, stdout="", stderr="")):
        mols = ad.generate(pock, n_samples=1, workdir=wdir)
    assert len(mols) == 1
