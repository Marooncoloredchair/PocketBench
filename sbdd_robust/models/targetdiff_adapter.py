"""TargetDiff inference via upstream ``scripts/sample_for_pocket.py`` (guanjq/targetdiff)."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rdkit import Chem

from sbdd_robust.datasets.pdb_io import write_pocket_pdb
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.base_adapter import BaseSBDDAdapter


class TargetDiffAdapter(BaseSBDDAdapter):
    """
    Calls TargetDiff's pocket sampler with a pocket-only PDB.

    Upstream layout (typical clone): ``scripts/sample_for_pocket.py`` with
    ``config``, ``--pdb_path``, ``--result_path``, ``--num_samples`` (see that file's argparse).
    Set ``TARGETDIFF_REPO`` / config ``repo_root`` to your clone root; extend ``PYTHONPATH`` to include it.
    """

    name = "targetdiff"

    def __init__(
        self,
        repo_root: Path,
        config_yaml: Path,
        *,
        python_exe: Optional[str] = None,
        script_path: Optional[Path] = None,
        device: str = "cuda:0",
        batch_size: int = 100,
        sanitize: bool = True,
        extra_args: Optional[List[str]] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.config_yaml = Path(config_yaml).resolve()
        self.python_exe = python_exe or sys.executable
        self.script_path = Path(script_path).resolve() if script_path else self.repo_root / "scripts" / "sample_for_pocket.py"
        self.device = device
        self.batch_size = int(batch_size)
        self.sanitize = bool(sanitize)
        self.extra_args = list(extra_args or [])
        if not self.script_path.is_file():
            raise FileNotFoundError(
                f"TargetDiff script not found at {self.script_path}. "
                "Pass model.script_path or clone https://github.com/guanjq/targetdiff"
            )

    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        workdir.mkdir(parents=True, exist_ok=True)
        pdb_path = workdir / f"{pocket.pocket_id}_pocket.pdb"
        write_pocket_pdb(pocket, pdb_path)
        result_path = workdir / "targetdiff_out"
        if result_path.exists():
            for child in result_path.iterdir():
                if child.is_file():
                    child.unlink()
        result_path.mkdir(parents=True, exist_ok=True)

        cmd: list[str] = [
            str(self.python_exe),
            str(self.script_path),
            str(self.config_yaml),
            "--pdb_path",
            str(pdb_path),
            "--result_path",
            str(result_path),
            "--num_samples",
            str(int(n_samples)),
            "--device",
            str(self.device),
            "--batch_size",
            str(self.batch_size),
        ]
        cmd.extend(self.extra_args)
        env = os.environ.copy()
        # TargetDiff imports ``utils``, ``datasets``, etc. from repo root.
        pp = str(self.repo_root)
        if env.get("PYTHONPATH"):
            env["PYTHONPATH"] = pp + os.pathsep + env["PYTHONPATH"]
        else:
            env["PYTHONPATH"] = pp

        subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            env=env,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        mols: list[Chem.Mol] = []
        sdf_dir = result_path / "sdf"
        if sdf_dir.is_dir():
            for p in sorted(sdf_dir.glob("*.sdf")):
                suppl = Chem.SDMolSupplier(str(p), sanitize=False, removeHs=False)
                for mol in suppl:
                    if mol is None:
                        continue
                    if self.sanitize:
                        try:
                            Chem.SanitizeMol(Chem.Mol(mol))
                        except Exception:
                            continue
                    mols.append(mol)

        pt_path = result_path / "sample.pt"
        if not mols and pt_path.is_file():
            # Optional: user may extend to torch-load coordinates; leave empty list.
            pass

        return mols[: int(n_samples)]

    def cleanup(self, workdir: Path) -> None:
        return
