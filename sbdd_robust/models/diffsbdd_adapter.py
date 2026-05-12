"""Run DiffSBDD ``generate_ligands.py`` on a (possibly merged) PDB pocket."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import List, Optional

from rdkit import Chem

from sbdd_robust.datasets.pdb_merge import merge_pocket_into_full_pdb
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.base_adapter import BaseSBDDAdapter

# CLI surface verified against upstream ``generate_ligands.py`` (positional
# ``checkpoint``, then ``--pdbfile``, ``--ref_ligand``, ``--outfile``, etc.):
# https://github.com/arneschneuing/DiffSBDD/blob/main/generate_ligands.py
# Re-verify with: ``python generate_ligands.py --help`` in your clone before runs.


class DiffSBDDAdapter(BaseSBDDAdapter):
    """
    Wraps DiffSBDD inference via subprocess.

    Expects ``repo_root`` containing ``generate_ligands.py`` and a ``checkpoint`` path.
    ``ref_ligand`` uses DiffSBDD syntax, e.g. ``A:330`` (chain:residue number).

    If ``pocket.metadata`` contains ``resi_list`` (list of strings like ``A:12``),
    ``--resi_list`` is forwarded so conditioning matches the extracted pocket residues.
    """

    name = "diffsbdd"

    def __init__(
        self,
        repo_root: Path,
        checkpoint: Path,
        ref_ligand: Optional[str] = None,
        full_pdb: Optional[Path] = None,
        python_exe: Optional[str] = None,
        sanitize: bool = False,
        resamplings: int = 10,
        jump_length: int = 1,
        timesteps: Optional[int] = None,
        relax: bool = False,
        batch_size: Optional[int] = None,
        all_frags: bool = False,
        num_nodes_lig: Optional[int] = None,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint = Path(checkpoint).resolve()
        self.ref_ligand = ref_ligand.strip() if ref_ligand else None
        self.full_pdb = Path(full_pdb).resolve() if full_pdb else None
        self.python_exe = python_exe or sys.executable
        self.sanitize = sanitize
        self.resamplings = int(resamplings)
        self.jump_length = int(jump_length)
        self.timesteps = int(timesteps) if timesteps is not None else None
        self.relax = bool(relax)
        self.batch_size = int(batch_size) if batch_size is not None else None
        self.all_frags = bool(all_frags)
        self.num_nodes_lig = int(num_nodes_lig) if num_nodes_lig is not None else None
        script = self.repo_root / "generate_ligands.py"
        if not script.is_file():
            raise FileNotFoundError(f"DiffSBDD generate_ligands.py not found at {script}")

    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        workdir.mkdir(parents=True, exist_ok=True)
        if self.full_pdb is not None:
            pdb_in = merge_pocket_into_full_pdb(pocket, self.full_pdb)
        else:
            from sbdd_robust.datasets.pdb_io import write_pocket_pdb

            pdb_in = workdir / f"{pocket.pocket_id}_pocket_only.pdb"
            write_pocket_pdb(pocket, pdb_in)

        out_sdf = workdir / f"{pocket.pocket_id}_gen.sdf"
        n = int(n_samples)
        batch = self.batch_size if self.batch_size is not None else n
        if n % batch != 0:
            raise ValueError(
                f"n_samples ({n}) must be divisible by batch_size ({batch}); "
                "DiffSBDD asserts this when batch_size is set."
            )

        argv: List[str] = [
            "generate_ligands.py",
            str(self.checkpoint),
            "--pdbfile",
            str(pdb_in),
            "--outfile",
            str(out_sdf),
            "--n_samples",
            str(n),
            "--resamplings",
            str(self.resamplings),
            "--jump_length",
            str(self.jump_length),
        ]

        rl = pocket.metadata.get("resi_list")
        if rl:
            # DiffSBDD asserts exactly one of ``pocket_ids`` (``--resi_list``) or ``ref_ligand``.
            argv.append("--resi_list")
            argv.extend(str(x) for x in rl)
        else:
            if not self.ref_ligand:
                raise ValueError(
                    "DiffSBDD needs either pocket.metadata['resi_list'] or adapter ref_ligand; "
                    "both are missing."
                )
            argv.extend(["--ref_ligand", self.ref_ligand])

        if self.batch_size is not None:
            argv.extend(["--batch_size", str(batch)])

        if self.num_nodes_lig is not None:
            argv.extend(["--num_nodes_lig", str(self.num_nodes_lig)])

        if self.all_frags:
            argv.append("--all_frags")

        if self.sanitize:
            argv.append("--sanitize")

        if self.relax:
            argv.append("--relax")

        if self.timesteps is not None:
            argv.extend(["--timesteps", str(self.timesteps)])

        argv_path = workdir / "_sbdd_robust_diffsbdd_argv.json"
        argv_path.write_text(json.dumps(argv), encoding="utf-8")
        gl_path = (self.repo_root / "generate_ligands.py").resolve()
        launcher = workdir / "_sbdd_robust_diffsbdd_launcher.py"
        launcher.write_text(
            textwrap.dedent(
                f"""
                import json, runpy, sys
                from pathlib import Path

                _wd = Path(__file__).resolve().parent
                _REPO = {json.dumps(str(self.repo_root))}
                if _REPO not in sys.path:
                    sys.path.insert(0, _REPO)
                sys.argv = json.loads((_wd / "_sbdd_robust_diffsbdd_argv.json").read_text(encoding="utf-8"))
                runpy.run_path({json.dumps(str(gl_path))}, run_name="__main__")
                """
            ).strip()
            + "\n",
            encoding="utf-8",
        )

        cmd: List[str] = [self.python_exe, str(launcher)]

        env = os.environ.copy()
        env["PYTHONPATH"] = str(self.repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            cmd,
            cwd=str(self.repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"DiffSBDD failed (code {proc.returncode}): {proc.stderr[-4000:]}"
            )
        if not out_sdf.is_file():
            raise FileNotFoundError(f"Expected output SDF missing: {out_sdf}")

        suppl = Chem.SDMolSupplier(str(out_sdf), sanitize=False, removeHs=False)
        mols: List[Chem.Mol] = [m for m in suppl if m is not None]
        if self.full_pdb is not None:
            try:
                pdb_in.unlink(missing_ok=True)
            except OSError:
                pass
        return mols
