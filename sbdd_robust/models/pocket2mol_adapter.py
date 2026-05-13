"""Run Pocket2Mol ``sample_drug.py`` on a pocket PDB; parse ``.pt`` molecules for metrics."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator, List, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from sbdd_robust.datasets.pdb_io import write_pocket_pdb
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.models.base_adapter import BaseSBDDAdapter

_SMILES_KEYS = ("smiles", "SMILES", "smi", "SMILES_string")
_POS_KEYS = ("pos", "positions", "coords", "coordinates")
_ELEM_KEYS = ("element", "elements", "atomic_num", "atomic_nums", "z", "atom_type", "atomic_numbers")


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _coerce_atomic_nums(elements: Any) -> Optional[np.ndarray]:
    """Return (N,) int atomic numbers, or ``None`` if not interpretable."""
    arr = _to_numpy(elements)
    if arr.dtype.kind in ("U", "S", "O"):
        flat = np.asarray(arr).ravel()
        if flat.size == 0:
            return np.zeros((0,), dtype=np.int64)
        first = flat[0]
        if isinstance(first, (str, bytes)):
            pt = Chem.GetPeriodicTable()
            out = []
            for s in flat:
                sym = str(s).strip().capitalize() if s is not None else ""
                if not sym:
                    return None
                try:
                    out.append(pt.GetAtomicNumber(sym))
                except Exception:
                    return None
            return np.array(out, dtype=np.int64)
        try:
            return np.array([int(x) for x in flat], dtype=np.int64)
        except (TypeError, ValueError):
            return None
    return np.array(arr, dtype=np.int64).ravel()


def mol_from_atom_arrays(
    atomic_nums: np.ndarray,
    positions: np.ndarray,
) -> Optional[Chem.Mol]:
    """Build an RDKit mol from atomic numbers and 3D positions (uses ``DetermineBonds``)."""
    z = np.asarray(atomic_nums, dtype=np.int64).ravel()
    pos = np.asarray(positions, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 3 or z.shape[0] != pos.shape[0] or z.size == 0:
        return None
    rw = Chem.RWMol()
    for an in z:
        rw.AddAtom(Chem.Atom(int(an)))
    mol = rw.GetMol()
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i in range(mol.GetNumAtoms()):
        x, y, zz = pos[i]
        conf.SetAtomPosition(i, (float(x), float(y), float(zz)))
    mol.AddConformer(conf)
    try:
        rdDetermineBonds.DetermineBonds(mol)
    except Exception:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    return mol


def _dict_smiles(d: dict) -> Optional[str]:
    for k in _SMILES_KEYS:
        if k in d and d[k] is not None:
            s = str(d[k]).strip()
            if s:
                return s
    return None


def _dict_elements_positions(d: dict) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    pos = None
    for k in _POS_KEYS:
        if k in d and d[k] is not None:
            pos = _to_numpy(d[k])
            if pos.ndim == 3 and pos.shape[0] == 1:
                pos = pos[0]
            break
    elems = None
    for k in _ELEM_KEYS:
        if k in d and d[k] is not None:
            elems = _coerce_atomic_nums(d[k])
            break
    return elems, pos


def mol_from_molecule_dict(entry: dict) -> Optional[Chem.Mol]:
    """Parse a single Pocket2Mol result dict into an RDKit mol (SMILES preferred)."""
    smi = _dict_smiles(entry)
    if smi:
        mol = Chem.MolFromSmiles(smi)
        return mol if mol is not None else None
    z, pos = _dict_elements_positions(entry)
    if z is not None and pos is not None:
        return mol_from_atom_arrays(z, pos)
    return None


def _getattr_si(obj: Any, name: str) -> Any:
    return getattr(obj, name, None)


def mol_from_entry(entry: Any, *, sanitize: bool = True) -> Optional[Chem.Mol]:
    """Convert a dict or saved Pocket2Mol object to ``Chem.Mol``."""
    if entry is None:
        return None
    if isinstance(entry, Chem.Mol):
        mol = entry
    elif isinstance(entry, dict):
        mol = mol_from_molecule_dict(entry)
    else:
        rdmol = _getattr_si(entry, "rdmol")
        if isinstance(rdmol, Chem.Mol):
            mol = rdmol
        else:
            smi = _getattr_si(entry, "smiles") or _getattr_si(entry, "SMILES")
            if smi:
                mol = Chem.MolFromSmiles(str(smi).strip())
            else:
                elem = _getattr_si(entry, "element")
                pos = _getattr_si(entry, "pos")
                if elem is not None and pos is not None:
                    z = _coerce_atomic_nums(elem)
                    p = _to_numpy(pos)
                    if z is not None:
                        mol = mol_from_atom_arrays(z, p)
                    else:
                        mol = None
                else:
                    mol = None
    if mol is None:
        return None
    if sanitize:
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
    return mol


def iter_pocket2mol_pt_entries(raw: Any) -> Iterator[Any]:
    """Flatten common Pocket2Mol ``torch.save`` payload shapes to a stream of entries."""
    if raw is None:
        return
    if isinstance(raw, (list, tuple)):
        yield from raw
        return
    if isinstance(raw, dict):
        for key in ("finished", "molecules", "samples", "generated", "results"):
            if key in raw:
                seq = raw[key]
                if isinstance(seq, (list, tuple)):
                    yield from seq
                    return
        return
    finished = getattr(raw, "finished", None)
    if isinstance(finished, (list, tuple)):
        yield from finished


def load_pocket2mol_pt(path: Path) -> Any:
    """Load a Pocket2Mol result tensor file (supports PyTorch 2.6 ``weights_only`` default)."""
    import torch

    path = Path(path)
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def mols_from_smiles_file(path: Path, *, sanitize: bool = True) -> List[Chem.Mol]:
    """Load RDKit mols from one SMILES per line (Pocket2Mol ``SMILES.txt``)."""
    mols: List[Chem.Mol] = []
    text = Path(path).read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        if sanitize:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
        mols.append(mol)
    return mols


def mols_from_pocket2mol_payload(
    raw: Any,
    *,
    sanitize: bool = True,
) -> List[Chem.Mol]:
    """Convert a loaded ``.pt`` payload to a list of valid RDKit molecules."""
    mols: List[Chem.Mol] = []
    for entry in iter_pocket2mol_pt_entries(raw):
        m = mol_from_entry(entry, sanitize=sanitize)
        if m is not None:
            mols.append(m)
    return mols


class Pocket2MolAdapter(BaseSBDDAdapter):
    """
    Wrap Pocket2Mol inference via ``sample_drug.py`` (``arneschneuing/Pocket2Mol`` or
    ``luost26/Pocket2Mol`` upstream).

    Expects a repo layout with ``sample_drug.py`` at ``repo_root`` (or ``script_path``).
    Invokes::

        python sample_drug.py --pdb_path <pdb> --num_samples <n> --result_path <out.pt>

    Optional ``--checkpoint`` is appended when ``checkpoint`` is set. Extra CLI tokens
    can be passed via ``extra_args`` (e.g. ``--device cpu``).
    """

    name = "pocket2mol"

    def __init__(
        self,
        repo_root: Path,
        checkpoint: Optional[Path] = None,
        python_exe: Optional[str] = None,
        script_path: Optional[Path] = None,
        extra_args: Optional[Sequence[str]] = None,
        sanitize: bool = True,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.checkpoint = Path(checkpoint).resolve() if checkpoint is not None else None
        self.python_exe: str = python_exe or sys.executable
        self.script_path = (
            Path(script_path).resolve()
            if script_path is not None
            else (self.repo_root / "sample_drug.py").resolve()
        )
        self.extra_args = list(extra_args) if extra_args else []
        self.sanitize = bool(sanitize)
        if not self.script_path.is_file():
            raise FileNotFoundError(
                f"Pocket2Mol sample script not found at {self.script_path}. "
                "Set model.script_path if your fork names the entrypoint differently."
            )

    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        workdir.mkdir(parents=True, exist_ok=True)
        pdb_path = workdir / f"{pocket.pocket_id}_pocket2mol_in.pdb"
        write_pocket_pdb(pocket, pdb_path)

        out_pt = workdir / f"{pocket.pocket_id}_pocket2mol_out.pt"

        argv: List[str] = [
            str(self.python_exe),
            str(self.script_path),
            "--pdb_path",
            str(pdb_path.resolve()),
            "--num_samples",
            str(int(n_samples)),
            "--result_path",
            str(out_pt.resolve()),
        ]
        if self.checkpoint is not None:
            argv.extend(["--checkpoint", str(self.checkpoint)])
        argv.extend(self.extra_args)

        env = os.environ.copy()
        prepend = str(self.repo_root)
        env["PYTHONPATH"] = (
            prepend + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else prepend
        )

        proc = subprocess.run(
            argv,
            cwd=str(self.repo_root),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Pocket2Mol failed (exit {proc.returncode}). stderr tail:\n"
                f"{proc.stderr[-4000:] if proc.stderr else '(empty)'}"
            )
        if not out_pt.is_file():
            raise FileNotFoundError(f"Expected Pocket2Mol output missing: {out_pt}")

        sidecar = out_pt.with_name(f"{out_pt.stem}_smiles.txt")
        if sidecar.is_file():
            return mols_from_smiles_file(sidecar, sanitize=self.sanitize)

        raw = load_pocket2mol_pt(out_pt)
        return mols_from_pocket2mol_payload(raw, sanitize=self.sanitize)
