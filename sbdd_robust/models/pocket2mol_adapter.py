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
from sbdd_robust.datasets.pdb_merge import merge_pocket_into_full_pdb
from sbdd_robust.models.pocket2mol_ghosts import ghost_keys_for_frame_perturbation
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


def _ligand_centered_bbox(
    pocket: Pocket,
    *,
    margin: float = 3.0,
    min_box: float = 23.0,
    max_box: float = 34.0,
) -> float:
    """Cubic bbox edge for Pocket2Mol from pocket atoms around the ligand centroid."""
    if pocket.ligand_centroid is not None:
        center = np.asarray(pocket.ligand_centroid, dtype=np.float64)
    else:
        center = pocket.coords.mean(axis=0)
    extent = float(np.max(np.linalg.norm(pocket.coords - center, axis=1)))
    return float(max(min_box, min(max_box, 2.0 * extent + 2.0 * margin)))


def _should_mask_bbox_ghosts(pocket: Pocket) -> bool:
    """Drop full-PDB atoms inside the Pocket2Mol box that crop / face-peel removed."""
    tag = str(pocket.metadata.get("perturbation_tag", "original"))
    if tag in ("original", "atom_shuffle", "coordinate_jitter"):
        return False
    if tag.startswith("anchor_offset"):
        return False
    return tag.startswith("crop_radius") or tag.startswith("face_peel")


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
        full_pdb: Optional[Path] = None,
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
        self.full_pdb = Path(full_pdb).resolve() if full_pdb is not None else None
        if not self.script_path.is_file():
            raise FileNotFoundError(
                f"Pocket2Mol sample script not found at {self.script_path}. "
                "Set model.script_path if your fork names the entrypoint differently."
            )

    def _resolve_pdb_path(
        self,
        pocket: Pocket,
        workdir: Path,
        *,
        mask_bbox: bool,
    ) -> tuple[Path, Optional[Path], float]:
        bbox_size = _ligand_centered_bbox(pocket)
        merged_tmp: Optional[Path] = None
        if self.full_pdb is not None:
            ghosts = ghost_keys_for_frame_perturbation(pocket) if mask_bbox else None
            merged_tmp = merge_pocket_into_full_pdb(
                pocket,
                self.full_pdb,
                bbox_size=bbox_size if mask_bbox else None,
                ghost_atom_keys=ghosts,
            )
            pdb_path = merged_tmp
        else:
            pdb_path = workdir / f"{pocket.pocket_id}_pocket2mol_in.pdb"
            write_pocket_pdb(pocket, pdb_path)
        return pdb_path, merged_tmp, bbox_size

    def _run_subprocess(
        self,
        pocket: Pocket,
        pdb_path: Path,
        out_pt: Path,
        n_samples: int,
        bbox_size: float,
    ) -> List[Chem.Mol]:
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

        if pocket.ligand_centroid is not None:
            lc = pocket.ligand_centroid
            center_s = ",".join(str(float(v)) for v in lc)
            argv.extend([f"--center={center_s}", "--bbox_size", str(bbox_size)])

        co = pocket.metadata.get("center_offset")
        if co is not None:
            if isinstance(co, str):
                offset_s = co.strip()
            else:
                if hasattr(co, "tolist"):
                    co = co.tolist()
                offset_s = ",".join(str(float(v)) for v in co)
            argv.append(f"--center_offset={offset_s}")
        bscale = pocket.metadata.get("bbox_scale")
        if bscale is not None and float(bscale) != 1.0:
            argv.extend(["--bbox_scale", str(float(bscale))])

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

    def generate(self, pocket: Pocket, n_samples: int, workdir: Path) -> List[Chem.Mol]:
        workdir.mkdir(parents=True, exist_ok=True)
        out_pt = workdir / f"{pocket.pocket_id}_pocket2mol_out.pt"
        mask_bbox = self.full_pdb is not None and _should_mask_bbox_ghosts(pocket)
        # Frame perturbations must not fall back to unmasked full PDB (ghost atoms).
        attempts = [True] if mask_bbox else [False]

        for use_mask in attempts:
            merged_tmp: Optional[Path] = None
            try:
                pdb_path, merged_tmp, bbox_size = self._resolve_pdb_path(
                    pocket, workdir, mask_bbox=use_mask
                )
                mols = self._run_subprocess(pocket, pdb_path, out_pt, n_samples, bbox_size)
                if mols or not use_mask or not mask_bbox:
                    return mols
            finally:
                if merged_tmp is not None:
                    try:
                        merged_tmp.unlink(missing_ok=True)
                    except OSError:
                        pass
        return []
