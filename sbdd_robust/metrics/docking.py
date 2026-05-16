"""AutoDock Vina docking scores for benchmark molecules (optional dependency)."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from sbdd_robust.datasets.pocket import Pocket


def _stderr_tail(b: bytes | None, max_len: int = 800) -> str:
    if not b:
        return ""
    try:
        s = b.decode("utf-8", errors="replace").strip()
    except Exception:
        return "<decode error>"
    if len(s) > max_len:
        return s[-max_len:] + "..."
    return s


def _write_pocket_pdb(pocket: Pocket, path: Path) -> None:
    """Write pocket heavy atoms as a minimal ATOM-only PDB."""
    lines: list[str] = []
    serial = 1
    for i in range(pocket.coords.shape[0]):
        x, y, z = (float(pocket.coords[i, j]) for j in range(3))
        elem = str(pocket.elements[i])[:2].strip() or "C"
        resname = str(pocket.residue_names[i])[:3].ljust(3)
        chain = str(pocket.chain_ids[i]).strip()[:1] or "A"
        resseq = int(pocket.residue_numbers[i])
        aname = str(pocket.atom_names[i]).strip()[:4].ljust(4)
        lines.append(
            f"ATOM  {serial:5d} {aname} {resname} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem:>2s}"
        )
        serial += 1
    lines.append("END")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _strip_receptor_pdbqt_for_vina_rigid(path: Path) -> tuple[bool, str]:
    """
    AutoDock Vina 1.2 ``--receptor`` expects rigid PDBQT without ROOT/BRANCH trees.

    Meeko and Open Babel often emit AutoDock-style flex trees even for pocket-only
    inputs; Vina then errors: "Unknown or inappropriate tag ... > ROOT".
    Keep only ATOM/HETATM records.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        return False, f"read failed: {e}"
    skip_prefixes = ("ROOT", "ENDROOT", "BRANCH", "ENDBRANCH", "TORSDOF")
    out_lines: list[str] = []
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(skip_prefixes):
            continue
        if line.startswith("ATOM") or line.startswith("HETATM"):
            out_lines.append(line.rstrip())
    if not out_lines:
        return False, "no ATOM/HETATM lines after strip"
    path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return True, ""


def _prepare_receptor_pdbqt(receptor_pdb: Path, out_pdbqt: Path) -> tuple[bool, str]:
    """Try meeko CLI module, mk_prepare_receptor on PATH, Open Babel, prepare_receptor4.py."""
    failures: list[str] = []
    # Note: meeko uses -r / --reactive_flexres for *reactive flexible residues*, not the input PDB.
    # Input is --read_pdb (RDKit) or -i / --read_with_prody. Wrong -r produced ROOT/BRANCH PDBQT
    # that Vina 1.2 rejects as a rigid receptor.
    for argv in (
        [
            sys.executable,
            "-m",
            "meeko.mk_prepare_receptor",
            "--read_pdb",
            str(receptor_pdb),
            "--write_pdbqt",
            str(out_pdbqt),
        ],
    ):
        try:
            subprocess.run(argv, check=True, capture_output=True, timeout=120)
            if out_pdbqt.is_file():
                ok_s, rs = _strip_receptor_pdbqt_for_vina_rigid(out_pdbqt)
                if ok_s:
                    return True, ""
                failures.append(f"meeko.mk_prepare_receptor: rigid strip failed: {rs}")
                try:
                    out_pdbqt.unlink()
                except OSError:
                    pass
        except FileNotFoundError as e:
            failures.append(f"meeko.mk_prepare_receptor: FileNotFoundError: {e}")
            continue
        except subprocess.CalledProcessError as e:
            failures.append(
                "meeko.mk_prepare_receptor: exit "
                f"{e.returncode}; stderr={_stderr_tail(e.stderr)}"
            )
            continue
        except subprocess.TimeoutExpired:
            failures.append("meeko.mk_prepare_receptor: timeout (120s)")
            continue
    mk = shutil.which("mk_prepare_receptor")
    if mk:
        try:
            subprocess.run(
                [mk, "--read_pdb", str(receptor_pdb), "--write_pdbqt", str(out_pdbqt)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            if out_pdbqt.is_file():
                ok_s, rs = _strip_receptor_pdbqt_for_vina_rigid(out_pdbqt)
                if ok_s:
                    return True, ""
                failures.append(f"mk_prepare_receptor: rigid strip failed: {rs}")
                try:
                    out_pdbqt.unlink()
                except OSError:
                    pass
        except subprocess.CalledProcessError as e:
            failures.append(
                f"mk_prepare_receptor: exit {e.returncode}; stderr={_stderr_tail(e.stderr)}"
            )
        except subprocess.TimeoutExpired:
            failures.append("mk_prepare_receptor: timeout (120s)")
    else:
        failures.append("mk_prepare_receptor: not on PATH")
    obabel = shutil.which("obabel")
    if obabel:
        try:
            subprocess.run(
                [obabel, str(receptor_pdb), "-O", str(out_pdbqt)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            if out_pdbqt.is_file():
                ok_s, rs = _strip_receptor_pdbqt_for_vina_rigid(out_pdbqt)
                if ok_s:
                    return True, ""
                failures.append(f"obabel: rigid strip failed: {rs}")
                try:
                    out_pdbqt.unlink()
                except OSError:
                    pass
            else:
                failures.append("obabel: completed but output pdbqt missing")
        except subprocess.CalledProcessError as e:
            failures.append(f"obabel: exit {e.returncode}; stderr={_stderr_tail(e.stderr)}")
        except subprocess.TimeoutExpired:
            failures.append("obabel: timeout (120s)")
    else:
        failures.append("obabel: not on PATH")
    prep = shutil.which("prepare_receptor4.py")
    if prep:
        try:
            subprocess.run(
                [prep, "-r", str(receptor_pdb), "-o", str(out_pdbqt)],
                check=True,
                capture_output=True,
                timeout=300,
            )
            if out_pdbqt.is_file():
                ok_s, rs = _strip_receptor_pdbqt_for_vina_rigid(out_pdbqt)
                if ok_s:
                    return True, ""
                failures.append(f"prepare_receptor4.py: rigid strip failed: {rs}")
                try:
                    out_pdbqt.unlink()
                except OSError:
                    pass
            else:
                failures.append("prepare_receptor4.py: completed but output pdbqt missing")
        except subprocess.CalledProcessError as e:
            failures.append(
                f"prepare_receptor4.py: exit {e.returncode}; stderr={_stderr_tail(e.stderr)}"
            )
        except subprocess.TimeoutExpired:
            failures.append("prepare_receptor4.py: timeout (300s)")
    else:
        failures.append("prepare_receptor4.py: not on PATH")
    return False, " | ".join(failures) if failures else "all receptor prep methods failed (unknown)"


def _mol_to_pdbqt(mol: Chem.Mol, out_pdbqt: Path) -> tuple[bool, str]:
    try:
        from meeko import MoleculePreparation  # type: ignore import-not-found
        from meeko import PDBQTWriterLegacy  # type: ignore import-not-found

        preparator = MoleculePreparation()
        prep = preparator.prepare(mol)
        if isinstance(prep, tuple):
            setups = prep[0]
        else:
            setups = prep
        if not setups:
            return False, "meeko: MoleculePreparation returned no setups"
        st0 = setups[0]
        for pdbqt_str, ok, err_msg in PDBQTWriterLegacy.write_string(st0):
            if ok and pdbqt_str:
                out_pdbqt.write_text(pdbqt_str, encoding="utf-8")
                return True, ""
        return False, "meeko: PDBQTWriterLegacy produced no valid string"
    except Exception as e:
        meeko_err = f"meeko: {type(e).__name__}: {e}"
        obabel = shutil.which("obabel")
        if not obabel:
            return False, f"{meeko_err}; obabel fallback: not on PATH"
        tdir = tempfile.mkdtemp(prefix="sbdd_ob_")
        try:
            sdf = Path(tdir) / "lig.sdf"
            w = Chem.SDWriter(str(sdf))
            w.write(mol)
            w.close()
            proc = subprocess.run(
                [obabel, str(sdf), "-O", str(out_pdbqt), "-xh"],
                check=True,
                capture_output=True,
                timeout=60,
            )
            if out_pdbqt.is_file():
                return True, ""
            return (
                False,
                f"{meeko_err}; obabel: output missing after success",
            )
        except subprocess.CalledProcessError as e2:
            return (
                False,
                f"{meeko_err}; obabel: exit {e2.returncode}; stderr={_stderr_tail(e2.stderr)}",
            )
        except subprocess.TimeoutExpired:
            return False, f"{meeko_err}; obabel: timeout (60s)"


def _embed_mol(mol: Chem.Mol) -> tuple[Chem.Mol | None, str]:
    m = Chem.Mol(mol)
    try:
        m = Chem.AddHs(m)
        params = AllChem.ETKDGv3()
        params.randomSeed = 0xC0FFEE
        err = AllChem.EmbedMolecule(m, params)
        if err:
            err = AllChem.EmbedMolecule(m, randomSeed=0xC0FFEE)
        if err:
            return None, f"rdkit_embed: EmbedMolecule failed (code {err})"
        try:
            AllChem.MMFFOptimizeMolecule(m)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(m)
            except Exception:
                pass
        return m, ""
    except Exception as e:
        return None, f"rdkit_embed: {type(e).__name__}: {e}"


def _best_affinity_from_vina_text(log: str) -> float | None:
    """Parse best (mode 1) kcal/mol affinity from Vina stdout/stderr."""
    for line in log.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "1":
            try:
                return float(parts[1])
            except ValueError:
                continue
    m = re.search(
        r"(?m)^\s*1\s+(-?\d+(?:\.\d+)?)\s+",
        log,
    )
    if m:
        return float(m.group(1))
    return None


def _resolve_vina_executable() -> str | None:
    env = os.environ.get("VINA_EXE", "").strip()
    if env and Path(env).is_file():
        return env
    return shutil.which("vina") or shutil.which("vina.exe")


def _score_one_vina_binding(
    receptor_pdbqt: Path,
    ligand_pdbqt: Path,
    center: np.ndarray,
    box: tuple[float, float, float] = (20, 20, 20),
    n_cpus: int = 4,
) -> tuple[float | None, str]:
    try:
        from vina import Vina  # type: ignore import-not-found

        v = Vina(sf_name="vina")
        v.set_receptor(str(receptor_pdbqt))
        v.set_ligand_from_file(str(ligand_pdbqt))
        cx, cy, cz = float(center[0]), float(center[1]), float(center[2])
        v.compute_vina_maps(center=[cx, cy, cz], box_size=list(box), spacing=0.375)
        v.dock(exhaustiveness=max(4, min(16, n_cpus * 2)), n_poses=1)
        arr = v.energies()
        if arr is None or len(arr) == 0:
            return None, "python_vina: energies() empty after dock"
        row = arr[0]
        return float(row[0]) if hasattr(row, "__getitem__") else float(row), ""
    except Exception as e:
        py_err = f"python_vina: {type(e).__name__}: {e}"

    vina_exe = _resolve_vina_executable()
    if not vina_exe:
        hint = f"{py_err}; " if py_err else ""
        return None, f"{hint}vina_cli: no executable (set VINA_EXE or PATH vina.exe)"
    try:
        out_pdbqt = ligand_pdbqt.with_name("dock_out.pdbqt")
        conf_lines = [
            f"receptor = {receptor_pdbqt}",
            f"ligand = {ligand_pdbqt}",
            f"center_x = {float(center[0]):.4f}",
            f"center_y = {float(center[1]):.4f}",
            f"center_z = {float(center[2]):.4f}",
            f"size_x = {box[0]}",
            f"size_y = {box[1]}",
            f"size_z = {box[2]}",
            f"cpu = {int(n_cpus)}",
            f"out = {out_pdbqt}",
            "exhaustiveness = 8",
            "num_modes = 3",
        ]
        conf = ligand_pdbqt.with_name("vina_conf.txt")
        conf.write_text("\n".join(conf_lines) + "\n", encoding="utf-8")
        proc = subprocess.run(
            [vina_exe, "--config", str(conf)],
            check=False,
            capture_output=True,
            timeout=600,
        )
        blob = ""
        if proc.stdout:
            blob += proc.stdout.decode("utf-8", errors="replace")
        if proc.stderr:
            blob += "\n" + proc.stderr.decode("utf-8", errors="replace")
        aff = _best_affinity_from_vina_text(blob)
        if aff is not None:
            return aff, ""
        tail = blob.strip()[-600:] if blob.strip() else "(no vina output captured)"
        return (
            None,
            f"vina_cli: returncode={proc.returncode}; could not parse affinity; tail={tail!r}",
        )
    except FileNotFoundError as e:
        return None, f"vina_cli: FileNotFoundError running {vina_exe!r}: {e}"
    except subprocess.TimeoutExpired:
        return None, "vina_cli: timeout (600s)"
    except Exception as e:
        return None, f"vina_cli: {type(e).__name__}: {e}"


def score_molecules(
    mols: list[Chem.Mol],
    pocket: Pocket,
    n_cpus: int = 4,
    *,
    verbose: bool = False,
) -> list[float | None]:
    """
    Return best Vina score (kcal/mol) per molecule, or ``None`` on failure.

    Receptor: pocket heavy atoms as PDB → PDBQT via meeko / obabel / prepare_receptor4.
    Box: center ``pocket.ligand_centroid``, size 20×20×20 Å.

    Set ``verbose=True`` or environment ``SBDD_DOCKING_VERBOSE=1`` to print per-molecule
    failure reasons to stderr.
    """
    env_verb = os.environ.get("SBDD_DOCKING_VERBOSE", "").lower() in (
        "1",
        "true",
        "yes",
        "debug",
    )
    verbose = bool(verbose or env_verb)

    def vlog(msg: str) -> None:
        if verbose:
            print(f"[docking] {msg}", file=sys.stderr)

    n = len(mols)
    if pocket.ligand_centroid is None:
        vlog("pocket.ligand_centroid is None — skipping all molecules")
        return [None] * n
    center = np.asarray(pocket.ligand_centroid, dtype=np.float64).reshape(3)
    warned = False

    with tempfile.TemporaryDirectory(prefix="sbdd_vina_") as tmp:
        tdir = Path(tmp)
        rec_pdb = tdir / "receptor.pdb"
        rec_pdbqt = tdir / "receptor.pdbqt"
        _write_pocket_pdb(pocket, rec_pdb)
        ok_rec, rec_reason = _prepare_receptor_pdbqt(rec_pdb, rec_pdbqt)
        if not ok_rec:
            vlog(f"receptor PDBQT prep failed: {rec_reason}")
            if not warned:
                warnings.warn(
                    f"Could not prepare receptor PDBQT; docking scores will be None. ({rec_reason})",
                    RuntimeWarning,
                    stacklevel=2,
                )
                warned = True
            return [None] * n

        out: list[float | None] = []
        for i, mol in enumerate(mols):
            if mol is None or mol.GetNumAtoms() == 0:
                vlog(f"molecule[{i}]: skipped (None or empty)")
                out.append(None)
                continue
            emb, er = _embed_mol(mol)
            if emb is None:
                vlog(f"molecule[{i}]: ligand embed failed — {er}")
                out.append(None)
                continue
            lig_pdbqt = tdir / f"lig_{len(out)}.pdbqt"
            ok_lig, lr = _mol_to_pdbqt(emb, lig_pdbqt)
            if not ok_lig:
                vlog(f"molecule[{i}]: ligand PDBQT conversion failed — {lr}")
                out.append(None)
                continue
            score, sr = _score_one_vina_binding(rec_pdbqt, lig_pdbqt, center, n_cpus=n_cpus)
            if score is None:
                vlog(f"molecule[{i}]: Vina scoring failed — {sr}")
            elif verbose:
                vlog(f"molecule[{i}]: score={score} kcal/mol")
            out.append(score)
        return out
