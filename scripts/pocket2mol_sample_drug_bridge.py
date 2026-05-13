#!/usr/bin/env python3
"""
Bridge Pocket2Mol ``sample_for_pdb.py`` to the ``sbdd-robust`` adapter CLI
(``--pdb_path``, ``--num_samples``, ``--result_path``, ``--checkpoint``).

The public ``pengxingang/Pocket2Mol`` fork does not ship ``sample_drug.py``;
it uses ``sample_for_pdb.py`` with a YAML config, pocket center, and bbox size.
This script derives center and bbox from the input PDB (pocket-only or full),
runs sampling, then copies ``samples_all.pt`` to ``--result_path``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml


def _default_pocket2mol_root() -> Path:
    """Prefer env / Linux Colab layout; ignore Windows ``D:/...`` in POCKET2MOL_ROOT on POSIX."""
    raw = (os.environ.get("POCKET2MOL_ROOT") or "").strip()
    if raw and sys.platform != "win32" and len(raw) > 1 and raw[1] == ":":
        raw = ""
    if raw:
        return Path(raw).expanduser().resolve()
    for cand in (Path("/content/Pocket2Mol"), Path.cwd()):
        if (cand / "sample_for_pdb.py").is_file():
            return cand.resolve()
    return Path("/content/Pocket2Mol")


def _coerce_pocket2mol_root(explicit: Path | None) -> Path:
    """Resolve bridge repo root; on POSIX, drop explicit Windows ``X:/...`` paths (broken under pathlib)."""
    if explicit is not None:
        s = str(explicit)
        if not (sys.platform != "win32" and len(s) > 1 and s[1] == ":"):
            return explicit.expanduser().resolve()
    return _default_pocket2mol_root()


def _pocket_center_and_bbox(
    pdb_path: Path,
    margin: float = 3.0,
    min_box: float = 23.0,
    max_box: float = 34.0,
) -> tuple[list[float], float]:
    """Centroid and cubic bbox edge from protein ATOM records; cap box for full-assembly PDBs."""
    xs, ys, zs = [], [], []
    with open(pdb_path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            # Pocket-only PDBs from sbdd-robust use ATOM for all heavy atoms; full RCSB files
            # mix chains — restrict to standard ATOM rows to avoid mapping the entire crystal.
            if not line.startswith("ATOM  "):
                continue
            if len(line) < 54:
                continue
            elem = line[76:78].strip().upper() if len(line) >= 78 else ""
            if not elem and len(line) >= 16:
                elem = line[12:14].strip().upper()[:1]
            if elem in ("H", "D", "", "Q"):
                continue
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
            except ValueError:
                continue
            xs.append(x)
            ys.append(y)
            zs.append(z)
    if not xs:
        raise RuntimeError(
            f"No protein ATOM heavy atoms in {pdb_path}; use a pocket-only PDB or a file with ATOM records."
        )
    c = np.array([np.mean(xs), np.mean(ys), np.mean(zs)])
    pts = np.stack([xs, ys, zs], axis=1)
    extent = float(np.max(np.abs(pts - c)))
    bbox = float(max(min_box, min(max_box, 2.0 * extent + 2.0 * margin)))
    return c.tolist(), bbox


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb_path", type=Path, required=True)
    ap.add_argument("--num_samples", type=int, required=True)
    ap.add_argument("--result_path", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument(
        "--pocket2mol_root",
        type=Path,
        default=None,
        help="Clone of pengxingang/Pocket2Mol (used as cwd for sample_for_pdb.py).",
    )
    ap.add_argument(
        "--base_config",
        type=Path,
        default=None,
        help="Optional YAML base (default: <pocket2mol_root>/configs/sample_for_pdb.yml).",
    )
    args, extra = ap.parse_known_args()

    root = _coerce_pocket2mol_root(args.pocket2mol_root)
    sample_py = root / "sample_for_pdb.py"
    if not sample_py.is_file():
        raise FileNotFoundError(f"Expected {sample_py} — set --pocket2mol_root or POCKET2MOL_ROOT.")

    base_cfg = args.base_config or (root / "configs" / "sample_for_pdb.yml")
    with open(base_cfg, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    ckpt = args.checkpoint.resolve()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint missing: {ckpt}\nDownload pretrained weights per Pocket2Mol ckpt/README.md "
            "(e.g. pretrained_Pocket2Mol.pt into the ckpt/ folder)."
        )
    cfg.setdefault("model", {})
    cfg["model"]["checkpoint"] = str(ckpt)
    cfg.setdefault("sample", {})
    cfg["sample"]["num_samples"] = int(args.num_samples)

    center, bbox_size = _pocket_center_and_bbox(Path(args.pdb_path))

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        cfg_path = td_path / "bridge_sample.yml"
        with open(cfg_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        out_root = td_path / "p2m_out"
        out_root.mkdir(parents=True, exist_ok=True)

        center_s = ",".join(str(round(x, 4)) for x in center)
        cmd = [
            sys.executable,
            str(sample_py),
            "--pdb_path",
            str(Path(args.pdb_path).resolve()),
            "--center",
            center_s,
            "--bbox_size",
            str(bbox_size),
            "--config",
            str(cfg_path),
            "--device",
            args.device,
            "--outdir",
            str(out_root),
        ]
        cmd.extend(extra)

        env = os.environ.copy()
        prepend = str(root)
        env["PYTHONPATH"] = prepend + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else prepend

        proc = subprocess.run(cmd, cwd=str(root), env=env, capture_output=True, text=True, check=False)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "")[-6000:]
            raise RuntimeError(f"sample_for_pdb.py failed ({proc.returncode}):\n{tail}")

        cands = list(out_root.rglob("samples_all.pt"))
        if not cands:
            raise FileNotFoundError(
                f"No samples_all.pt under {out_root}. stdout/stderr tail:\n"
                f"{(proc.stdout or '')[-2000:]}\n{(proc.stderr or '')[-2000:]}"
            )
        best = max(cands, key=lambda p: p.stat().st_mtime)
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, args.result_path)

        # Sidecar for sbdd-robust driver (Python 3.11) so it can parse molecules without
        # importing Pocket2Mol/torch unpickling deps for ``samples_all.pt`` payloads.
        sm_path = best.parent / "SMILES.txt"
        if not sm_path.is_file():
            raise FileNotFoundError(
                f"Pocket2Mol did not write {sm_path} next to packaged results; "
                "refusing to copy only samples_all.pt (driver cannot unpickle without Pocket2Mol on PYTHONPATH)."
            )
        sidecar = args.result_path.with_name(f"{args.result_path.stem}_smiles.txt")
        shutil.copyfile(sm_path, sidecar)


if __name__ == "__main__":
    main()
