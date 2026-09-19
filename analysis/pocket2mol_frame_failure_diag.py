#!/usr/bin/env python3
"""Diagnose Pocket2Mol 0-mol frame failures: atom counts + optional subprocess probe."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sbdd_robust.cli import _apply_perturbations  # noqa: E402
from sbdd_robust.datasets.load_complexes import load_pocket_from_complex  # noqa: E402
from sbdd_robust.datasets.pdb_merge import (  # noqa: E402
    count_merged_protein_atoms,
    ghost_atom_keys,
    merge_pocket_into_full_pdb,
)
from sbdd_robust.models.pocket2mol_adapter import (  # noqa: E402
    Pocket2MolAdapter,
    _ligand_centered_bbox,
)
from sbdd_robust.models.pocket2mol_ghosts import (  # noqa: E402
    ghost_keys_for_frame_perturbation,
    reference_pocket_for_masking,
)

PANEL = ["1AO7", "1B0R", "1HXC", "1I4F"]
FRAME_TAGS = ["crop_radius_minus_1.5", "face_peel_0.25"]


def _load_config(path: Path) -> dict:
    with path.open() as f:
        return yaml.safe_load(f)


def _pocket_cfg(cfg: dict, pocket_id: str) -> dict:
    for p in cfg["pockets"]:
        if str(p["id"]).upper() == pocket_id.upper():
            return p
    raise KeyError(pocket_id)


def _perturbation_cfg(cfg: dict, tag: str) -> dict:
    for p in cfg["perturbations"]:
        if p["tag"] == tag:
            return p
    raise KeyError(tag)


def diagnose(
    cfg_path: Path,
    *,
    probe: bool = False,
    n_samples: int = 4,
) -> None:
    cfg = _load_config(cfg_path)
    model_cfg = cfg["model"]
    adapter = Pocket2MolAdapter(
        repo_root=Path(model_cfg["repo_root"]),
        checkpoint=Path(model_cfg["checkpoint"]) if model_cfg.get("checkpoint") else None,
        python_exe=model_cfg.get("python_exe"),
        script_path=Path(model_cfg["script_path"]) if model_cfg.get("script_path") else None,
        extra_args=model_cfg.get("extra_args"),
        sanitize=model_cfg.get("sanitize", True),
        full_pdb=None,
    )

    print(f"Config: {cfg_path}")
    print(f"{'pocket':6s} {'tag':22s} {'p_atoms':>7s} {'ghosts':>6s} "
          f"{'bbox':>5s} {'aggr_in':>7s} {'sel_in':>7s} {'unm_in':>7s}  probe")
    print("-" * 90)

    for pid in PANEL:
        pocket_cfg = _pocket_cfg(cfg, pid)
        full_pdb = Path(pocket_cfg["full_pdb"]).resolve()
        adapter.full_pdb = full_pdb

        for tag in FRAME_TAGS:
            base = load_pocket_from_complex(
                Path(pocket_cfg["pdb"]),
                radius=float(cfg.get("extraction_radius", 8.0)),
                pocket_id=pid,
                ligand_chain=pocket_cfg.get("ligand_chain"),
                ligand_resseq=pocket_cfg.get("ligand_resseq"),
                metadata={"source_pdb": str(pocket_cfg["pdb"]), "full_pdb": pocket_cfg.get("full_pdb")},
            )
            base.metadata["base_extraction_radius"] = float(cfg.get("extraction_radius", 8.0))
            base.metadata["extraction_radius"] = float(cfg.get("extraction_radius", 8.0))
            pert = _apply_perturbations(base, [_perturbation_cfg(cfg, tag)], pocket_cfg)[0]
            bbox = _ligand_centered_bbox(pert)
            center = pert.ligand_centroid if pert.ligand_centroid is not None else pert.coords.mean(axis=0)
            half = bbox / 2.0

            ref = reference_pocket_for_masking(pert)
            ghosts = ghost_keys_for_frame_perturbation(pert)
            n_ghosts = len(ghosts)

            unmerged = merge_pocket_into_full_pdb(pert, full_pdb, bbox_size=None)
            aggressive = merge_pocket_into_full_pdb(pert, full_pdb, bbox_size=bbox)
            selective = merge_pocket_into_full_pdb(
                pert, full_pdb, bbox_size=bbox, ghost_atom_keys=ghosts
            )

            unm = count_merged_protein_atoms(unmerged, center=center, half=half)["in_bbox"]
            aggr = count_merged_protein_atoms(aggressive, center=center, half=half)["in_bbox"]
            sel = count_merged_protein_atoms(selective, center=center, half=half)["in_bbox"]

            probe_n = ""
            if probe:
                with tempfile.TemporaryDirectory() as td:
                    work = Path(td)
                    for label, use_mask in [("sel", True), ("aggr", False)]:
                        if label == "sel":
                            merged = selective
                        else:
                            merged = aggressive
                        out_pt = work / f"{pid}_{tag}_{label}.pt"
                        try:
                            mols = adapter._run_subprocess(pert, merged, out_pt, n_samples, bbox)
                            probe_n += f" {label}={len(mols)}"
                        except Exception as exc:
                            probe_n += f" {label}=ERR({exc.__class__.__name__})"
                    for p in (unmerged, aggressive, selective):
                        try:
                            p.unlink(missing_ok=True)
                        except OSError:
                            pass
            else:
                for p in (unmerged, aggressive, selective):
                    try:
                        p.unlink(missing_ok=True)
                    except OSError:
                        pass

            print(
                f"{pid:6s} {tag:22s} {pert.coords.shape[0]:7d} {n_ghosts:6d} "
                f"{bbox:5.1f} {aggr:7d} {sel:7d} {unm:7d}{probe_n}"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--config",
        default=str(_ROOT / "configs/experiments/pocket2mol_isr_matched4.yaml"),
    )
    ap.add_argument("--probe", action="store_true", help="Run Pocket2Mol on merged PDBs")
    ap.add_argument("--n-samples", type=int, default=4)
    args = ap.parse_args()
    diagnose(Path(args.config), probe=args.probe, n_samples=args.n_samples)


if __name__ == "__main__":
    main()
