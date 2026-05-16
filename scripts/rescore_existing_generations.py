#!/usr/bin/env python3
"""
Re-score existing generation SDFs with AutoDock Vina and update metrics CSV (no DiffSBDD rerun).

Expected layout (from ``cmd_run``): ``{generations}/run_{run_id}/{pocket_id}/{perturbation_tag}/{pocket_id}_gen.sdf``
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rdkit import Chem

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbdd_robust.cli import _apply_perturbations, _resolve  # noqa: PLC2701
from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.metrics import docking as dock_mod

_DEFAULT_VINA_EXE = r"D:\obsfu\tools\vina\vina.exe"


def _sdf_path_for_condition(
    gen_root: Path, pocket_id: str, perturbation_tag: str
) -> Path:
    """
    Primary path used by the driver: ``{pid}/{tag}/{pid}_gen.sdf``.

    Optional flat fallback: ``{pid}_{tag}_gen.sdf`` under ``gen_root``.
    """
    stem = Path(pocket_id).name
    nested = gen_root / stem / str(perturbation_tag) / f"{stem}_gen.sdf"
    if nested.is_file():
        return nested
    flat = gen_root / f"{stem}_{perturbation_tag}_gen.sdf"
    return flat


def _pocket_for_tag(
    base_pocket: Any,
    perturbation_tag: str,
    pert_specs: List[dict[str, Any]],
    pocket_cfg: dict[str, Any],
) -> Any:
    pert_pockets = _apply_perturbations(base_pocket, pert_specs, pocket_cfg=pocket_cfg)
    for p in pert_pockets:
        if str(p.metadata.get("perturbation_tag", "")) == str(perturbation_tag):
            return p
    raise KeyError(
        f"No pocket with perturbation_tag={perturbation_tag!r} "
        f"(have {[p.metadata.get('perturbation_tag') for p in pert_pockets]})"
    )


def _load_sdf_mols(path: Path) -> list[Chem.Mol | None]:
    suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    return [m for m in suppl]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs/experiments/diffsbdd_meaningful_real20.yaml",
        help="YAML with pockets, perturbations, paths, run_id, extraction_radius, vina_cpu",
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Override metrics CSV (default: {paths.results}/metrics_per_condition__run{run_id}.csv)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="List SDF paths only; do not dock or modify CSV",
    )
    args = ap.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = OmegaConf.load(cfg_path)
    root = Path(cfg.get("project_root", cfg_path.parent.parent)).resolve()
    run_id = str(cfg.get("run_id", ""))
    if not run_id:
        print("Config missing run_id", file=sys.stderr)
        return 2

    results_dir = _resolve(root, cfg.paths.results)
    gen_root = _resolve(root, cfg.paths.generations) / f"run_{run_id}"

    metrics_csv = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv
        else (results_dir / f"metrics_per_condition__run{run_id}.csv")
    )
    backup_csv = results_dir / f"metrics_per_condition__run{run_id}_previna.csv"

    radius = float(cfg.get("extraction_radius", 8.0))
    n_cpu = int(cfg.get("vina_cpu", cfg.get("docking_n_cpu", 4)))
    pert_specs = [OmegaConf.to_container(p, resolve=True) for p in cfg.perturbations]
    pockets_cfg: list[Any] = list(cfg.pockets)

    tasks: list[tuple[str, str, Path]] = []
    missing: list[tuple[str, str, Path]] = []
    for pc in pockets_cfg:
        pid = str(pc.id)
        pocket_cfg = OmegaConf.to_container(pc, resolve=True)  # type: ignore[arg-type]
        for spec in pert_specs:
            tag = str(spec.get("tag", "unknown"))
            sdf = _sdf_path_for_condition(gen_root, pid, tag)
            tasks.append((pid, tag, sdf))
            if not sdf.is_file():
                missing.append((pid, tag, sdf))

    print(f"run_id={run_id!r} gen_root={gen_root}", flush=True)
    print(
        f"Configured conditions: {len(tasks)}; SDFs found: {len(tasks) - len(missing)}",
        flush=True,
    )
    if missing:
        print("MISSING SDF:", file=sys.stderr)
        for pid, tag, p in missing:
            print(f"  {pid!r} {tag!r} -> {p}", file=sys.stderr)
        return 1

    for src_i, (pid, tag, sdf) in enumerate(tasks, start=1):
        print(
            f"[{src_i}/{len(tasks)}] would rescore pocket_id={pid!r} tag={tag!r} sdf={sdf}",
            flush=True,
        )

    if args.dry_run:
        print("--dry-run: done (no docking, no CSV changes).")
        return 0

    vina_exe = os.environ.get("VINA_EXE", "").strip()
    if not vina_exe:
        os.environ["VINA_EXE"] = _DEFAULT_VINA_EXE
        print(f"VINA_EXE unset; using fallback {_DEFAULT_VINA_EXE!r}", flush=True)
    elif not Path(vina_exe).is_file():
        print(
            f"Warning: VINA_EXE={vina_exe!r} is not a file; "
            f"docking.resolve may still find another binary.",
            file=sys.stderr,
        )

    if not metrics_csv.is_file():
        print(f"Metrics CSV not found: {metrics_csv}", file=sys.stderr)
        return 1

    shutil.copy2(metrics_csv, backup_csv)
    print(f"Wrote backup {backup_csv}", flush=True)

    df = pd.read_csv(metrics_csv)
    if "perturbation_tag" not in df.columns:
        print("CSV missing perturbation_tag column", file=sys.stderr)
        return 1

    for pc in pockets_cfg:
        pid = str(pc.id)
        pdb = _resolve(root, pc.pdb)
        lc = pc.get("ligand_chain")
        lr = pc.get("ligand_resseq")
        sdf_path_opt = pc.get("sdf")
        sdf_path = _resolve(root, sdf_path_opt) if sdf_path_opt else None
        pocket_cfg = OmegaConf.to_container(pc, resolve=True)  # type: ignore[arg-type]

        base_pocket = load_pocket_from_complex(
            pdb_path=pdb,
            sdf_path=sdf_path,
            radius=radius,
            pocket_id=pid,
            metadata={"full_pdb": str(pdb)},
            ligand_chain=str(lc) if lc is not None else None,
            ligand_resseq=int(lr) if lr is not None else None,
        )

        for spec in pert_specs:
            tag = str(spec.get("tag", "unknown"))
            sdf = _sdf_path_for_condition(gen_root, pid, tag)
            pock = _pocket_for_tag(base_pocket, tag, pert_specs, pocket_cfg)
            mols = _load_sdf_mols(sdf)
            scores = dock_mod.score_molecules(
                mols, pock, n_cpus=n_cpu, verbose=False
            )
            finite = [s for s in scores if s is not None and np.isfinite(s)]
            mean_v = float(np.mean(finite)) if finite else float("nan")
            std_v = float(np.std(finite)) if len(finite) > 1 else float("nan")

            mask = (df["pocket_id"].astype(str) == pid) & (
                df["perturbation_tag"].astype(str) == tag
            )
            if not mask.any():
                print(
                    f"No CSV row for pocket_id={pid!r} perturbation_tag={tag!r}",
                    file=sys.stderr,
                )
                continue
            df.loc[mask, "vina_score_mean"] = mean_v
            df.loc[mask, "vina_score_std"] = std_v
            print(
                f"Updated {pid} {tag}: mean={mean_v} std={std_v} (n_finite={len(finite)}/{len(scores)})",
                flush=True,
            )

    df.to_csv(metrics_csv, index=False, na_rep="")
    print(f"Wrote {metrics_csv}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
