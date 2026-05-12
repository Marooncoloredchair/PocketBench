"""Single entrypoint: ``sbdd-robust run --config ...``."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from sbdd_robust.analysis import figures as fig_mod
from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust import provenance as provenance_mod
from sbdd_robust.metrics import brittleness_rate as br_rate_mod
from sbdd_robust.metrics import chemistry as chem_mod
from sbdd_robust.metrics import robustness_score as rob_mod
from sbdd_robust.models.base_adapter import BaseSBDDAdapter
from sbdd_robust.models.diffsbdd_adapter import DiffSBDDAdapter
from sbdd_robust.models.mock_adapter import MockSBDDAdapter
from sbdd_robust.perturbations.invariant import atom_shuffle, coordinate_jitter, crop_radius
from sbdd_robust.perturbations.meaningful import residue_mutation as residue_mut


def _resolve(root: Path, p: str | Path) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path
    return (root / path).resolve()


def _build_adapter(cfg: Any, pocket_cfg: Any, root: Path) -> BaseSBDDAdapter:
    mcfg = cfg.model
    typ = str(mcfg.get("type", "mock")).lower()
    if typ == "mock":
        return MockSBDDAdapter()
    if typ == "diffsbdd":
        full_pdb = getattr(pocket_cfg, "full_pdb", None) or pocket_cfg.pdb
        ref = pocket_cfg.get("ref_ligand") or mcfg.get("ref_ligand")
        ref = str(ref).strip() if ref else None
        return DiffSBDDAdapter(
            repo_root=_resolve(root, mcfg.repo_root),
            checkpoint=_resolve(root, mcfg.checkpoint),
            ref_ligand=ref,
            full_pdb=_resolve(root, full_pdb),
            python_exe=mcfg.get("python_exe"),
            sanitize=bool(mcfg.get("sanitize", False)),
            resamplings=int(mcfg.get("resamplings", 10)),
            jump_length=int(mcfg.get("jump_length", 1)),
            timesteps=mcfg.get("timesteps"),
            relax=bool(mcfg.get("relax", False)),
            batch_size=mcfg.get("batch_size"),
            all_frags=bool(mcfg.get("all_frags", False)),
            num_nodes_lig=mcfg.get("num_nodes_lig"),
        )
    raise ValueError(f"Unknown model.type: {typ}")


def _apply_perturbations(base: Pocket, specs: List[Dict[str, Any]]) -> List[Pocket]:
    out: List[Pocket] = []
    for spec in specs:
        tag = str(spec.get("tag", spec.get("type", "unknown")))
        typ = str(spec.get("type", "identity")).lower()
        if typ == "identity" or typ == "original":
            p = base.copy()
            p.metadata = dict(base.metadata)
            p.metadata["perturbation_type"] = "original"
            p.metadata["perturbation_tag"] = "original"
            out.append(p)
            continue
        if typ == "atom_shuffle":
            rng = np.random.default_rng(int(spec.get("seed", 0)))
            p = atom_shuffle.shuffle_atom_order(base, rng=rng)
            p.metadata["perturbation_tag"] = tag
            out.append(p)
            continue
        if typ == "coordinate_jitter":
            rng = np.random.default_rng(int(spec.get("seed", 1)))
            sigma = float(spec.get("sigma", 0.05))
            p = coordinate_jitter.jitter_coordinates(base, sigma=sigma, rng=rng)
            p.metadata["perturbation_tag"] = tag
            out.append(p)
            continue
        if typ == "crop_radius_plus":
            d = float(spec.get("delta_angstrom", spec.get("delta", 1.5)))
            p = crop_radius.crop_radius_plus(base, delta_angstrom=d)
            p.metadata["perturbation_tag"] = tag
            out.append(p)
            continue
        if typ == "crop_radius_minus":
            d = float(spec.get("delta_angstrom", spec.get("delta", 1.5)))
            p = crop_radius.crop_radius_minus(base, delta_angstrom=d)
            p.metadata["perturbation_tag"] = tag
            out.append(p)
            continue
        if typ == "residue_mutation":
            rid = str(spec["residue_id"])
            taa = str(spec["target_aa"])
            p = residue_mut.mutate_residue(base, rid, taa)
            out.append(p)
            continue
        raise ValueError(f"Unknown perturbation type: {typ}")
    return out


def cmd_run(args: argparse.Namespace) -> int:
    cfg_path = Path(args.config).resolve()
    cfg = OmegaConf.load(cfg_path)
    root = Path(cfg.get("project_root", cfg_path.parent.parent)).resolve()
    sbdd_repo_root = Path(__file__).resolve().parent.parent
    np.random.seed(int(cfg.get("seed", 0)))

    results_dir = _resolve(root, cfg.paths.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    run_id = str(cfg.get("run_id") or int(time.time()))
    metrics_csv = results_dir / f"metrics_per_condition__run{run_id}.csv"

    pockets_cfg: List[Any] = list(cfg.pockets)
    pert_specs = [OmegaConf.to_container(p, resolve=True) for p in cfg.perturbations]
    invariant_tags = list(cfg.get("invariant_tags", ["atom_shuffle", "coordinate_jitter"]))
    threshold = float(cfg.get("brittleness_std_threshold", 0.1))
    radius = float(cfg.get("extraction_radius", 8.0))
    n_samples = int(cfg.model.get("n_samples", 10))

    rows: list[dict[str, Any]] = []

    for pc in pockets_cfg:
        pid = str(pc.id)
        pdb = _resolve(root, pc.pdb)
        sdf = pc.get("sdf")
        sdf_path = _resolve(root, sdf) if sdf else None
        lc = pc.get("ligand_chain")
        lr = pc.get("ligand_resseq")
        base_pocket = load_pocket_from_complex(
            pdb_path=pdb,
            sdf_path=sdf_path,
            radius=radius,
            pocket_id=pid,
            metadata={"full_pdb": str(pdb)},
            ligand_chain=str(lc) if lc is not None else None,
            ligand_resseq=int(lr) if lr is not None else None,
        )
        adapter = _build_adapter(cfg, pc, root)
        pert_pockets = _apply_perturbations(base_pocket, pert_specs)
        gen_root = _resolve(root, cfg.paths.generations) / f"run_{run_id}"
        for pock in pert_pockets:
            wdir = gen_root / pid / str(pock.metadata.get("perturbation_tag", "unknown"))
            wdir.mkdir(parents=True, exist_ok=True)
            mols = adapter.generate(pock, n_samples=n_samples, workdir=wdir)
            summary = chem_mod.summarize_molecules(mols)
            row = {
                "pocket_id": pid,
                "perturbation_type": pock.metadata.get("perturbation_type", ""),
                "perturbation_tag": pock.metadata.get("perturbation_tag", ""),
                "model_name": adapter.name,
                "run_id": run_id,
                **summary,
            }
            rows.append(row)
        if isinstance(adapter, DiffSBDDAdapter):
            pass

    df = pd.DataFrame(rows)
    df.to_csv(metrics_csv, index=False)

    flagged = rob_mod.flag_invariant_brittleness(
        df,
        invariant_tags=invariant_tags,
        original_tag="original",
        metric_std_threshold=threshold,
    )
    flagged_path = results_dir / f"metrics_flagged__run{run_id}.csv"
    flagged.to_csv(flagged_path, index=False)

    br_stats = br_rate_mod.brittleness_rate_from_flagged(flagged)

    summary = rob_mod.summarize_robustness_vs_original(
        df,
        invariant_tags=invariant_tags,
        original_tag="original",
    )
    summary_path = results_dir / f"robustness_summary__run{run_id}.csv"
    summary.to_csv(summary_path, index=False)

    figure_paths: list[str] = []
    plot_names = []
    for m in ("validity", "uniqueness", "mean_qed", "mean_sa"):
        if f"{m}_original" in summary.columns and f"{m}_inv_mean" in summary.columns:
            plot_names.append(m)
    if plot_names:
        fig_dir = results_dir / "figures"
        written = fig_mod.plot_all_metrics(summary, plot_names, fig_dir, prefix=f"run{run_id}")
        figure_paths = [str(p) for p in written]

    diffsbdd_repo = None
    if str(cfg.model.get("type", "mock")).lower() == "diffsbdd" and cfg.model.get("repo_root"):
        diffsbdd_repo = _resolve(root, cfg.model.repo_root)

    meta = {
        "run_id": run_id,
        "metrics_csv": str(metrics_csv),
        "flagged_csv": str(flagged_path),
        "summary_csv": str(summary_path),
        "figure_paths": figure_paths,
        "n_rows": int(df.shape[0]),
        "brittleness_rate": br_stats["brittleness_rate"],
        "brittleness_counts": {
            "brittle_pocket_model_pairs": br_stats["brittle_pairs"],
            "total_pocket_model_pairs": br_stats["total_pairs"],
        },
        "git_provenance": provenance_mod.collect_git_provenance(sbdd_repo_root, diffsbdd_repo),
    }
    (results_dir / f"run_meta__{run_id}.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {flagged_path}")
    print(f"Wrote {summary_path}")
    print(
        f"brittleness_rate={br_stats['brittleness_rate']:.4f} "
        f"({br_stats['brittle_pairs']}/{br_stats['total_pairs']} pocket–model pairs brittle on invariants)"
    )
    if figure_paths:
        print("Figures:")
        for fp in figure_paths:
            print(f"  {fp}")
    return 0


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="sbdd-robust")
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="Run benchmark pipeline from YAML config")
    run_p.add_argument("--config", type=str, required=True)
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
