"""Single entrypoint: ``sbdd-robust run --config ...``."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def _register_cfg_resolvers() -> None:
    """Allow ``${env:VAR}`` in YAML configs (used by experiments/diffsbdd_real47.yaml)."""

    def _env_resolver(key: str) -> str:
        val = os.environ.get(key)
        if val is None or val == "":
            raise ValueError(
                f"Environment variable {key!r} is not set but is required by the config "
                f"(${{env:{key}}}). For DiffSBDD 47-pocket reruns export DIFFSBDD_REPO, "
                "DIFFSBDD_CHECKPOINT, and DIFFSBDD_PYTHON (see configs/experiments/README.md)."
            )
        return val

    # replace=True: safe to call on repeated imports / reloads
    OmegaConf.register_new_resolver("env", _env_resolver, replace=True)


_register_cfg_resolvers()

from sbdd_robust.analysis import figures as fig_mod
from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust import provenance as provenance_mod
from sbdd_robust.metrics import brittleness_rate as br_rate_mod
from sbdd_robust.metrics import chemistry as chem_mod
from sbdd_robust.metrics import robustness_score as rob_mod
from sbdd_robust.models.base_adapter import BaseSBDDAdapter
from sbdd_robust.models.diffsbdd_adapter import DiffSBDDAdapter
from sbdd_robust.models.mock_adapter import MockSBDDAdapter
from sbdd_robust.models.pocket2mol_adapter import Pocket2MolAdapter
from sbdd_robust.models.targetdiff_adapter import TargetDiffAdapter
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
        extras = mcfg.get("extra_args")
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
            extra_args=list(extras) if extras is not None else None,
        )
    if typ == "pocket2mol":
        ckpt = mcfg.get("checkpoint")
        sp = mcfg.get("script_path")
        extras = mcfg.get("extra_args")
        return Pocket2MolAdapter(
            repo_root=_resolve(root, mcfg.repo_root),
            checkpoint=_resolve(root, ckpt) if ckpt else None,
            python_exe=mcfg.get("python_exe"),
            script_path=_resolve(root, sp) if sp else None,
            extra_args=list(extras) if extras is not None else None,
            sanitize=bool(mcfg.get("sanitize", True)),
        )
    if typ == "targetdiff":
        cfg_yml = mcfg.get("config_yaml") or mcfg.get("config")
        if not cfg_yml:
            raise ValueError("model.config_yaml (path to TargetDiff sampling YAML) is required for targetdiff")
        sp = mcfg.get("script_path")
        extras = mcfg.get("extra_args")
        return TargetDiffAdapter(
            repo_root=_resolve(root, mcfg.repo_root),
            config_yaml=_resolve(root, cfg_yml),
            python_exe=mcfg.get("python_exe"),
            script_path=_resolve(root, sp) if sp else None,
            device=str(mcfg.get("device", "cuda:0")),
            batch_size=int(mcfg.get("batch_size", 100)),
            sanitize=bool(mcfg.get("sanitize", True)),
            extra_args=list(extras) if extras is not None else None,
        )
    raise ValueError(f"Unknown model.type: {typ}")


def _apply_perturbations(
    base: Pocket,
    specs: List[Dict[str, Any]],
    pocket_cfg: Dict[str, Any] | None = None,
) -> List[Pocket]:
    pc = pocket_cfg or {}
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
            if spec.get("residue_id_from_pocket"):
                rid = str(pc.get("mutation_residue_id", "")).strip()
                if not rid:
                    raise ValueError(
                        "residue_id_from_pocket requires pocket entry field mutation_residue_id"
                    )
            else:
                rid = str(spec["residue_id"])
            taa = str(spec["target_aa"])
            ptag = spec.get("tag")
            p = residue_mut.mutate_residue(
                base, rid, taa, perturbation_tag=str(ptag) if ptag else None
            )
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
    # A stable run_id is required for resume to match a prior partial run; a bare
    # timestamp default would never line up across sessions (e.g. Colab disconnects).
    run_id = str(getattr(args, "run_id", None) or cfg.get("run_id") or int(time.time()))
    resume = bool(getattr(args, "resume", False) or cfg.get("resume", False))
    metrics_csv = results_dir / f"metrics_per_condition__run{run_id}.csv"

    pockets_cfg: List[Any] = list(cfg.pockets)
    pert_specs = [OmegaConf.to_container(p, resolve=True) for p in cfg.perturbations]
    pert_tags = [str(s.get("tag", s.get("type", "unknown"))) for s in pert_specs]
    invariant_tags = list(cfg.get("invariant_tags", ["atom_shuffle", "coordinate_jitter"]))
    threshold = float(cfg.get("brittleness_std_threshold", 0.1))
    radius = float(cfg.get("extraction_radius", 8.0))
    n_samples = int(cfg.model.get("n_samples", 10))
    skip_failed = bool(cfg.get("skip_failed_pockets", False))

    rows: list[dict[str, Any]] = []
    done: set[tuple[str, str]] = set()
    if resume and metrics_csv.is_file():
        try:
            if metrics_csv.stat().st_size == 0:
                raise ValueError("checkpoint file is empty")
            prev = pd.read_csv(metrics_csv)
            if prev.empty:
                raise ValueError("checkpoint CSV has no rows")
            rows = prev.to_dict("records")
            done = {
                (str(r.get("pocket_id")), str(r.get("perturbation_tag")))
                for r in rows
            }
            print(
                f"[sbdd_robust] resume: loaded {len(rows)} rows / {len(done)} completed "
                f"conditions from {metrics_csv}",
                file=sys.stderr,
            )
        except Exception as e:  # corrupt/partial CSV: start clean rather than crash
            print(
                f"[sbdd_robust] resume: could not read {metrics_csv} ({e}); starting fresh",
                file=sys.stderr,
            )
            rows, done = [], set()
            try:
                metrics_csv.unlink(missing_ok=True)
            except OSError:
                pass

    def _flush_csv() -> None:
        if not rows:
            return
        # Atomic rewrite (temp + os.replace) so a disconnect mid-write can't corrupt
        # the checkpoint the next session resumes from.
        tmp = metrics_csv.with_name(metrics_csv.name + ".tmp")
        pd.DataFrame(rows).to_csv(tmp, index=False)
        os.replace(tmp, metrics_csv)

    for pc in pockets_cfg:
        pid = str(pc.id)
        if resume and pert_tags and all((pid, t) in done for t in pert_tags):
            print(
                f"[sbdd_robust] resume: skip pocket {pid} (all {len(pert_tags)} conditions done)",
                file=sys.stderr,
            )
            continue
        try:
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
            pocket_cfg_dict: Dict[str, Any] = OmegaConf.to_container(pc, resolve=True)  # type: ignore[assignment]
            pert_pockets = _apply_perturbations(base_pocket, pert_specs, pocket_cfg=pocket_cfg_dict)
            gen_root = _resolve(root, cfg.paths.generations) / f"run_{run_id}"
            compute_dock = bool(cfg.get("compute_docking", False))
            for pock in pert_pockets:
                ptag = str(pock.metadata.get("perturbation_tag", "unknown"))
                if resume and (pid, ptag) in done:
                    print(
                        f"[sbdd_robust] resume: skip {pid}/{ptag} (already done)",
                        file=sys.stderr,
                    )
                    continue
                wdir = gen_root / pid / ptag
                wdir.mkdir(parents=True, exist_ok=True)
                mols = adapter.generate(pock, n_samples=n_samples, workdir=wdir)
                summary = chem_mod.summarize_molecules(mols)
                if compute_dock and pock.ligand_centroid is not None:
                    from sbdd_robust.metrics import docking as dock_mod

                    n_cpu = int(cfg.get("vina_cpu", cfg.get("docking_n_cpu", 4)))
                    scores = dock_mod.score_molecules(
                        mols,
                        pock,
                        n_cpus=n_cpu,
                        verbose=bool(cfg.get("docking_verbose", False)),
                    )
                    finite = [s for s in scores if s is not None and np.isfinite(s)]
                    summary["vina_score_mean"] = float(np.mean(finite)) if finite else float("nan")
                    summary["vina_score_std"] = float(np.std(finite)) if len(finite) > 1 else float("nan")
                row = {
                    "pocket_id": pid,
                    "perturbation_type": pock.metadata.get("perturbation_type", ""),
                    "perturbation_tag": ptag,
                    "model_name": adapter.name,
                    "run_id": run_id,
                    **summary,
                }
                rows.append(row)
                done.add((pid, ptag))
                _flush_csv()  # checkpoint after every condition so progress survives a disconnect
        except Exception as e:
            if not skip_failed:
                raise
            print(f"[sbdd_robust] SKIP pocket {pid}: {e}", file=sys.stderr)

    _flush_csv()
    if not rows:
        raise RuntimeError(
            f"No benchmark rows were written to {metrics_csv}. "
            "All pockets may have failed (see SKIP lines above), or the run was interrupted "
            "before the first condition finished. Re-run with skip_failed_pockets: false "
            "to surface the first pocket error."
        )
    # Re-read for consistent dtypes (resumed rows came back as strings/NaN from CSV).
    df = pd.read_csv(metrics_csv)

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

    mtype = str(cfg.model.get("type", "mock")).lower()
    diffsbdd_repo = None
    pocket2mol_repo = None
    targetdiff_repo = None
    if mtype == "diffsbdd" and cfg.model.get("repo_root"):
        diffsbdd_repo = _resolve(root, cfg.model.repo_root)
    if mtype == "pocket2mol" and cfg.model.get("repo_root"):
        pocket2mol_repo = _resolve(root, cfg.model.repo_root)
    if mtype == "targetdiff" and cfg.model.get("repo_root"):
        targetdiff_repo = _resolve(root, cfg.model.repo_root)

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
        "git_provenance": provenance_mod.collect_git_provenance(
            sbdd_repo_root,
            diffsbdd_repo=diffsbdd_repo,
            pocket2mol_repo=pocket2mol_repo,
            targetdiff_repo=targetdiff_repo,
        ),
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
    run_p.add_argument(
        "--run-id",
        dest="run_id",
        type=str,
        default=None,
        help="Stable run id (overrides config). Required for resume to match across sessions.",
    )
    run_p.add_argument(
        "--resume",
        action="store_true",
        help="Skip (pocket, perturbation) conditions already present in the run's metrics CSV.",
    )
    run_p.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
