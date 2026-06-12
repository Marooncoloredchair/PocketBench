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
from sbdd_robust.perturbations.invariant import (
    anchor_offset as anchor_offset_mod,
    atom_shuffle,
    coordinate_jitter,
    crop_radius,
    directional_crop,
)
from sbdd_robust.perturbations.meaningful import residue_mutation as residue_mut


# Normalized chemistry metrics live on a comparable [0,1] scale, so a single std
# threshold is meaningful for all of them. Raw counts (n_total/n_valid/...) and
# unnormalized SA (mean_sa/std_sa) trivially exceed a [0,1]-tuned threshold and
# dominated the historical "brittleness rate 1.0" headline; the normalized view
# isolates instability in the chemistry the model is actually asked to optimize.
NORMALIZED_METRICS: List[str] = ["validity", "uniqueness", "mean_qed", "std_qed"]


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
        if typ == "face_peel" or typ == "directional_crop":
            ax = spec.get("axis", "pca")
            ax = ax if isinstance(ax, str) else list(ax)
            p = directional_crop.face_peel(
                base,
                fraction=float(spec.get("fraction", 0.25)),
                axis=ax,
                direction=str(spec.get("direction", "plus")),
                tag=tag,
            )
            p.metadata["perturbation_tag"] = tag
            out.append(p)
            continue
        if typ == "anchor_offset":
            ax = spec.get("direction", "pca")
            ax = ax if isinstance(ax, str) else list(ax)
            p = anchor_offset_mod.anchor_offset(
                base,
                offset_angstrom=float(spec.get("offset_angstrom", spec.get("offset", 2.0))),
                direction=ax,
                bbox_scale=float(spec.get("bbox_scale", 1.0)),
                tag=tag,
            )
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
    normalized_only = bool(getattr(args, "normalized_only", False) or cfg.get("normalized_only", False))
    normalized_metrics = list(cfg.get("normalized_metrics", NORMALIZED_METRICS))
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

    def _flush_csv(best_effort: bool = False) -> None:
        if not rows:
            return
        # Atomic rewrite (temp + os.replace) so a disconnect mid-write can't corrupt
        # the checkpoint the next session resumes from.
        try:
            tmp = metrics_csv.with_name(metrics_csv.name + ".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, metrics_csv)
        except OSError as e:
            # Per-condition checkpoints can hit transient Google Drive FUSE write errors.
            # Don't let that bubble into the pocket-level handler (which, with
            # skip_failed=True, would discard an otherwise-good pocket). The row stays
            # in memory and is re-checkpointed on the next condition / final flush.
            if best_effort:
                print(
                    f"[sbdd_robust] WARN: checkpoint write failed ({e}); "
                    "keeping rows in memory, will retry next flush.",
                    file=sys.stderr,
                )
                return
            raise

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
                # checkpoint after every condition so progress survives a disconnect
                _flush_csv(best_effort=True)
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

    # Normalized brittleness view: same threshold restricted to comparable [0,1]
    # chemistry metrics. Emitted alongside the raw all-column flag so reports can
    # separate true chemistry instability from raw-count/SA-scale artefacts.
    flagged_norm_path = None
    br_stats_norm = None
    if normalized_only:
        present_norm = [m for m in normalized_metrics if m in df.columns]
        flagged_norm = rob_mod.flag_invariant_brittleness(
            df,
            invariant_tags=invariant_tags,
            original_tag="original",
            metric_std_threshold=threshold,
            metrics_subset=present_norm,
        )
        flagged_norm_path = results_dir / f"metrics_flagged_normalized__run{run_id}.csv"
        flagged_norm.to_csv(flagged_norm_path, index=False)
        br_stats_norm = br_rate_mod.brittleness_rate_from_flagged(flagged_norm)

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
    if br_stats_norm is not None:
        meta["normalized_only"] = True
        meta["normalized_metrics"] = [m for m in normalized_metrics if m in df.columns]
        meta["flagged_normalized_csv"] = str(flagged_norm_path)
        meta["brittleness_rate_normalized"] = br_stats_norm["brittleness_rate"]
        meta["brittleness_counts_normalized"] = {
            "brittle_pocket_model_pairs": br_stats_norm["brittle_pairs"],
            "total_pocket_model_pairs": br_stats_norm["total_pairs"],
        }
    (results_dir / f"run_meta__{run_id}.json").write_text(json.dumps(meta, indent=2))

    print(f"Wrote {metrics_csv}")
    print(f"Wrote {flagged_path}")
    print(f"Wrote {summary_path}")
    print(
        f"brittleness_rate={br_stats['brittleness_rate']:.4f} "
        f"({br_stats['brittle_pairs']}/{br_stats['total_pairs']} pocket–model pairs brittle "
        "on invariants, ALL metric columns)"
    )
    if br_stats_norm is not None:
        print(f"Wrote {flagged_norm_path}")
        print(
            f"brittleness_rate_normalized={br_stats_norm['brittleness_rate']:.4f} "
            f"({br_stats_norm['brittle_pairs']}/{br_stats_norm['total_pairs']} pairs brittle on "
            f"normalized metrics only: {', '.join(meta['normalized_metrics'])})"
        )
    if figure_paths:
        print("Figures:")
        for fp in figure_paths:
            print(f"  {fp}")
    return 0


def _load_metrics_csv(path: str, model: str | None) -> "pd.DataFrame":
    from sbdd_robust import report as report_mod

    df = pd.read_csv(path)
    if "pocket_id" not in df.columns:
        raise ValueError(
            f"{path} is missing required column 'pocket_id'. Expected a per-condition metrics "
            "CSV (pocket_id, perturbation_tag, validity, uniqueness, mean_qed, std_qed, ...)."
        )
    return report_mod.filter_model(df, model)


def cmd_pbsi(args: argparse.Namespace) -> int:
    from sbdd_robust import report as report_mod

    df = _load_metrics_csv(args.metrics, args.model)
    name = args.dataset or Path(args.metrics).stem
    pp, summary = report_mod.pocket_boundary_sensitivity(df, dataset=name)
    if pp.empty:
        raise SystemExit(
            "No crop-radius conditions found. Need 'original' plus crop_radius_plus_/minus_ rows."
        )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pp.to_csv(out_dir / "pocket_boundary_sensitivity.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "pocket_boundary_sensitivity_summary.csv", index=False)
    print(f"Pocket Boundary Sensitivity Index (PBSI) -- {name}")
    print(f"  PBSI (median |dQED per A|)      : {summary['PBSI_median_abs_slope_qed_per_A']}")
    print(f"  shrinking lowers QED in         : {summary['frac_shrinking_lowers_qed']:.0%} of pockets")
    print(f"  median boundary/noise SNR (1.5A): {summary['median_snr_1.5A']}x")
    print(f"  boundary swing > noise floor in : {summary['frac_boundary_exceeds_noise']:.0%} of pockets")
    print(f"  wrote {out_dir / 'pocket_boundary_sensitivity.csv'}")
    return 0


def cmd_crop_test(args: argparse.Namespace) -> int:
    from sbdd_robust import report as report_mod

    df = _load_metrics_csv(args.metrics, args.model)
    name = args.dataset or Path(args.metrics).stem
    crop_tag = report_mod.resolve_crop_tag(df, args.crop_tag)
    if crop_tag is None:
        raise SystemExit(f"No crop_radius_minus_* conditions found in {args.metrics}.")
    if crop_tag != args.crop_tag:
        print(f"[note] '{args.crop_tag}' absent; using '{crop_tag}'.")
    res = report_mod.crop_paired_wilcoxon(df, crop_tag=crop_tag, dataset=name)
    if res.empty:
        raise SystemExit(f"No paired rows for '{crop_tag}' vs 'original' in {args.metrics}.")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(res.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


def cmd_brittleness(args: argparse.Namespace) -> int:
    from sbdd_robust import report as report_mod

    df = _load_metrics_csv(args.metrics, args.model)
    name = args.dataset or Path(args.metrics).stem
    norm = report_mod.normalized_brittleness(df, taus=args.taus, dataset=name)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(args.out, index=False)
    print("Normalized brittleness (validity, uniqueness, mean_qed, std_qed):")
    print(norm.to_string(index=False))
    if args.raw:
        all_metrics = [c for c in df.columns if c not in (
            "pocket_id", "perturbation_type", "perturbation_tag", "model_name", "run_id",
            "brittle_invariant", "brittleness_note") and pd.api.types.is_numeric_dtype(df[c])]
        raw = report_mod.normalized_brittleness(df, taus=args.taus, metrics=all_metrics, dataset=name)
        print("\nRaw all-column brittleness (cautionary; raw counts/SA inflate the flag):")
        print(raw.to_string(index=False))
    print(f"\nWrote {args.out}")
    return 0


def cmd_isr(args: argparse.Namespace) -> int:
    from sbdd_robust import report as report_mod

    df = _load_metrics_csv(args.metrics, args.model)
    name = args.dataset or Path(args.metrics).stem
    row = report_mod.initialization_sensitivity(
        df, metric=args.metric, dataset=name, model=args.model
    )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(args.out, index=False)
    print(f"Initialization Sensitivity Ratio (ISR) -- {name} [{row['model']}]")
    print(f"  metric                      : {row['metric']}")
    print(f"  frame median |delta|        : {row['frame_median_abs_delta']} "
          f"(n={row['n_frame_pairs']}; tags: {row['frame_tags'] or '(none)'})")
    print(f"  featurization median |delta|: {row['featurization_median_abs_delta']} "
          f"(n={row['n_featurization_pairs']}; tags: {row['featurization_tags'] or '(none)'})")
    print(f"  ISR                         : {row['ISR']} "
          f"[95% CI {row['ISR_ci95_lo']}, {row['ISR_ci95_hi']}]")
    print("  (ISR >> 1: frame-sensitive like autoregressive Pocket2Mol; ~1: diffusion-like)")
    print(f"\nWrote {args.out}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from sbdd_robust import report as report_mod

    df = _load_metrics_csv(args.metrics, args.model)
    name = args.dataset or Path(args.metrics).stem
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = report_mod.normalized_brittleness(df, dataset=name)
    norm.to_csv(out_dir / "normalized_brittleness.csv", index=False)
    crop_tag = report_mod.resolve_crop_tag(df, args.crop_tag) or args.crop_tag
    crop = report_mod.crop_paired_wilcoxon(df, crop_tag=crop_tag, dataset=name)
    crop.to_csv(out_dir / "crop_radius_wilcoxon.csv", index=False)
    pp, summary = report_mod.pocket_boundary_sensitivity(df, dataset=name)
    pp.to_csv(out_dir / "pocket_boundary_sensitivity.csv", index=False)
    pd.DataFrame([summary]).to_csv(out_dir / "pocket_boundary_sensitivity_summary.csv", index=False)
    isr = report_mod.initialization_sensitivity(df, dataset=name, model=args.model)
    pd.DataFrame([isr]).to_csv(out_dir / "initialization_sensitivity.csv", index=False)

    lines: list[str] = [f"# PocketBench reliability report -- {name}", ""]
    b10 = norm[abs(norm["tau"] - 0.10) < 1e-9]
    if not b10.empty:
        r = b10.iloc[0]
        lines += [f"- **Normalized brittleness @ tau=0.10:** {r['brittleness_rate']:.3f} "
                  f"({int(r['n_brittle'])}/{int(r['total'])} pockets)"]
    if not crop.empty:
        for _, r in crop.iterrows():
            sig = "significant" if r["p_value"] < 0.05 else "n.s."
            lines += [f"- **Crop {r['metric']} ({r['comparison']}):** median delta={r['median_delta']:+.3f}, "
                      f"p={r['p_value']:.2e} ({sig}), rank-biserial r={r['rank_biserial_r']:+.3f}, n={int(r['n_pairs'])}"]
    if summary.get("n_pockets"):
        lines += [f"- **PBSI:** {summary['PBSI_median_abs_slope_qed_per_A']} QED/A (median |slope|); "
                  f"boundary swing exceeds featurization noise in {summary.get('frac_boundary_exceeds_noise', float('nan')):.0%} of pockets"]
    if not (isinstance(isr["ISR"], float) and pd.isna(isr["ISR"])):
        lines += [f"- **ISR:** {isr['ISR']} [95% CI {isr['ISR_ci95_lo']}, {isr['ISR_ci95_hi']}] "
                  f"(frame |delta| {isr['frame_median_abs_delta']} vs featurization "
                  f"{isr['featurization_median_abs_delta']}; >>1 = autoregressive first-atom signature)"]
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote report + CSVs to {out_dir}")
    return 0


def main(argv: List[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="pocketbench")
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
    run_p.add_argument(
        "--normalized-only",
        dest="normalized_only",
        action="store_true",
        help=(
            "Additionally emit a brittleness flag computed on normalized chemistry metrics only "
            "(validity, uniqueness, mean_qed, std_qed), written alongside the raw all-column flag. "
            "Separates true chemistry instability from raw-count / SA-scale artefacts."
        ),
    )
    run_p.set_defaults(func=cmd_run)

    # ---- Analysis subcommands (operate on a per-condition metrics CSV; no GPU) ----
    pbsi_p = sub.add_parser(
        "pbsi",
        help="Pocket Boundary Sensitivity Index from a metrics CSV (QED-vs-crop-radius slope).",
    )
    pbsi_p.add_argument("--metrics", required=True, help="Per-condition metrics CSV.")
    pbsi_p.add_argument("--model", default=None, help="Filter to this model_name (optional).")
    pbsi_p.add_argument("--dataset", default=None, help="Label for output rows (default: CSV stem).")
    pbsi_p.add_argument("--out-dir", default="pocketbench_out", help="Output directory.")
    pbsi_p.set_defaults(func=cmd_pbsi)

    crop_p = sub.add_parser(
        "crop-test",
        help="Paired Wilcoxon test of a crop-radius condition vs original (ΔQED, ΔSA).",
    )
    crop_p.add_argument("--metrics", required=True, help="Per-condition metrics CSV.")
    crop_p.add_argument("--model", default=None, help="Filter to this model_name (optional).")
    crop_p.add_argument("--crop-tag", default="crop_radius_minus_1.5", help="Crop condition tag.")
    crop_p.add_argument("--dataset", default=None, help="Label for output rows (default: CSV stem).")
    crop_p.add_argument("--out", default="pocketbench_out/crop_radius_wilcoxon.csv", help="Output CSV.")
    crop_p.set_defaults(func=cmd_crop_test)

    brit_p = sub.add_parser(
        "brittleness",
        help="Normalized brittleness rate vs threshold from a metrics CSV.",
    )
    brit_p.add_argument("--metrics", required=True, help="Per-condition metrics CSV.")
    brit_p.add_argument("--model", default=None, help="Filter to this model_name (optional).")
    brit_p.add_argument("--dataset", default=None, help="Label for output rows (default: CSV stem).")
    brit_p.add_argument("--taus", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20])
    brit_p.add_argument("--raw", action="store_true", help="Also show the cautionary all-column flag.")
    brit_p.add_argument("--out", default="pocketbench_out/normalized_brittleness.csv", help="Output CSV.")
    brit_p.set_defaults(func=cmd_brittleness)

    isr_p = sub.add_parser(
        "isr",
        help="Initialization Sensitivity Ratio: frame-moving vs featurization |delta| from a CSV.",
    )
    isr_p.add_argument("--metrics", required=True, help="Per-condition metrics CSV.")
    isr_p.add_argument("--model", default=None, help="Filter to this model_name (optional).")
    isr_p.add_argument("--metric", default="mean_qed", help="Metric to contrast (default mean_qed).")
    isr_p.add_argument("--dataset", default=None, help="Label for output rows (default: CSV stem).")
    isr_p.add_argument("--out", default="pocketbench_out/initialization_sensitivity.csv", help="Output CSV.")
    isr_p.set_defaults(func=cmd_isr)

    rep_p = sub.add_parser(
        "report",
        help="Run brittleness + crop-test + PBSI + ISR and write a paper-ready folder (no GPU).",
    )
    rep_p.add_argument("--metrics", required=True, help="Per-condition metrics CSV.")
    rep_p.add_argument("--model", default=None, help="Filter to this model_name (optional).")
    rep_p.add_argument("--crop-tag", default="crop_radius_minus_1.5", help="Crop condition tag.")
    rep_p.add_argument("--dataset", default=None, help="Label for output rows (default: CSV stem).")
    rep_p.add_argument("--out-dir", default="pocketbench_out", help="Output directory.")
    rep_p.set_defaults(func=cmd_report)

    args = parser.parse_args(argv)
    code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
