#!/usr/bin/env python3
"""Receptor-control arm for Job A Vina analysis (crop vs original).

Problem
-------
Job A docks each condition's molecules against that condition's *pocket-only*
receptor. Under ``crop_radius_minus_1.5`` the receptor loses atoms, so a worse
(more positive) Vina score confounds **model molecule change** with **scoring
geometry change**.

Control
-------
Dock the **original** condition's molecules against the **cropped** receptor
(``original_mols_cropped_receptor``). Decomposition per pocket:

  total ΔVina     = V(crop mols, crop receptor) − V(orig mols, orig receptor)
  receptor effect = V(orig mols, crop receptor) − V(orig mols, orig receptor)
  molecule effect = total − receptor
                  = V(crop mols, crop receptor) − V(orig mols, crop receptor)

Paired Wilcoxon tests are reported on each of the three Δ series.

Caveat (printed and written to the summary): absolute Vina means here are
**pocket-only** receptors (~−2–3 kcal/mol on smoke tests) and are **not**
comparable to whole-protein docking literature. Claims are about **relative**
shifts between conditions.

Inputs (Job A when landed)
--------------------------
  configs/cluster/diffsbdd_real100_vina.yaml
  data/generations/run_{run_id}/{pocket}/original/{pocket}_gen.sdf
  data/results/metrics_per_condition__run{run_id}.csv   # vina_score_mean for orig & crop

Outputs
-------
  paper/vina_receptor_control_per_pocket.csv
  paper/vina_receptor_control_wilcoxon.csv
  paper/vina_receptor_control_summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import OmegaConf
from rdkit import Chem
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbdd_robust.cli import _apply_perturbations, _resolve  # noqa: PLC2701
from sbdd_robust.datasets.load_complexes import load_pocket_from_complex
from sbdd_robust.datasets.pocket import Pocket
from sbdd_robust.metrics import docking as dock_mod
from sbdd_robust.report import _rank_biserial

CROP_TAG = "crop_radius_minus_1.5"
CONTROL_TAG = "original_mols_cropped_receptor"
ABS_SCORE_NOTE = (
    "Absolute Vina scores use pocket-only receptors (Job A / this control), so "
    "means are typically much weaker than whole-protein docking literature "
    "(e.g. smoke ~-2.5 kcal/mol). Interpret only relative shifts between "
    "conditions (dVina), not absolute affinity."
)


def _sdf_path(gen_root: Path, pocket_id: str, tag: str) -> Path:
    stem = Path(pocket_id).name
    nested = gen_root / stem / str(tag) / f"{stem}_gen.sdf"
    if nested.is_file():
        return nested
    return gen_root / f"{stem}_{tag}_gen.sdf"


def _load_mols(path: Path, max_n: int | None = None) -> list[Chem.Mol]:
    if not path.is_file():
        return []
    suppl = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
    out: list[Chem.Mol] = []
    for m in suppl:
        if m is None:
            continue
        out.append(m)
        if max_n is not None and len(out) >= max_n:
            break
    return out


def _mean_vina(scores: list[float | None]) -> float:
    finite = [s for s in scores if s is not None and np.isfinite(s)]
    return float(np.mean(finite)) if finite else float("nan")


def _pocket_for_tag(
    base: Pocket,
    tag: str,
    pert_specs: list[dict[str, Any]],
    pocket_cfg: dict[str, Any],
) -> Pocket:
    for p in _apply_perturbations(base, pert_specs, pocket_cfg=pocket_cfg):
        if str(p.metadata.get("perturbation_tag", "")) == str(tag):
            return p
    raise KeyError(f"No pocket with tag={tag!r}")


def _with_centroid(pocket: Pocket, centroid: np.ndarray) -> Pocket:
    """Copy pocket but force docking box center (keep receptor atoms unchanged)."""
    p = pocket.copy()
    p.ligand_centroid = np.asarray(centroid, dtype=np.float64).reshape(3).copy()
    return p


def _wilcoxon_delta(
    deltas: np.ndarray,
    *,
    label: str,
    comparison: str,
    n_boot: int = 10000,
    seed: int = 20260812,
) -> dict[str, Any]:
    d = np.asarray(deltas, dtype=float)
    d = d[np.isfinite(d)]
    n = int(d.size)
    n_nz = int(np.sum(d != 0))
    rng = np.random.default_rng(seed)
    if n_nz == 0:
        W, p, z, r_z = float("nan"), 1.0, float("nan"), float("nan")
    else:
        # wilcoxon on (x, 0) equivalent to signed-rank on deltas
        res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
        W, p = float(res.statistic), float(res.pvalue)
        ranks = stats.rankdata(np.abs(d[d != 0]))
        r_pos = ranks[d[d != 0] > 0].sum()
        mean_w = n_nz * (n_nz + 1) / 4.0
        std_w = np.sqrt(n_nz * (n_nz + 1) * (2 * n_nz + 1) / 24.0)
        z = (r_pos - mean_w) / std_w if std_w > 0 else float("nan")
        r_z = abs(z) / np.sqrt(n_nz) if n_nz else float("nan")
    rbc = _rank_biserial(d)
    if n_boot and n >= 2:
        idx = rng.integers(0, n, size=(n_boot, n))
        med = np.median(d[idx], axis=1)
        med_lo, med_hi = np.percentile(med, [2.5, 97.5])
    else:
        med_lo = med_hi = float("nan")
    return {
        "component": label,
        "comparison": comparison,
        "n_pairs": n,
        "n_zero_diff": int(np.sum(d == 0)) if n else 0,
        "mean_delta": round(float(np.mean(d)), 4) if n else float("nan"),
        "median_delta": round(float(np.median(d)), 4) if n else float("nan"),
        "wilcoxon_W": round(W, 2) if not np.isnan(W) else np.nan,
        "p_value": p,
        "rank_biserial_r": round(rbc, 4) if not np.isnan(rbc) else np.nan,
        "z": round(float(z), 4) if not np.isnan(z) else np.nan,
        "effect_size_r_z": round(float(r_z), 4) if not np.isnan(r_z) else np.nan,
        "median_ci95_lo": round(float(med_lo), 4),
        "median_ci95_hi": round(float(med_hi), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--config",
        type=Path,
        default=_REPO / "configs/cluster/diffsbdd_real100_vina.yaml",
        help="Job A YAML (pockets, perturbations, paths, run_id).",
    )
    ap.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Job A metrics with vina_score_mean (default: data/results/metrics_per_condition__run{run_id}.csv).",
    )
    ap.add_argument(
        "--generations-root",
        type=Path,
        default=None,
        help="Override generations/run_{id} directory.",
    )
    ap.add_argument("--crop-tag", default=CROP_TAG)
    ap.add_argument("--vina-cpu", type=int, default=None)
    ap.add_argument("--max-mols", type=int, default=None, help="Cap molecules per pocket (smoke).")
    ap.add_argument("--limit-pockets", type=int, default=None, help="Only first N config pockets.")
    ap.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="JSONL checkpoint of control docking means (resume).",
    )
    ap.add_argument("--out-dir", type=Path, default=_REPO / "paper")
    ap.add_argument("--dry-run", action="store_true", help="Validate paths; no docking.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    cfg_path = args.config.resolve()
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path}")
    cfg = OmegaConf.load(cfg_path)
    root = Path(cfg.get("project_root", cfg_path.parent.parent.parent)).resolve()
    # configs/cluster/*.yaml → project_root usually "."; parent.parent.parent of cluster yaml is repo
    if (cfg_path.parent.name == "cluster") and cfg.get("project_root", ".") in (".", None):
        root = cfg_path.parents[2]

    run_id = str(cfg.get("run_id", "diffsbdd_real100_vina"))
    results_dir = _resolve(root, cfg.paths.results)
    gen_root = (
        Path(args.generations_root).resolve()
        if args.generations_root
        else (_resolve(root, cfg.paths.generations) / f"run_{run_id}")
    )
    metrics_csv = (
        Path(args.metrics_csv).resolve()
        if args.metrics_csv
        else (results_dir / f"metrics_per_condition__run{run_id}.csv")
    )
    crop_tag = str(args.crop_tag)
    radius = float(cfg.get("extraction_radius", 8.0))
    n_cpu = int(args.vina_cpu or cfg.get("vina_cpu", cfg.get("docking_n_cpu", 4)))
    pert_specs = [OmegaConf.to_container(p, resolve=True) for p in cfg.perturbations]
    pockets_cfg = list(cfg.pockets)
    if args.limit_pockets is not None:
        pockets_cfg = pockets_cfg[: int(args.limit_pockets)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.checkpoint or (out_dir / f"vina_receptor_control__{run_id}.jsonl")

    print(f"run_id={run_id}")
    print(f"metrics_csv={metrics_csv} exists={metrics_csv.is_file()}")
    print(f"gen_root={gen_root} exists={gen_root.is_dir()}")
    print(f"crop_tag={crop_tag} control={CONTROL_TAG}")
    print(ABS_SCORE_NOTE)
    print()

    if not metrics_csv.is_file():
        msg = (
            f"Job A metrics not found: {metrics_csv}\n"
            "Re-run after the array finishes / merge_array_results.py."
        )
        if args.dry_run:
            print(f"DRY-RUN NOTE: {msg}")
            # Still check how many original SDFs exist under an alternate local run if present
            alt = _resolve(root, cfg.paths.generations) / "run_diffsbdd_real100"
            print(f"(optional local gens check) {alt} exists={alt.is_dir()}")
            return 0
        raise SystemExit(msg)

    metrics = pd.read_csv(metrics_csv)
    if "vina_score_mean" not in metrics.columns:
        raise SystemExit(
            f"{metrics_csv} has no vina_score_mean - Job A must run with compute_docking: true."
        )
    tcol = "perturbation_tag" if "perturbation_tag" in metrics.columns else "perturbation_type"

    # Resume map: pocket_id -> control mean
    done: dict[str, float] = {}
    if ckpt_path.is_file():
        for line in ckpt_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            done[str(rec["pocket_id"])] = float(rec["vina_orig_mols_crop_rec"])
        print(f"checkpoint: {len(done)} pockets from {ckpt_path}")

    plan_ok = 0
    plan_missing_sdf = 0
    for pc in pockets_cfg:
        pid = str(pc.id)
        sdf = _sdf_path(gen_root, pid, "original")
        if sdf.is_file():
            plan_ok += 1
        else:
            plan_missing_sdf += 1
            print(f"MISSING original SDF: {pid} -> {sdf}")
    print(f"pockets with original SDF: {plan_ok}; missing: {plan_missing_sdf}")
    if args.dry_run:
        print("--dry-run: no docking.")
        return 0 if plan_missing_sdf == 0 else 1

    if not os.environ.get("VINA_EXE", "").strip():
        fallback = Path(r"D:\obsfu\tools\vina\vina.exe")
        if fallback.is_file():
            os.environ["VINA_EXE"] = str(fallback)

    rows: list[dict[str, Any]] = []
    ckpt_f = ckpt_path.open("a", encoding="utf-8")

    try:
        for i, pc in enumerate(pockets_cfg, start=1):
            pid = str(pc.id)
            pocket_cfg = OmegaConf.to_container(pc, resolve=True)  # type: ignore[assignment]

            m_orig = metrics[
                (metrics["pocket_id"].astype(str) == pid) & (metrics[tcol].astype(str) == "original")
            ]
            m_crop = metrics[
                (metrics["pocket_id"].astype(str) == pid) & (metrics[tcol].astype(str) == crop_tag)
            ]
            if m_orig.empty or m_crop.empty:
                print(f"[{i}/{len(pockets_cfg)}] SKIP {pid}: missing metrics row(s)")
                continue
            v_oo = float(pd.to_numeric(m_orig.iloc[0]["vina_score_mean"], errors="coerce"))
            v_cc = float(pd.to_numeric(m_crop.iloc[0]["vina_score_mean"], errors="coerce"))

            sdf = _sdf_path(gen_root, pid, "original")
            if not sdf.is_file():
                print(f"[{i}/{len(pockets_cfg)}] SKIP {pid}: no original SDF")
                continue

            if pid in done and np.isfinite(done[pid]):
                v_oc = done[pid]
                print(f"[{i}/{len(pockets_cfg)}] {pid}: control from checkpoint V_oc={v_oc:.4f}")
            else:
                pdb = _resolve(root, pc.pdb)
                sdf_lig = pc.get("sdf")
                base = load_pocket_from_complex(
                    pdb_path=pdb,
                    sdf_path=_resolve(root, sdf_lig) if sdf_lig else None,
                    radius=radius,
                    pocket_id=pid,
                    metadata={"full_pdb": str(pdb)},
                    ligand_chain=str(pc.ligand_chain) if pc.get("ligand_chain") is not None else None,
                    ligand_resseq=int(pc.ligand_resseq) if pc.get("ligand_resseq") is not None else None,
                )
                orig_pock = _pocket_for_tag(base, "original", pert_specs, pocket_cfg)
                crop_pock = _pocket_for_tag(base, crop_tag, pert_specs, pocket_cfg)
                # Same docking box as original condition; only receptor atoms change.
                crop_rec = _with_centroid(crop_pock, np.asarray(orig_pock.ligand_centroid))

                mols = _load_mols(sdf, max_n=args.max_mols)
                if not mols:
                    print(f"[{i}/{len(pockets_cfg)}] SKIP {pid}: empty SDF")
                    continue
                print(
                    f"[{i}/{len(pockets_cfg)}] {pid}: docking {len(mols)} orig mols "
                    f"on cropped receptor (n_cpu={n_cpu})…",
                    flush=True,
                )
                scores = dock_mod.score_molecules(
                    mols, crop_rec, n_cpus=n_cpu, verbose=bool(args.verbose)
                )
                v_oc = _mean_vina(scores)
                rec = {"pocket_id": pid, "vina_orig_mols_crop_rec": v_oc, "n_mols": len(mols)}
                ckpt_f.write(json.dumps(rec) + "\n")
                ckpt_f.flush()
                done[pid] = v_oc
                print(f"  V_oc={v_oc:.4f}  V_oo={v_oo:.4f}  V_cc={v_cc:.4f}")

            total = v_cc - v_oo
            receptor_eff = v_oc - v_oo
            molecule_eff = total - receptor_eff  # == v_cc - v_oc
            rows.append({
                "pocket_id": pid,
                "vina_orig_mols_orig_rec": v_oo,
                "vina_crop_mols_crop_rec": v_cc,
                "vina_orig_mols_crop_rec": v_oc,
                "dVina_total": total,
                "dVina_receptor_effect": receptor_eff,
                "dVina_molecule_effect": molecule_eff,
                "control_tag": CONTROL_TAG,
                "crop_tag": crop_tag,
                "run_id": run_id,
            })
    finally:
        ckpt_f.close()

    if not rows:
        raise SystemExit("No pockets completed - nothing to write.")

    per = pd.DataFrame(rows)
    per_path = out_dir / "vina_receptor_control_per_pocket.csv"
    per.to_csv(per_path, index=False)

    wilcox_rows = [
        _wilcoxon_delta(
            per["dVina_total"].to_numpy(),
            label="total",
            comparison=f"{crop_tag} mols/rec vs original mols/rec",
        ),
        _wilcoxon_delta(
            per["dVina_receptor_effect"].to_numpy(),
            label="receptor_effect",
            comparison=f"{CONTROL_TAG} − original (same mols, cropped rec)",
        ),
        _wilcoxon_delta(
            per["dVina_molecule_effect"].to_numpy(),
            label="molecule_effect",
            comparison=f"crop mols on crop rec − orig mols on crop rec",
        ),
    ]
    wil = pd.DataFrame(wilcox_rows)
    wil_path = out_dir / "vina_receptor_control_wilcoxon.csv"
    wil.to_csv(wil_path, index=False)

    # Fraction of |total| attributable to |receptor| when signs agree (descriptive)
    tot = per["dVina_total"].to_numpy(dtype=float)
    rec_e = per["dVina_receptor_effect"].to_numpy(dtype=float)
    mol_e = per["dVina_molecule_effect"].to_numpy(dtype=float)
    frac_rec = np.where(np.abs(tot) > 1e-9, rec_e / tot, np.nan)

    lines = [
        "Vina receptor-control decomposition (Job A)",
        f"run_id: {run_id}",
        f"n_pockets: {len(per)}",
        f"crop_tag: {crop_tag}",
        f"control: {CONTROL_TAG}",
        "",
        ABS_SCORE_NOTE,
        "",
        "Per-component paired Wilcoxon (Δ vs 0):",
    ]
    for _, r in wil.iterrows():
        lines.append(
            f"  {r['component']}: median d={r['median_delta']:+.4f}, "
            f"p={r['p_value']:.3e}, rank-biserial r={r['rank_biserial_r']:+.3f}, n={int(r['n_pairs'])}"
        )
    lines += [
        "",
        f"median |dVina_total|={float(np.nanmedian(np.abs(tot))):.4f}",
        f"median |dVina_receptor_effect|={float(np.nanmedian(np.abs(rec_e))):.4f}",
        f"median |dVina_molecule_effect|={float(np.nanmedian(np.abs(mol_e))):.4f}",
        f"median receptor/total ratio (signed)={float(np.nanmedian(frac_rec)):.3f}",
        "",
        f"Wrote {per_path}",
        f"Wrote {wil_path}",
    ]
    summary_path = out_dir / "vina_receptor_control_summary.txt"
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
