#!/usr/bin/env python3
"""Generate a **brittleness-stress** sweep config to test Prof. Ma's first-atom mechanism.

Builds, from any base benchmark config, a `perturbations` block with three families:

  * **featurization floor** — atom_shuffle, coordinate_jitter (should barely move output)
  * **boundary dose** — crop_radius_minus across a grid (symmetric boundary change)
  * **frame-moving** — face_peel (asymmetric, model-agnostic, shifts pocket centroid)
    and, for Pocket2Mol only, anchor_offset (moves the first-atom seeding region with
    chemistry held fixed)

The Initialization Sensitivity Ratio (ISR) then contrasts the frame-moving family
against the featurization floor. The mechanism predicts ISR >> 1 for autoregressive
Pocket2Mol and ISR ~ 1 for diffusion models (DiffSBDD / TargetDiff).

anchor_offset is a deliberate no-op for adapters that derive their own pocket frame
from residues (DiffSBDD), so it is only added when the base model is pocket2mol unless
``--force-anchor`` is given.

Example:
    python analysis/make_brittleness_stress_config.py \
        --base-config configs/pocket2mol_real100.yaml \
        --out configs/pocket2mol_real100_stress.yaml
    python analysis/make_brittleness_stress_config.py \
        --base-config configs/diffsbdd_real100.yaml \
        --out configs/diffsbdd_real100_stress.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]


def build_perturbations(
    crop_deltas: list[float],
    peel_fractions: list[float],
    anchor_offsets: list[float],
    include_anchor: bool,
) -> list[dict]:
    specs: list[dict] = [
        {"tag": "original", "type": "identity"},
        {"tag": "atom_shuffle", "type": "atom_shuffle", "seed": 0},
        {"tag": "coordinate_jitter", "type": "coordinate_jitter", "sigma": 0.1, "seed": 1},
    ]
    for d in crop_deltas:
        ds = f"{float(d):g}"
        specs.append({"tag": f"crop_radius_minus_{ds}", "type": "crop_radius_minus", "delta_angstrom": float(d)})
    for f in peel_fractions:
        fs = f"{float(f):g}"
        specs.append({"tag": f"face_peel_{fs}", "type": "face_peel", "fraction": float(f), "axis": "pca", "direction": "plus"})
    if include_anchor:
        for a in anchor_offsets:
            as_ = f"{float(a):g}"
            specs.append({"tag": f"anchor_offset_{as_}", "type": "anchor_offset", "offset_angstrom": float(a), "direction": "pca"})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a brittleness-stress sweep config from a base config.")
    ap.add_argument("--base-config", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument(
        "--minimal",
        action="store_true",
        help="Use a tiny ISR panel: one crop (1.5 A), one face_peel (0.25), one anchor (2.0 A). "
        "Pair with --max-pockets 3-5 for a quick mechanism sniff test.",
    )
    ap.add_argument(
        "--max-pockets",
        type=int,
        default=None,
        help="Keep only the first N pockets from the base config (after --pocket-ids filtering).",
    )
    ap.add_argument(
        "--pocket-ids",
        nargs="+",
        default=None,
        help="Keep only these pocket IDs (case-insensitive), in the order given.",
    )
    ap.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Override model.n_samples (lower = faster smoke runs).",
    )
    ap.add_argument("--crop-deltas", nargs="+", type=float, default=[0.5, 1.0, 1.5, 2.0, 2.5])
    ap.add_argument("--peel-fractions", nargs="+", type=float, default=[0.15, 0.25, 0.35])
    ap.add_argument("--anchor-offsets", nargs="+", type=float, default=[1.0, 2.0, 3.0])
    ap.add_argument("--run-id", type=str, default=None, help="Override run_id (default: <base>_stress).")
    ap.add_argument("--force-anchor", action="store_true", help="Add anchor_offset even for non-pocket2mol models.")
    ap.add_argument("--no-anchor", action="store_true", help="Never add anchor_offset conditions.")
    args = ap.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = (_ROOT / base_path).resolve()
    cfg = OmegaConf.load(base_path)

    if args.pocket_ids:
        wanted = {str(x).strip().upper() for x in args.pocket_ids}
        order = {str(x).strip().upper(): i for i, x in enumerate(args.pocket_ids)}
        pockets = [p for p in list(cfg.pockets) if str(p.get("id", "")).upper() in wanted]
        pockets.sort(key=lambda p: order[str(p.get("id", "")).upper()])
        missing = sorted(wanted - {str(p.get("id", "")).upper() for p in pockets})
        if missing:
            raise SystemExit(f"pocket IDs not found in {base_path}: {', '.join(missing)}")
        cfg.pockets = pockets
    if args.max_pockets is not None:
        if args.max_pockets < 1:
            raise SystemExit("--max-pockets must be >= 1")
        cfg.pockets = list(cfg.pockets)[: args.max_pockets]

    crop_deltas = [1.5] if args.minimal else list(args.crop_deltas)
    peel_fractions = [0.25] if args.minimal else list(args.peel_fractions)
    anchor_offsets = [2.0] if args.minimal else list(args.anchor_offsets)

    model_type = str(cfg.get("model", {}).get("type", "")).lower()
    include_anchor = (model_type == "pocket2mol" or args.force_anchor) and not args.no_anchor

    base_run = str(cfg.get("run_id", base_path.stem))
    if args.run_id:
        run_id = args.run_id
    elif args.minimal and args.max_pockets:
        run_id = f"{base_run}_isr_smoke{args.max_pockets}"
    elif args.minimal:
        run_id = f"{base_run}_isr_smoke"
    else:
        run_id = f"{base_run}_stress"
    cfg.run_id = run_id
    cfg.resume = True
    cfg.normalized_only = True
    if args.minimal:
        cfg.skip_failed_pockets = True
    if args.n_samples is not None:
        cfg.setdefault("model", {})
        cfg.model.n_samples = int(args.n_samples)

    crop = sorted({round(float(d), 3) for d in crop_deltas})
    peel = sorted({round(float(f), 3) for f in peel_fractions})
    anchor = sorted({round(float(a), 3) for a in anchor_offsets})
    specs = build_perturbations(crop, peel, anchor, include_anchor)
    cfg.perturbations = specs
    # invariant_tags keep the legacy brittleness aggregate comparable across runs.
    cfg.invariant_tags = ["atom_shuffle", "coordinate_jitter", "crop_radius_minus_1.5"]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)

    n_pockets = len(list(cfg.pockets))
    n_cond = len(specs)
    print(f"Wrote {out_path}")
    print(f"  model.type={model_type or '(unset)'}  run_id={run_id}  resume=true  normalized_only=true")
    print(f"  {n_pockets} pockets x {n_cond} conditions = {n_pockets * n_cond} generations")
    print(f"  featurization floor : atom_shuffle, coordinate_jitter")
    print(f"  boundary dose       : crop_radius_minus {', '.join(f'{d:g}' for d in crop)} A")
    print(f"  frame-moving        : face_peel {', '.join(f'{f:g}' for f in peel)}"
          + (f"; anchor_offset {', '.join(f'{a:g}' for a in anchor)} A" if include_anchor else " (no anchor_offset)"))


if __name__ == "__main__":
    main()
