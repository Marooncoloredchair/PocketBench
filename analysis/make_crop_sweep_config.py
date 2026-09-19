#!/usr/bin/env python3
"""Generate a fine crop-radius **sweep** config from a base benchmark config.

The base config perturbs each pocket at a single ±1.5 Å crop. This rewrites the
``perturbations`` block to sweep the crop radius across a grid of offsets so the
QED-vs-radius dose-response (and the Pocket Boundary Sensitivity Index) can be
fit on many points instead of three. Atom-shuffle and coordinate-jitter are kept
so the featurization-noise floor (and PBSI signal-to-noise) is still measurable.

The pockets list, model block, and paths are carried over unchanged, so the sweep
runs on exactly the same panel. Use the resumable CLI on Colab:

    python -m sbdd_robust run --config configs/diffsbdd_real100_cropsweep.yaml --resume --normalized-only

Example:
    python analysis/make_crop_sweep_config.py \
        --base-config configs/diffsbdd_real100.yaml \
        --out configs/diffsbdd_real100_cropsweep.yaml \
        --deltas 0.5 1.0 1.5 2.0 2.5 3.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]


def build_perturbations(deltas: list[float], keep_noise: bool) -> list[dict]:
    specs: list[dict] = [{"tag": "original", "type": "identity"}]
    if keep_noise:
        specs.append({"tag": "atom_shuffle", "type": "atom_shuffle", "seed": 0})
        specs.append({"tag": "coordinate_jitter", "type": "coordinate_jitter", "sigma": 0.1, "seed": 1})
    for d in deltas:
        d = float(d)
        ds = f"{d:g}"
        specs.append({"tag": f"crop_radius_minus_{ds}", "type": "crop_radius_minus", "delta_angstrom": d})
        specs.append({"tag": f"crop_radius_plus_{ds}", "type": "crop_radius_plus", "delta_angstrom": d})
    return specs


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a crop-radius sweep config from a base config.")
    ap.add_argument("--base-config", required=True, type=str)
    ap.add_argument("--out", required=True, type=str)
    ap.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help="Crop-radius offsets in Å; each is applied as both minus and plus.",
    )
    ap.add_argument("--run-id", type=str, default=None, help="Override run_id (default: <base>_cropsweep).")
    ap.add_argument("--no-noise", action="store_true", help="Drop atom_shuffle/coordinate_jitter conditions.")
    args = ap.parse_args()

    base_path = Path(args.base_config)
    if not base_path.is_absolute():
        base_path = (_ROOT / base_path).resolve()
    cfg = OmegaConf.load(base_path)

    base_run = str(cfg.get("run_id", base_path.stem))
    run_id = args.run_id or f"{base_run}_cropsweep"
    cfg.run_id = run_id
    cfg.resume = True
    cfg.normalized_only = True

    deltas = sorted({round(float(d), 3) for d in args.deltas})
    specs = build_perturbations(deltas, keep_noise=not args.no_noise)
    cfg.perturbations = specs
    # invariant_tags drive the brittleness aggregate; keep the noise + ±1.5 set so the
    # existing brittleness outputs stay comparable across runs.
    cfg.invariant_tags = ["atom_shuffle", "coordinate_jitter", "crop_radius_plus_1.5", "crop_radius_minus_1.5"]

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = (_ROOT / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, out_path)

    n_pockets = len(list(cfg.pockets))
    n_cond = len(specs)
    print(f"Wrote {out_path}")
    print(f"  run_id={run_id}  resume=true  normalized_only=true")
    print(f"  {n_pockets} pockets × {n_cond} conditions = {n_pockets * n_cond} generations")
    print(f"  crop offsets (A): +/-{', +/-'.join(f'{d:g}' for d in deltas)}")
    r0 = float(cfg.get("extraction_radius"))
    print(f"  base extraction_radius = {r0:g} A -> radii span "
          f"{r0 - max(deltas):g}-{r0 + max(deltas):g} A")


if __name__ == "__main__":
    main()
