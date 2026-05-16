# Paper experiment configs

These YAML files match the benchmark described in the manuscript (47-pocket panel, same invariant tags, radii, and perturbation parameters).

| File | Purpose |
|------|---------|
| `pocket2mol_real47.yaml` | Pocket2Mol on the full 47-pocket panel (requires Pocket2Mol checkout + weights). |
| `diffsbdd_real47.yaml` | DiffSBDD on the same 47 pockets. Requires **environment variables** (no secrets—just local paths to your DiffSBDD install): |
| `diffsbdd_meaningful_real20.yaml` | Meaningful mutation panel (DiffSBDD, 20 pockets). |
| `diffsbdd_nsamples50_subset10.yaml` | BrSubset for n=50 vs n=20 sensitivity (10 pockets). |

## DiffSBDD 47-pocket rerun (`diffsbdd_real47.yaml`)

Export before `python -m sbdd_robust run --config configs/experiments/diffsbdd_real47.yaml`:

- `DIFFSBDD_REPO` — root of the DiffSBDD clone (contains `generate_ligands.py` and checkpoints).
- `DIFFSBDD_CHECKPOINT` — path to `.ckpt` (e.g. `.../crossdocked_fullatom_cond.ckpt`).
- `DIFFSBDD_PYTHON` — Python executable in the **DiffSBDD** conda/env (not necessarily the `sbdd-robust` env).

The config uses OmegaConf interpolation: `repo_root: ${env:DIFFSBDD_REPO}`, etc.

For **exact** Table 1 / Fig. 1 numbers from the paper, use the frozen metrics CSV in `data/results/` and `experiments/reproduce_table1.sh` (default path; no GPU required).
