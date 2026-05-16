# Raw structures

Structure files used as inputs to the benchmark.

| Directory | Contents |
|-----------|----------|
| **`smoke/`** | Tiny synthetic example for `configs/examples/smoke.yaml`. |
| **`real50/`** | Forty-seven RCSB complexes for the paper invariant panel (`configs/experiments/pocket2mol_real47.yaml`). Download or rebuild with `scripts/generate_pocket2mol_real47_yaml.py` (see `POCKET2MOL_SETUP.md`). |
| **`real/`** | Additional PDBs used in other examples. |

**Reproducibility:** archived **metrics and summaries** live under `data/results/` (frozen CSVs from completed runs). Raw PDBs are large; you can regenerate them from the same PDB IDs and extraction recipe documented in the paper configs.
