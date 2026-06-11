# Raw structures

Structure files used as inputs to the benchmark.

| Directory | Contents |
|-----------|----------|
| **`smoke/`** | Tiny synthetic example for `configs/examples/smoke.yaml`. |
| **`real50/`** | The **47-complex** RCSB invariant panel used in the paper (the `real50/` name is historical; the panel size is 47). Rebuild with `scripts/generate_pocket2mol_real47_yaml.py` (see `POCKET2MOL_SETUP.md`). |
| **`real100/`** | Expanded **100-pocket** panel referenced by `configs/diffsbdd_real100.yaml` / `configs/pocket2mol_real100.yaml`. |
| **`real/`** | Additional PDBs used in other examples. |

**Reproducibility:** archived **metrics and summaries** live under `data/results/` (frozen CSVs from completed runs). Raw PDBs are large; you can regenerate them from the same PDB IDs and extraction recipe documented in the paper configs.
