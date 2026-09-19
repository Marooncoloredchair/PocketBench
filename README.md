# PocketBench

**PocketBench** measures whether structure-based drug design (SBDD) generative models are **reliable** — not just whether they generate good molecules, but whether their outputs survive perturbations to how the binding pocket is defined and featurized. It separates two things benchmarks usually conflate:

- **Featurization noise** (atom-order shuffle, sub-ångström coordinate jitter) that *should* leave the answer unchanged, and
- **Pocket boundary definition** (the crop radius) — an unreported preprocessing choice that, it turns out, silently moves drug-likeness.

Its headline deliverable is the **Pocket Boundary Sensitivity Index (PBSI)**: a single, comparable number quantifying how much a model's drug-likeness depends on the crop radius, reported *relative to that model's own featurization-noise floor*. In our runs, the crop choice moves QED more than featurization noise does for **~3 of every 4 pockets**.

> The importable package / CLI module is **`sbdd_robust`** (`python -m sbdd_robust` or the `pocketbench` entry point); the project/repository is **PocketBench**.

## Analyze your own model in 30 seconds (no GPU)

You do **not** need to rerun PocketBench's generation to use it. If your model can already sample molecules for a pocket, produce a per-condition metrics CSV (one row per pocket × perturbation) and PocketBench will score it:

```bash
pip install -e .

# brittleness + paired crop-radius test + PBSI, written to a paper-ready folder
pocketbench report --metrics my_model_metrics.csv --out-dir my_report

# or individual analyses
pocketbench pbsi        --metrics my_model_metrics.csv     # Pocket Boundary Sensitivity Index
pocketbench crop-test   --metrics my_model_metrics.csv     # paired Wilcoxon ΔQED / ΔSA
pocketbench brittleness --metrics my_model_metrics.csv --raw
```

### Input schema (bring-your-own-model)

One row per (pocket, perturbation condition):

| column | type | notes |
|---|---|---|
| `pocket_id` | str | target identifier |
| `perturbation_tag` | str | `original`, `atom_shuffle`, `coordinate_jitter`, `crop_radius_plus_<x>`, `crop_radius_minus_<x>` |
| `model_name` | str | optional; filter with `--model` |
| `validity`, `uniqueness` | float | in [0,1] |
| `mean_qed`, `std_qed` | float | RDKit QED over the sampled molecules |
| `mean_sa`, `std_sa` | float | optional (RDKit SA, ~1–10) |

`pocketbench pbsi` needs `original` plus at least one `crop_radius_*` condition; with the standard ±1.5 Å pair it returns a coarse estimate, and a fine sweep (below) yields the full dose-response.

## Run the full benchmark (with a model backend)

```bash
git clone https://github.com/Marooncoloredchair/PocketBench.git
cd PocketBench
pip install -e ".[dev]"

# Fast sanity check (mock model, a few pockets)
python -m sbdd_robust run --config configs/examples/smoke.yaml

# Report normalized brittleness alongside the raw flag automatically
python -m sbdd_robust run --config mymodel.yaml --normalized-only --resume
```

### Crop-radius sweep (the dose-response)

Generate a fine sweep config from any base config, then run it (resumable) to chart the full QED-vs-radius curve and per-pocket critical radii:

```bash
python analysis/make_crop_sweep_config.py \
  --base-config configs/diffsbdd_real100.yaml \
  --out configs/diffsbdd_real100_cropsweep.yaml \
  --deltas 0.5 1.0 1.5 2.0 2.5 3.0
python -m sbdd_robust run --config configs/diffsbdd_real100_cropsweep.yaml --resume --normalized-only
pocketbench pbsi --metrics data/results/metrics_per_condition__rundiffsbdd_real100_cropsweep.csv
```

See **`configs/examples/`** for `diffsbdd.example.yaml`, `pocket2mol.example.yaml`, and small real panels. See **`sbdd_robust/models/`** to add a new `model.type`.

## Metric definitions

- **Normalized brittleness rate** — fraction of pockets whose std across the four perturbation conditions exceeds threshold τ, on the normalized metric subset (validity, uniqueness, mean/std QED). Reported as the *primary* view; the all-column flag over raw counts/SA is a cautionary secondary (it inflates any [0,1]-tuned τ).
- **Crop-radius paired test** — within-pocket paired two-sided Wilcoxon signed-rank of (crop − original) for ΔQED and ΔSA, with matched-pairs rank-biserial effect size and bootstrap 95% CIs.
- **Pocket Boundary Sensitivity Index (PBSI)** — median |slope| of QED vs crop-radius offset (QED/Å), with a per-pocket boundary-to-noise SNR = (|slope|·1.5 Å)/(QED std across featurization-noise conditions). Implemented in `sbdd_robust/report.py`.

## Reproducing the paper results

Frozen **per-condition metrics** and summaries from the manuscript are under **`data/results/`** (the reproducibility record for reviewers and NMI-style checks).

One command to regenerate **Table 1** and the **Fig. 1** threshold curve from those CSVs (no GPU):

```bash
bash experiments/reproduce_all.sh
```

Or step by step:

```bash
bash experiments/reproduce_table1.sh   # Table 1 + threshold_sensitivity.pdf
bash experiments/reproduce_fig1.sh     # same Fig. 1 plot (explicit)
bash experiments/00_smoke_test.sh       # mock-model regression smoke
```

To **re-run** the full DiffSBDD 47-pocket generator pipeline (not required to match archived numbers; stochastic):

```bash
export DIFFSBDD_REPO=/path/to/DiffSBDD
export DIFFSBDD_CHECKPOINT=/path/to/crossdocked_fullatom_cond.ckpt
export DIFFSBDD_PYTHON=/path/to/diffsbdd-env/bin/python
export SBDD_ROBUST_FULL_RERUN=1
bash experiments/reproduce_table1.sh
```

Details: **`configs/experiments/README.md`**.

## Repository layout

| Path | Role |
|------|------|
| `sbdd_robust/` | Importable library + `python -m sbdd_robust` / `pocketbench` CLI (**the tool**). |
| `sbdd_robust/report.py` | Shared metric library (PBSI, crop test, normalized brittleness) used by the CLI and `analysis/`. |
| `configs/examples/` | Small configs for new users and CI. |
| `configs/experiments/` | Full paper configs (47-pocket panel, meaningful runs, …). |
| `data/raw/` | Input PDBs: `real50/` (47-complex paper invariant panel), `real100/` (expanded 100-pocket panel), `smoke/` (tiny synthetic). |
| `data/results/` | Frozen paper CSVs and run logs. |
| `paper/` | Manuscript, figures, CSV outputs (`pocket_boundary_sensitivity.csv`, `crop_radius_wilcoxon.csv`, `normalized_brittleness.csv`). |
| `experiments/` | Shell scripts that reproduce paper outputs. |
| `analysis/` | Thin multi-panel wrappers over `sbdd_robust/report.py` + the crop-sweep config generator. |
| `tests/` | `pytest` suite |

## Citation

If you use this software, cite the **bioRxiv preprint** (update DOI when posted) and the repository. GitHub reads **`CITATION.cff`** for the “Cite this repository” widget.

## Acknowledgements & third-party code

PocketBench evaluates external generative models through thin adapters; it does **not** redistribute their weights. We gratefully build on:

- **DiffSBDD** — Schneuing, A. *et al.* "Structure-based drug design with equivariant diffusion models." Repository: <https://github.com/arneschneuing/DiffSBDD> (MIT License). The Colab pipeline applies small, clearly marked **derivative patches** to upstream DiffSBDD files (e.g. `generate_ligands.py`: `torch.load` compatibility shim and a `--device` flag) at runtime; these are modifications of the original MIT-licensed sources and are credited to the DiffSBDD authors.
- **Pocket2Mol** — Peng, X. *et al.* Repository: <https://github.com/pengxingang/Pocket2Mol>. See `POCKET2MOL_SETUP.md`.

If you use the corresponding model in your run, please cite that model's paper in addition to PocketBench.

## License

MIT — see **`LICENSE`**. Third-party models retain their own licenses.
