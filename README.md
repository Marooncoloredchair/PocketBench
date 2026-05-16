# SBDD-Robust

**SBDD-Robust** is an open-source benchmark toolkit for measuring **reliability** of structure-based drug design (SBDD) generative models under **pocket perturbations** that preserve chemistry but change featurization (atom order, coordinate jitter, crop radius). Plug in your model with a small adapter, run on any set of PDB complexes, and get **brittleness rates**, **coverage**, chemistry metrics, and optional **docking** scores—typically in one afternoon once your backend is wired.

The goal: if someone training a new SBDD model can run

```bash
python -m sbdd_robust run --config mymodel.yaml
```

and receive a clear robustness report, they will **use** the tool and **cite** the paper—building a long-term citation habit around reproducible stress testing.

## Quick start

```bash
git clone https://github.com/Marooncoloredchair/PocketBench.git
cd PocketBench
pip install -e ".[dev]"

# Fast sanity check (mock model, a few pockets)
python -m sbdd_robust run --config configs/examples/smoke.yaml
```

See **`configs/examples/`** for `diffsbdd.example.yaml`, `pocket2mol.example.yaml`, and small real panels. See **`sbdd_robust/models/`** to add a new `model.type`.

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
| `sbdd_robust/` | Importable library + `python -m sbdd_robust` CLI (**the tool**). |
| `configs/examples/` | Small configs for new users and CI. |
| `configs/experiments/` | Full paper configs (47-pocket panel, meaningful runs, …). |
| `data/raw/` | Input PDBs (`real50/` panel, `smoke/`, …). |
| `data/results/` | Frozen paper CSVs and run logs. |
| `paper/` | Manuscript, figures, LaTeX (`paper/biorxiv_submission/`). |
| `experiments/` | Shell scripts that reproduce paper outputs. |
| `analysis/` | Plotting and aggregate analysis. |
| `tests/` | `pytest` suite |

## Citation

If you use this software, cite the **bioRxiv preprint** (update DOI when posted) and the repository. GitHub reads **`CITATION.cff`** for the “Cite this repository” widget.

## License

MIT — see **`LICENSE`**.
