# TargetDiff integration (PocketBench)

The upstream repository layout assumed here is **`https://github.com/guanjq/targetdiff`** (ICLR 2023; J. Qu et al. naming on Zenodo differs—clone the public GitHub tree).

## Why not `sample_diffusion.py`?

Upstream `scripts/sample_diffusion.py` samples entries from a **torch-geometric dataset** by integer `--data_id`. For arbitrary PDB pockets from this benchmark we call **`scripts/sample_for_pocket.py`**, which accepts:

- positional `config` (YAML path)
- `--pdb_path` — pocket/protein PDB
- `--result_path` — output directory
- `--num_samples` — overrides batch count
- `--device`, `--batch_size`

Generated molecules appear under ``<result_path>/sdf/*.sdf``; the adapter loads them with RDKit.

## Environment

1. Clone https://github.com/guanjq/targetdiff and download pretrained checkpoints into
   ``pretrained_models/`` (see upstream README Google Drive link).
2. On this machine we reuse the **pocket2mol** conda env (torch 1.13 + pyg) for upstream
   sampling; install **openbabel** if missing:

```powershell
conda install -n pocket2mol -c conda-forge openbabel -y
```

3. NumPy ≥1.24 removed ``np.long`` / ``np.bool`` aliases used upstream; patch
   ``utils/data.py`` and ``datasets/protein_ligand.py`` in your clone (``np.int64``,
   ``np.bool_``) or use an older NumPy pin.
4. Do **not** set ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`` for torch 1.13.
5. Set:

```text
set TARGETDIFF_REPO=D:\obsfu\targetdiff
set TARGETDIFF_PYTHON=D:\Miniforge\envs\pocket2mol\python.exe
set TARGETDIFF_SAMPLING_YAML=D:\obsfu\targetdiff\configs\sampling.yml
```

6. ISR smoke panel (matched 5-pocket panel vs Pocket2Mol):

```powershell
powershell -File scripts/run_targetdiff_isr_smoke_local.ps1
```

`run_meta__*.json` records `targetdiff_commit` when `TARGETDIFF_REPO` is a git clone.

## Config fields (`model` block)

| Key | Meaning |
|-----|--------|
| `type: targetdiff` | Selects `TargetDiffAdapter` |
| `repo_root` | Clone root (must contain `scripts/sample_for_pocket.py`) |
| `config_yaml` | Sampling YAML passed as first positional arg |
| `python_exe` | Interpreter with torch/pyg installed |
| `script_path` | Optional override if your fork moved the script |
| `device` | e.g. `cuda:0` |
| `batch_size` | Passed through to upstream argparse |

## Troubleshooting

- **`ModuleNotFoundError: utils`**: `PYTHONPATH` must include `repo_root`; the adapter sets this for the subprocess.
- **Empty SDF list**: check stderr from the upstream script; reconstruction may drop disconnected fragments.
