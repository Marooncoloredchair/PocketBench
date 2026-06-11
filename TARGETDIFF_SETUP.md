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

1. Create a conda env matching TargetDiff (see their `environment.yaml`).
2. Set:

```text
set TARGETDIFF_REPO=D:\path\to\targetdiff
set TARGETDIFF_PYTHON=D:\path\to\conda\envs\targetdiff\python.exe
set TARGETDIFF_SAMPLING_YAML=D:\path\to\targetdiff\configs\sampling.yml
```

3. Download a pretrained checkpoint referenced by that sampling config (see TargetDiff README).

4. Run:

```bash
pocketbench run --config configs/targetdiff_real100.yaml
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
