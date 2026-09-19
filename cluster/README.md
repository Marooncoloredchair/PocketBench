# PocketBench SLURM cluster port

Submittable wrappers for **URI Unity** and **EPFL** (both SLURM). Nothing here
runs jobs by itself — you `sbatch` from a login node after editing env vars.

## Layout

| File | Role |
|------|------|
| `env.example.sh` | Site paths, conda, CUDA, checkpoints as **variables** (no `D:\`) |
| `submit.sh` | Build + submit (or `--dry-run`) an array job from a YAML config |
| `run_array_task.sh` | One SLURM array task → one pocket via `--pocket-index` |
| `merge_array_results.py` | Concatenate `metrics_per_condition__run*__pNNNN.csv` → panel CSV |
| `submit_template.slurm` | Generated template reference (also emitted by `submit.sh`) |

## Unity access (from WSL)

```bash
ssh -o IdentitiesOnly=yes -i ~/.ssh/unity_key \
  terrell_osborne_uri_edu@login.unityhpc.org
```

Use PI work storage (not `/work/$USER`):

```
/work/pi_nzawia_uri_edu/pocketbench/
```

GPU partition: `uri-gpu`, long QoS: `-q long`. See also `D:\BNDF\scripts\unity\`.

## After Job A (Vina panel)

Decompose crop ΔVina into receptor vs molecule effects:

```bash
python analysis/vina_receptor_control.py \
  --config configs/cluster/diffsbdd_real100_vina.yaml \
  --metrics-csv "$POCKETBENCH_RESULTS/metrics_per_condition__rundiffsbdd_real100_vina.csv" \
  --generations-root "$POCKETBENCH_GENERATIONS/run_diffsbdd_real100_vina"
```

See script docstring for the `original_mols_cropped_receptor` control.

## Resume from a partial panel CSV (Job B)


If a non-array run already wrote `metrics_per_condition__run{ID}.csv` for some
pockets, seed shard files before submitting so `--resume` skips them:

```bash
python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir "$POCKETBENCH_RESULTS" \
  --run-id pocket2mol_real47_full_v2
```

## Conda environments (two, not one)

Pocket2Mol pins a 2022 stack (`env_cuda113.yml`: python 3.8, torch 1.10, PyG
2.0.4) that cannot coexist with the DiffSBDD/Vina stack. Job B failed once
because it reused the `pocketbench` env, which has no `easydict`. Keep them
separate:

| Env | Path | Used by | Entered how |
| --- | --- | --- | --- |
| `pocketbench` | `/work/pi_nzawia_uri_edu/pocketbench/envs/pocketbench` | PocketBench CLI, DiffSBDD, Vina/Meeko, all metrics | `conda activate` in the SLURM script |
| `p2m` | `/work/pi_nzawia_uri_edu/pocketbench/envs/p2m` | Pocket2Mol sampling **only** | never activated; invoked as `POCKET2MOL_PYTHON` by the adapter subprocess |

The Pocket2Mol adapter shells out to `POCKET2MOL_PYTHON`, so the two stacks
never share an interpreter. Set in `env.local.sh`:

```bash
export POCKET2MOL_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/p2m/bin/python
export POCKET2MOL_REPO=/work/pi_nzawia_uri_edu/pocketbench/Pocket2Mol
export POCKET2MOL_CHECKPOINT=/work/pi_nzawia_uri_edu/pocketbench/ckpts/pretrained_Pocket2Mol.pt
```

### Rebuilding the `p2m` env

`cluster/build_p2m_env.sh` is the reproducible recipe. The non-obvious pins,
each found by an actual import/run failure:

- **`torch-geometric==2.4.0`** — PyG >= 2.5 renamed `torch_geometric.utils.subgraph`
  to a private `_subgraph` module, which breaks `utils/transforms.py`. 2.4.0 is
  the newest version whose import surface Pocket2Mol still matches.
- **`torch-cluster`** — `utils/transforms.py` calls `knn_graph`, which PyG only
  dispatches when `torch-cluster` is installed. Compile with
  `--no-build-isolation` (torch must already be importable) or the build fails
  with `No module named 'torch'`. Include `6.1` in `TORCH_CUDA_ARCH_LIST`:
  Unity's `gpu` partition has GTX 1080 Ti (sm_61). A wheel built only for
  7.0+ raises `cudaErrorNoKernelImageForDevice`.
- **`numpy<2`** — the 2022-era code paths use APIs removed in NumPy 2.
- **`easydict`** — the dependency that broke the first Job B submission.

`torch-scatter` and `torch-cluster` build CUDA kernels from source. Compile
them **on a GPU node** (`srun --partition=gpu … bash cluster/build_p2m_env.sh`),
or kernels built on the login node will raise `cudaErrorNoKernelImageForDevice`
on gypsum. `torch-sparse` is not required by Pocket2Mol's sample path and
failed to compile here — leave it out.

Verify before submitting an array — never submit 28 tasks on an unverified env.
The Unity `gpu` partition includes Tesla M40 (sm_52). CUDA 12 cannot target
Maxwell, so the smoke and Job B request `--gres=gpu:2080_ti:1` (Turing sm_75).
Do not use bare `gpu:1` for Pocket2Mol.

```bash
bash cluster/_unity_p2m_smoke.sh   # 1 pocket, 4 samples, own run_id, asserts mols > 0
```

## Quick start


```bash
cd /work/pi_nzawia_uri_edu/pocketbench/repo   # or your clone
cp cluster/env.example.sh cluster/env.local.sh
# edit POCKETBENCH_ROOT, DIFFSBDD_*, modules, etc.
source cluster/env.local.sh

# Validate plan without sbatch:
bash cluster/submit.sh \
  --config configs/diffsbdd_real100.yaml \
  --job-name pb-real100 \
  --dry-run

# Submit array (one GPU task per pocket):
bash cluster/submit.sh \
  --config configs/diffsbdd_real100.yaml \
  --job-name pb-real100

# After array completes:
python cluster/merge_array_results.py \
  --results-dir "$POCKETBENCH_RESULTS" \
  --run-id diffsbdd_real100
```

## How array sharding works

1. `submit.sh` counts `pockets:` in the YAML and emits `#SBATCH --array=0-(N-1)`.
2. Each task runs `pocketbench run --config … --pocket-index $SLURM_ARRAY_TASK_ID`.
3. Metrics land in `metrics_per_condition__run{id}__p0000.csv`, … (no shared-file race).
4. Generations still go under `…/generations/run_{id}/{POCKET}/…` (one pocket per task).
5. `merge_array_results.py` writes the merged panel CSV + a short coverage report.

## Paths

Configs use **relative POSIX** paths (`data/raw/real100/…`). On the cluster,
set `project_root` in the YAML (or run from the repo root) and keep
`POCKETBENCH_ROOT` as the clone root. The loader (`sbdd_robust.cli._resolve`)
joins relative paths with `pathlib` — no backslash assumptions.

Override results/generations via env when writing a site-local config copy,
or edit `paths:` in a cluster YAML under `$POCKETBENCH_WORK/configs/`.

## EPFL notes

Override in `env.local.sh`:

```bash
export POCKETBENCH_SITE=epfl
export POCKETBENCH_PARTITION=…          # site-specific
export POCKETBENCH_ACCOUNT=…            # if required
export POCKETBENCH_QOS=…
export POCKETBENCH_MODULES=…            # or leave empty and use module load in a custom block
export POCKETBENCH_ROOT=/path/to/PocketBench
```

`submit.sh` passes `--account` only when `POCKETBENCH_ACCOUNT` is set.

## Do not

- Paste private keys into chat or into committed files.
- Hardcode Windows paths (`D:\…`) in cluster scripts or env.
- Run `sbatch` from this Windows checkout without verifying POSIX paths first.
