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
