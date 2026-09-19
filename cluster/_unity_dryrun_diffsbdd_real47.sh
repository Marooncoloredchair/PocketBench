#!/bin/bash
# DRY RUN ONLY for the DiffSBDD real47 regeneration. Submits nothing.
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
cd "$WORK/repo"

module load conda/latest 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/pocketbench"
source cluster/env.local.sh

CFG=configs/experiments/diffsbdd_real47.yaml

export DIFFSBDD_REPO="${DIFFSBDD_REPO:-$WORK/DiffSBDD}"
export DIFFSBDD_CHECKPOINT="${DIFFSBDD_CHECKPOINT:-$WORK/ckpts/crossdocked_fullatom_cond.ckpt}"
export DIFFSBDD_PYTHON="${DIFFSBDD_PYTHON:-$WORK/envs/pocketbench/bin/python}"
export POCKETBENCH_ACCOUNT=pi_nzawia_uri_edu
export POCKETBENCH_CONDA_ENV="$WORK/envs/pocketbench"
export POCKETBENCH_PYTHON="$WORK/envs/pocketbench/bin/python"
export POCKETBENCH_PARTITION="${POCKETBENCH_PARTITION:-gpu}"
export POCKETBENCH_GRES="${POCKETBENCH_GRES:-gpu:2080_ti:1}"

echo "=== resolved env ==="
echo "CFG=$CFG"
echo "DIFFSBDD_REPO=$DIFFSBDD_REPO"
echo "DIFFSBDD_CHECKPOINT=$DIFFSBDD_CHECKPOINT"
echo "DIFFSBDD_PYTHON=$DIFFSBDD_PYTHON"
echo "POCKETBENCH_RESULTS=${POCKETBENCH_RESULTS:-<unset>}"
echo "PARTITION=$POCKETBENCH_PARTITION GRES=$POCKETBENCH_GRES"

echo "=== preflight ==="
[[ -d "$DIFFSBDD_REPO" ]]       && echo "ok: DiffSBDD repo"   || { echo "FAIL: repo"; exit 1; }
[[ -f "$DIFFSBDD_CHECKPOINT" ]] && echo "ok: checkpoint"      || { echo "FAIL: ckpt"; exit 1; }
[[ -f "$CFG" ]]                 && echo "ok: config"          || { echo "FAIL: cfg"; exit 1; }
echo -n "pocket pdbs: "; ls data/raw/real50/*.pdb | wc -l
echo -n "run_id: "; grep '^run_id:' "$CFG"
echo -n "n_samples: "; grep 'n_samples:' "$CFG"

echo "=== existing panel with this run_id (must be absent to avoid clobber) ==="
ls -l data/results/metrics_per_condition__rundiffsbdd_real47*.csv 2>/dev/null || echo "none - clean"
ls -l data/results/per_molecule__rundiffsbdd_real47*.csv 2>/dev/null || echo "no sidecar yet - clean"

echo "=== DRY RUN ==="
bash cluster/submit.sh --config "$CFG" --job-name pb-diffsbdd-real47 --dry-run
echo "DRYRUN_DONE (nothing submitted)"
