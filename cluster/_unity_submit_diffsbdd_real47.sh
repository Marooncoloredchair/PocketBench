#!/bin/bash
# Submit DiffSBDD on the real47 panel (Job C).
#
# WHY: the MW mechanism test (paper/mw_mechanism_both_models.csv) compared
# Pocket2Mol on real47 against DiffSBDD on real100_vina. Those panels share
# ZERO pockets, so "opposite MW directions" is confounded with pocket set.
# DiffSBDD's original real47 run (run1778615375) was executed on Colab and its
# molecules were never archived, so they cannot be re-parsed.
#
# This run puts DiffSBDD on the EXACT 47 pockets Pocket2Mol used and keeps the
# SDFs, making the cross-model MW/QED/SA comparison pocket-paired.
#
# Run from a Unity login node:
#   bash cluster/_unity_submit_diffsbdd_real47.sh
#
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
cd "$WORK/repo"

module load conda/latest 2>/dev/null || true
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/pocketbench"
source cluster/env.local.sh

# --- locate the config (prefer cluster/, fall back to experiments/) ---
CFG=""
for c in configs/cluster/diffsbdd_real47.yaml configs/experiments/diffsbdd_real47.yaml; do
  [[ -f "$c" ]] && { CFG="$c"; break; }
done
if [[ -z "$CFG" ]]; then
  echo "ERROR: diffsbdd_real47.yaml not found on the cluster." >&2
  echo "Copy it up first, e.g.:" >&2
  echo "  scp configs/experiments/diffsbdd_real47.yaml \\" >&2
  echo "    <user>@login.unityhpc.org:$WORK/repo/configs/cluster/diffsbdd_real47.yaml" >&2
  exit 1
fi
echo "config: $CFG"

# --- DiffSBDD environment (Job A used these) ---
export DIFFSBDD_REPO="${DIFFSBDD_REPO:-$WORK/DiffSBDD}"
export DIFFSBDD_CHECKPOINT="${DIFFSBDD_CHECKPOINT:-$WORK/ckpts/crossdocked_fullatom_cond.ckpt}"
export POCKETBENCH_ACCOUNT=pi_nzawia_uri_edu
export POCKETBENCH_CONDA_ENV="$WORK/envs/pocketbench"
export POCKETBENCH_PYTHON="$WORK/envs/pocketbench/bin/python"
# Unity's `gpu` partition includes Tesla M40 (sm_52), which CUDA 12 cannot target.
export POCKETBENCH_PARTITION="${POCKETBENCH_PARTITION:-gpu}"
export POCKETBENCH_GRES="${POCKETBENCH_GRES:-gpu:2080_ti:1}"

echo "DIFFSBDD_REPO=$DIFFSBDD_REPO"
[[ -d "$DIFFSBDD_REPO" ]]       || { echo "ERROR: DiffSBDD repo missing" >&2; exit 1; }
[[ -f "$DIFFSBDD_CHECKPOINT" ]] || { echo "ERROR: checkpoint missing: $DIFFSBDD_CHECKPOINT" >&2; exit 1; }

echo "=== pocket PDBs present ==="
ls data/raw/real50/*.pdb 2>/dev/null | wc -l

echo "=== dry-run ==="
bash cluster/submit.sh --config "$CFG" --job-name pb-diffsbdd-real47 --dry-run

echo "=== submit ==="
bash cluster/submit.sh --config "$CFG" --job-name pb-diffsbdd-real47
squeue -u "$USER"
echo JOB_C_SUBMITTED
echo
echo "After the array finishes:"
echo "  python cluster/merge_array_results.py --results-dir \"\$POCKETBENCH_RESULTS\" \\"
echo "    --run-id diffsbdd_real47 --expected-pockets 47"
echo "Then re-run the MW test with both models on real47:"
echo "  python analysis/mw_mechanism_both_models.py"
