#!/bin/bash
# Submit DiffSBDD on the real47 panel (Job C), WITH per-molecule archiving.
#
# WHY THIS RUN EXISTS
# -------------------
# run1778615375 is DiffSBDD on the *same* 47 pockets and the *same* 5 conditions
# as runpocket2mol_real47_full_v2 (pocket-set overlap verified at 47/47). The two
# summary panels are perfectly matched. What run1778615375 lacks is per-molecule
# data: it was executed on Colab, no generations were archived, and the
# per-molecule sidecar did not exist in the code at that time. So molecular
# weight cannot be recovered from it at all.
#
# That forced the MW mechanism test to substitute DiffSBDD real100_vina, which
# shares ZERO pockets with the Pocket2Mol real47 panel. The "opposite MW
# directions" result is therefore confounded with pocket set.
#
# This run regenerates DiffSBDD on the exact real47 panel with the per-molecule
# sidecar active, producing data/results/per_molecule__rundiffsbdd_real47.csv so
# the cross-model MW/QED/SA comparison becomes pocket-paired.
#
# n_samples stays at 20 (the configured value, matching run1778615375). It is NOT
# lowered to Pocket2Mol's 12: the two samplers realize different counts anyway
# (DiffSBDD 19-20, Pocket2Mol 0-21 median 13), and unequal n affects the
# precision of each per-pocket mean, not its expectation.
#
# Prereq already satisfied: sbdd_robust/cli.py and sbdd_robust/metrics/chemistry.py
# were synced to the cluster so per_molecule_records() exists (originals kept as
# *.bak_presidecar).
#
# Run from a Unity login node:  bash cluster/_unity_submit_diffsbdd_real47.sh
#
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

# --- preflight: refuse to burn the allocation if archiving is not wired up ---
grep -q "per_molecule_records" sbdd_robust/cli.py \
  || { echo "ABORT: cli.py lacks the per-molecule sidecar; run would archive nothing." >&2; exit 1; }
grep -q "def per_molecule_records" sbdd_robust/metrics/chemistry.py \
  || { echo "ABORT: chemistry.py lacks per_molecule_records()." >&2; exit 1; }
[[ -d "$DIFFSBDD_REPO" ]]       || { echo "ABORT: DiffSBDD repo missing" >&2; exit 1; }
[[ -f "$DIFFSBDD_CHECKPOINT" ]] || { echo "ABORT: checkpoint missing" >&2; exit 1; }
echo "preflight OK (sidecar wired, repo + checkpoint present)"

echo "=== submit ==="
bash cluster/submit.sh --config "$CFG" --job-name pb-diffsbdd-real47
squeue -u "$USER"
echo JOB_C_SUBMITTED
echo
echo "After the array finishes:"
echo "  python cluster/merge_array_results.py --results-dir \"\$POCKETBENCH_RESULTS\" \\"
echo "    --run-id diffsbdd_real47 --expected-pockets 47"
echo "Then re-run the MW test with both models on the same 47 pockets:"
echo "  python analysis/mw_mechanism_both_models.py"
