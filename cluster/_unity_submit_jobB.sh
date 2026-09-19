#!/bin/bash
# Submit Job B (Pocket2Mol real47 v2 resume) after a passing smoke test.
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
cd "$WORK/repo"

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/pocketbench"
source cluster/env.local.sh

# Persist P2M interpreter so submit.sh's re-source of env.local.sh cannot wipe it
if grep -q '^export POCKET2MOL_PYTHON=' cluster/env.local.sh; then
  sed -i 's|^export POCKET2MOL_PYTHON=.*|export POCKET2MOL_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/p2m/bin/python|' cluster/env.local.sh
else
  echo 'export POCKET2MOL_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/p2m/bin/python' >> cluster/env.local.sh
fi
grep '^export POCKET2MOL_PYTHON=' cluster/env.local.sh
if grep -q '^export POCKETBENCH_PARTITION=' cluster/env.local.sh; then
  sed -i 's|^export POCKETBENCH_PARTITION=.*|export POCKETBENCH_PARTITION=gpu|' cluster/env.local.sh
else
  echo 'export POCKETBENCH_PARTITION=gpu' >> cluster/env.local.sh
fi
# Re-source so submit.sh sees the patched values
source cluster/env.local.sh
export POCKET2MOL_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/p2m/bin/python
export POCKET2MOL_REPO="$WORK/Pocket2Mol"
export POCKET2MOL_CHECKPOINT="$WORK/ckpts/pretrained_Pocket2Mol.pt"
# Turing sm_75; excludes Tesla M40 (sm_52) which CUDA 12 cannot target
export POCKETBENCH_GRES=gpu:2080_ti:1
export POCKETBENCH_PARTITION=gpu
export POCKETBENCH_ACCOUNT=pi_nzawia_uri_edu
export POCKETBENCH_CONDA_ENV="$WORK/envs/pocketbench"
export POCKETBENCH_PYTHON="$WORK/envs/pocketbench/bin/python"

echo "POCKET2MOL_PYTHON=$POCKET2MOL_PYTHON"
"$POCKET2MOL_PYTHON" -c "import easydict, torch_geometric, torch_cluster; print('p2m ok', torch_geometric.__version__)"

echo "=== seed shards from existing 19-pocket v2 panel ==="
python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir "${POCKETBENCH_RESULTS}" \
  --run-id pocket2mol_real47_full_v2 || true

echo "=== dry-run ==="
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --job-name pb-p2m-real47-v2 --dry-run

echo "=== submit ==="
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --job-name pb-p2m-real47-v2
squeue -u "$USER"
echo JOB_B_SUBMITTED
