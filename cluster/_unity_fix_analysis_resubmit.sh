#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
ENV_PREFIX=$WORK/envs/pocketbench
module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

scancel -u "$USER" || true

cd "$WORK/repo"
# shellcheck disable=SC1091
source cluster/env.local.sh
pip install -e . -q

# Smoke: DiffSBDD analysis import (cwd + PYTHONPATH = PocketBench would otherwise shadow)
cd "$WORK/DiffSBDD"
PYTHONPATH="$WORK/DiffSBDD" python - <<'PY'
from analysis.visualization import save_xyz_file
from lightning_modules import LigandPocketDDPM
print("DIFFSBDD_ANALYSIS_OK")
PY
cd "$WORK/repo"

rm -f "$POCKETBENCH_RESULTS"/metrics_per_condition__rundiffsbdd_real100_vina__p*.csv 2>/dev/null || true

bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina

python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir "$POCKETBENCH_RESULTS" \
  --run-id pocket2mol_real47_full_v2 || true
export POCKET2MOL_PYTHON="$POCKETBENCH_PYTHON"
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2
squeue -u "$USER"
echo RESUBMIT_DONE
