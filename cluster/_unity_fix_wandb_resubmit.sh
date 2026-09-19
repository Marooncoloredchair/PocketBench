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

# DiffSBDD runtime deps (match local working env)
pip install -q 'wandb==0.13.1' 'protobuf>=3.20,<4' networkx imageio pyyaml

python - <<'PY'
from openbabel import openbabel
import pkg_resources  # noqa: F401
import pytorch_lightning as pl
import torch
import torch_scatter
import wandb
import sys
sys.path.insert(0, "/work/pi_nzawia_uri_edu/pocketbench/DiffSBDD")
from lightning_modules import LigandPocketDDPM
print("IMPORTS_OK", "pl", pl.__version__, "torch", torch.__version__, "wandb", wandb.__version__)
PY

cd "$WORK/repo"
# shellcheck disable=SC1091
source cluster/env.local.sh
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
