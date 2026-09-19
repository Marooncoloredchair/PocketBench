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

# pkg_resources comes from setuptools; lightning 1.8 needs it
pip install -U 'setuptools<81' wheel
# Re-assert DiffSBDD stack after conda openbabel may have shuffled packages
pip install 'pytorch-lightning==1.8.4' 'torchmetrics<1.0'

python - <<'PY'
from openbabel import openbabel
print("openbabel ok")
import pkg_resources
print("pkg_resources ok")
import pytorch_lightning as pl
print("pl", pl.__version__)
import torch
print("torch", torch.__version__)
import torch_scatter
print("torch_scatter ok")
import sys
sys.path.insert(0, "/work/pi_nzawia_uri_edu/pocketbench/DiffSBDD")
from lightning_modules import LigandPocketDDPM
print("LigandPocketDDPM import ok")
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
