#!/bin/bash
set -uo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
P2M_ENV=$WORK/envs/p2m
P2M_REPO=$WORK/Pocket2Mol

module load cuda/12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P2M_ENV"

# Pocket2Mol targets PyG 2.0.4; PyG >=2.5 renamed utils submodules to _private form.
# Probe candidates newest-first and keep the first that imports P2M's module graph.
PICKED=""
for V in 2.4.0 2.3.1 2.2.0 2.0.4; do
  echo "=== trying torch-geometric==$V ==="
  pip install -q "torch-geometric==$V" 2>&1 | tail -3
  if ( cd "$P2M_REPO" && python -c "
import sys
sys.argv=['probe']
from utils.transforms import *
import utils.misc, utils.reconstruct
from models.maskfill import MaskFillModelVN
print('IMPORT_OK')
" 2>&1 | tail -6 | grep -q IMPORT_OK ); then
    echo "PYG_PICKED=$V"
    PICKED="$V"
    break
  else
    echo "--- $V failed, error was: ---"
    ( cd "$P2M_REPO" && python -c "
from utils.transforms import *
from models.maskfill import MaskFillModelVN
" 2>&1 | tail -5 )
  fi
done

if [ -z "$PICKED" ]; then
  echo "PYG_PROBE_FAILED"
  exit 1
fi
python -c "import torch_geometric,torch;print('final pyg',torch_geometric.__version__,'torch',torch.__version__)"
echo PROBE_DONE
