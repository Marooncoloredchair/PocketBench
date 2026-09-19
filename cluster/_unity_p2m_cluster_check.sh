#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
module load cuda/12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/p2m"
python - <<'PY'
import torch, torch_cluster, torch_scatter, torch_geometric
print("torch", torch.__version__)
print("pyg", torch_geometric.__version__)
from torch_geometric.nn import knn_graph
print("knn_graph", tuple(knn_graph(torch.randn(50, 3), 6).shape))
try:
    import torch_sparse
    print("torch_sparse", torch_sparse.__version__)
except Exception as e:
    print("torch_sparse missing (ok if knn_graph works):", type(e).__name__, e)
PY
( cd "$WORK/Pocket2Mol" && python -c "
from utils.transforms import *
import utils.misc, utils.reconstruct
from models.maskfill import MaskFillModelVN
print('Pocket2Mol imports OK')
" )
echo CLUSTER_IMPORT_OK
