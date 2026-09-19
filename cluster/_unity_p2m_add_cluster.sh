#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
P2M_ENV=$WORK/envs/p2m

module load cuda/12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$P2M_ENV"

# Compiling CUDA kernels for these takes a while; limit jobs so the login node stays usable.
export MAX_JOBS=4
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"

echo "=== torch-cluster (knn_graph used by P2M transforms) ==="
pip install -v torch-cluster --no-build-isolation 2>&1 | tail -5

echo "=== torch-sparse (pulled in by pyg pooling paths) ==="
pip install -v torch-sparse --no-build-isolation 2>&1 | tail -5

python -c "
import torch, torch_cluster, torch_scatter, torch_sparse, torch_geometric
print('torch', torch.__version__)
print('pyg', torch_geometric.__version__)
print('cluster/scatter/sparse all import OK')
from torch_geometric.nn import knn_graph
x = torch.randn(50,3)
print('knn_graph cpu ok', knn_graph(x, 6).shape)
"
echo P2M_CLUSTER_DEPS_DONE
