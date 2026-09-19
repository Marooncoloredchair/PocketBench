#!/usr/bin/env bash
# Build the dedicated Pocket2Mol conda env (`$POCKETBENCH_WORK/envs/p2m`).
#
# Pocket2Mol pins a 2022 stack (env_cuda113.yml: python 3.8 / torch 1.10 /
# PyG 2.0.4) that cannot share an interpreter with the DiffSBDD + Vina stack in
# the `pocketbench` env. The adapter shells out to $POCKET2MOL_PYTHON, so this
# env is never activated by the SLURM scripts - it only has to be able to run
# Pocket2Mol's sample_for_pdb.py.
#
# Pins below are each the result of an observed failure, not preference:
#   torch-geometric==2.4.0  PyG >=2.5 made utils.subgraph private -> breaks
#                           Pocket2Mol utils/transforms.py
#   torch-cluster           utils/transforms.py needs knn_graph
#   numpy<2                 2022-era code uses APIs removed in NumPy 2
#   easydict                missing dep that broke the first Job B submission
#
# Usage:  srun --partition=gpu --gres=gpu:1 bash cluster/build_p2m_env.sh
#         (CUDA extensions must be compiled on a GPU node, not the login node)
# Verify: bash cluster/_unity_p2m_smoke.sh
set -euo pipefail

WORK="${POCKETBENCH_WORK:-/work/pi_nzawia_uri_edu/pocketbench}"
P2M_ENV="${P2M_ENV:-$WORK/envs/p2m}"
P2M_REPO="${POCKET2MOL_REPO:-$WORK/Pocket2Mol}"
CUDA_MODULE="${POCKETBENCH_CUDA_MODULE:-cuda/12.6}"
TORCH_INDEX="${POCKETBENCH_TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"

module load "$CUDA_MODULE"
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$P2M_ENV" ]; then
  conda create -y -p "$P2M_ENV" python=3.10
fi
conda activate "$P2M_ENV"
python -V

pip install -q --upgrade pip
pip install -q 'numpy>=1.23,<2' 'setuptools<81' wheel
pip install -q torch --index-url "$TORCH_INDEX"

# CUDA-kernel extensions: torch must already be importable, hence no isolation.
# These compile from source (~20-40 min total).
export MAX_JOBS="${MAX_JOBS:-4}"
export FORCE_CUDA=1
# Prefer the GPU actually present; fall back to a Unity-wide list.
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(python -c "import torch; print('%d.%d'%torch.cuda.get_device_capability(0))")}"
else
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.0;7.5;8.0;8.6;9.0}"
  echo "WARNING: no GPU visible; torch-cluster kernels may not run on compute nodes"
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.1;7.0;7.5;8.0;8.6;9.0}"
fi
# torch-sparse is not required by Pocket2Mol's sample path (knn_graph lives in
# torch-cluster). Skip it: the source build failed on this cluster.
pip install torch-scatter torch-cluster --no-build-isolation

pip install -q \
  'torch-geometric==2.4.0' \
  easydict \
  pyyaml \
  tqdm \
  biopython \
  rdkit \
  lmdb \
  scipy \
  pandas \
  scikit-learn \
  tensorboard \
  yacs

echo "=== import check (PocketBench side) ==="
python - <<'PY'
import numpy, torch, torch_geometric, torch_scatter, torch_cluster
import easydict, rdkit, lmdb, yaml, tqdm
from Bio.PDB import PDBParser
from torch_geometric.nn import knn_graph
print("numpy", numpy.__version__)
print("torch", torch.__version__)
print("pyg", torch_geometric.__version__)
print("knn_graph", tuple(knn_graph(torch.randn(50, 3), 6).shape))
PY

echo "=== import check (Pocket2Mol module graph) ==="
( cd "$P2M_REPO" && python -c "
from utils.transforms import *
import utils.misc, utils.reconstruct
from models.maskfill import MaskFillModelVN
print('Pocket2Mol imports OK')
" )

echo "P2M env ready: $P2M_ENV"
echo "Set POCKET2MOL_PYTHON=$P2M_ENV/bin/python in cluster/env.local.sh"
