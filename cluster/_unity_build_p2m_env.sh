#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
P2M_ENV=$WORK/envs/p2m
P2M_REPO=$WORK/Pocket2Mol

module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$P2M_ENV" ]; then
  echo "=== creating dedicated Pocket2Mol env (python 3.10) ==="
  conda create -y -p "$P2M_ENV" python=3.10
fi
conda activate "$P2M_ENV"
python -V

echo "=== core scientific stack (numpy<2 for P2M-era code) ==="
pip install -q --upgrade pip
pip install -q 'numpy>=1.23,<2' 'setuptools<81' wheel

echo "=== torch (cu126 to match cluster CUDA 12.6) ==="
pip install -q torch --index-url https://download.pytorch.org/whl/cu126

echo "=== torch_scatter (needs torch present; no build isolation) ==="
pip install -q torch-scatter --no-build-isolation

echo "=== Pocket2Mol declared deps (from env_cuda113.yml) ==="
pip install -q \
  torch-geometric \
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

echo "=== import verification ==="
python - <<'PY'
import numpy, torch, yaml, tqdm, easydict, rdkit, lmdb
import torch_scatter, torch_geometric
from Bio.PDB import PDBParser  # biopython
print("numpy", numpy.__version__)
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
print("torch_geometric", torch_geometric.__version__)
print("easydict/rdkit/lmdb/torch_scatter/biopython OK")
PY

echo "=== P2M checkpoint present? ==="
ls -l "$WORK/ckpts/pretrained_Pocket2Mol.pt"
echo P2M_ENV_BUILD_DONE
