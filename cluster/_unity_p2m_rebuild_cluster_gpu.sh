#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
INNER=$WORK/runs/_p2m_rebuild_cluster_inner.sh
mkdir -p "$WORK/runs"

cat > "$INNER" <<'EOS'
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
module load cuda/12.6
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/p2m"

echo "host=$(hostname)"
nvidia-smi -L
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY

# Compile against the GPU actually present on this node.
CAP=$(python -c "import torch; print('%d.%d'%torch.cuda.get_device_capability(0))")
echo "compiling torch-cluster for sm_${CAP/./}"
export MAX_JOBS=4
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="$CAP"
pip uninstall -y torch-cluster
pip install --no-cache-dir torch-cluster --no-build-isolation

python - <<'PY'
import torch
from torch_cluster import knn
x = torch.randn(50, 3, device='cuda')
y = torch.randn(10, 3, device='cuda')
idx = knn(x, y, k=6)
print("knn cuda ok", tuple(idx.shape), idx.device)
print("REBUILD_OK")
PY
EOS

srun --partition=gpu -q short --account=pi_nzawia_uri_edu --time=00:50:00 \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  bash "$INNER"
echo REBUILD_DONE
