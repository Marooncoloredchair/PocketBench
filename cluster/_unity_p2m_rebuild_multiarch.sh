#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
INNER=$WORK/runs/_p2m_rebuild_multiarch_inner.sh
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
print("device", torch.cuda.get_device_name(0), "cap", torch.cuda.get_device_capability(0))
PY

# Cover Unity gpu partition: 1080 Ti (6.1), V100 (7.0), T4/2080 (7.5), A100 (8.0), A40 (8.6), H100 (9.0)
export MAX_JOBS=4
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;9.0"
echo "TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

pip uninstall -y torch-cluster torch-scatter
pip install --no-cache-dir torch-scatter torch-cluster --no-build-isolation

python - <<'PY'
import torch
from torch_cluster import knn
from torch_scatter import scatter_add
x = torch.randn(50, 3, device='cuda')
y = torch.randn(10, 3, device='cuda')
idx = knn(x, y, k=6)
print("knn cuda ok", tuple(idx.shape), idx.device)
src = torch.randn(20, device='cuda')
index = torch.randint(0, 5, (20,), device='cuda')
out = scatter_add(src, index, dim=0, dim_size=5)
print("scatter cuda ok", tuple(out.shape), out.device)
print("MULTIARCH_OK")
PY
EOS

srun --partition=gpu -q short --account=pi_nzawia_uri_edu --time=01:30:00 \
  --gres=gpu:1 --cpus-per-task=4 --mem=16G \
  bash "$INNER"
echo MULTIARCH_DONE
