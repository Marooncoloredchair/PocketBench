#!/bin/bash
# Finish Unity env after partial bootstrap; submit Job A (+ Job B if possible).
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
ENV_PREFIX=$WORK/envs/pocketbench

module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

# Reinstall a torch build that has prebuilt torch-scatter wheels
pip uninstall -y torch torchvision torchaudio torch-scatter 2>/dev/null || true
pip install --upgrade pip
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu124
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu124.html
pip install 'pytorch-lightning==1.8.4' meeko rdkit biopython pyyaml omegaconf pandas numpy scipy matplotlib seaborn tqdm einops
pip install -e "$WORK/repo"

# Vina
if [[ ! -x $WORK/bin/vina ]]; then
  mkdir -p "$WORK/bin" "$WORK/tmp"
  curl -L --fail -o "$WORK/tmp/vina.gz" \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64.gz"
  gunzip -c "$WORK/tmp/vina.gz" > "$WORK/bin/vina"
  chmod +x "$WORK/bin/vina"
fi

python - <<'PY'
import torch
import torch_scatter
import meeko
print("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.version.cuda)
print("torch_scatter ok")
print("meeko", getattr(meeko, "__version__", "?"))
PY
"$WORK/bin/vina" --version || true
test -f "$WORK/ckpts/crossdocked_fullatom_cond.ckpt"
echo "REPAIR_OK"

# Ensure env.local exists
test -f "$WORK/repo/cluster/env.local.sh"

cd "$WORK/repo"
# shellcheck disable=SC1091
source cluster/env.local.sh
mkdir -p "$POCKETBENCH_RESULTS" "$POCKETBENCH_GENERATIONS" "$POCKETBENCH_WORK/logs" "$POCKETBENCH_WORK/slurm"

echo "=== Job A dry-run ==="
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina --dry-run
echo "=== Job A submit ==="
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina

echo "=== Job B seed + submit ==="
python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir "$POCKETBENCH_RESULTS" \
  --run-id pocket2mol_real47_full_v2 || true
# Point P2M python at pocketbench for now (may need older env later)
export POCKET2MOL_PYTHON="$POCKETBENCH_PYTHON"
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2 --dry-run
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2

echo "=== squeue ==="
squeue -u "$USER"
echo "SUBMIT_DONE"
