#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
ENV_PREFIX=$WORK/envs/pocketbench
module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
# Prefer no-build-isolation so setup.py can import torch
pip install torch-scatter --no-build-isolation \
  || pip install torch-scatter -f https://data.pyg.org/whl/torch-2.6.0+cu126.html \
  || echo "WARN: torch-scatter install failed; DiffSBDD may still need it"

pip install -e "$WORK/repo"
pip install meeko 'pytorch-lightning==1.8.4' rdkit biopython pyyaml omegaconf pandas numpy scipy tqdm einops || true

if [[ ! -x $WORK/bin/vina ]]; then
  mkdir -p "$WORK/bin" "$WORK/tmp"
  curl -L --fail -o "$WORK/bin/vina" \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
  chmod +x "$WORK/bin/vina"
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import torch_scatter
    print("torch_scatter ok")
except Exception as e:
    print("torch_scatter MISSING", e)
import meeko
print("meeko ok")
PY
"$WORK/bin/vina" --version || true
test -f "$WORK/ckpts/crossdocked_fullatom_cond.ckpt"

cd "$WORK/repo"
# shellcheck disable=SC1091
source cluster/env.local.sh
mkdir -p "$POCKETBENCH_RESULTS" "$POCKETBENCH_GENERATIONS" "$POCKETBENCH_WORK/logs" "$POCKETBENCH_WORK/slurm"

bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina --dry-run
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina

python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir "$POCKETBENCH_RESULTS" \
  --run-id pocket2mol_real47_full_v2 || true
export POCKET2MOL_PYTHON="$POCKETBENCH_PYTHON"
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2 --dry-run
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2
squeue -u "$USER"
echo SUBMIT_DONE
