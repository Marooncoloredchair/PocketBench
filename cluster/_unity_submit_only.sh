#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
ENV_PREFIX=$WORK/envs/pocketbench
module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_PREFIX"

pip install -q gemmi

mkdir -p "$WORK/bin" "$WORK/tmp"
if [[ ! -x $WORK/bin/vina ]]; then
  curl -L --fail -o "$WORK/bin/vina" \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
  chmod +x "$WORK/bin/vina"
fi

python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
import torch_scatter
print("torch_scatter ok")
import meeko
print("meeko ok", getattr(meeko, "__version__", "?"))
PY
"$WORK/bin/vina" --version || "$WORK/bin/vina" --help | head -5
test -f "$WORK/ckpts/crossdocked_fullatom_cond.ckpt"
test -x "$WORK/bin/vina"

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
