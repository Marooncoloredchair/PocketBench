#!/bin/bash
# Runs ON Unity login node after repo rsync.
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
cd "$WORK"

cat > "$WORK/repo/cluster/env.local.sh" <<'ENV'
export POCKETBENCH_SITE=unity
export POCKETBENCH_ROOT=/work/pi_nzawia_uri_edu/pocketbench/repo
export POCKETBENCH_WORK=/work/pi_nzawia_uri_edu/pocketbench/runs
export POCKETBENCH_RESULTS=${POCKETBENCH_WORK}/results
export POCKETBENCH_GENERATIONS=${POCKETBENCH_WORK}/generations
export POCKETBENCH_CONDA_ENV=pocketbench
export POCKETBENCH_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/pocketbench/bin/python
export POCKETBENCH_MODULES="cuda/12.6"
export DIFFSBDD_REPO=/work/pi_nzawia_uri_edu/pocketbench/DiffSBDD
export DIFFSBDD_CHECKPOINT=/work/pi_nzawia_uri_edu/pocketbench/ckpts/crossdocked_fullatom_cond.ckpt
export DIFFSBDD_PYTHON=/work/pi_nzawia_uri_edu/pocketbench/envs/pocketbench/bin/python
export POCKET2MOL_REPO=/work/pi_nzawia_uri_edu/pocketbench/Pocket2Mol
export POCKET2MOL_CHECKPOINT=/work/pi_nzawia_uri_edu/pocketbench/ckpts/pretrained_Pocket2Mol.pt
export POCKET2MOL_PYTHON=
export POCKETBENCH_PARTITION=uri-gpu
export POCKETBENCH_QOS=long
export POCKETBENCH_ACCOUNT=pi_nzawia_uri_edu
export POCKETBENCH_TIME=1-00:00:00
export POCKETBENCH_GPUS=1
export POCKETBENCH_CPUS=4
export POCKETBENCH_MEM=32G
export VINA_EXE=/work/pi_nzawia_uri_edu/pocketbench/bin/vina
ENV

module load cuda/12.6
module load conda/latest
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

ENV_PREFIX="$WORK/envs/pocketbench"
if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  echo "Creating conda env at $ENV_PREFIX ..."
  conda create -y -p "$ENV_PREFIX" python=3.11 pip
  conda activate "$ENV_PREFIX"
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install rdkit biopython pyyaml omegaconf pandas numpy scipy matplotlib seaborn tqdm einops
  pip install 'pytorch-lightning==1.8.4'
  pip install meeko
  TORCH_VER=$(python -c "import torch; print(torch.__version__.split('+')[0])")
  pip install torch-scatter -f "https://data.pyg.org/whl/torch-${TORCH_VER}+cu126.html" || pip install torch-scatter
  pip install -e "$WORK/repo"
else
  echo "Conda env exists: $ENV_PREFIX"
  conda activate "$ENV_PREFIX"
fi

if [[ ! -d "$WORK/DiffSBDD/.git" ]]; then
  git clone --depth 1 https://github.com/arneschneuing/DiffSBDD.git "$WORK/DiffSBDD"
fi

if [[ ! -f "$WORK/ckpts/crossdocked_fullatom_cond.ckpt" ]]; then
  echo "Downloading DiffSBDD checkpoint (Zenodo)..."
  mkdir -p "$WORK/ckpts"
  curl -L --fail -o "$WORK/ckpts/crossdocked_fullatom_cond.ckpt" \
    "https://zenodo.org/records/8183747/files/crossdocked_fullatom_cond.ckpt?download=1"
fi

if [[ ! -x "$WORK/bin/vina" ]]; then
  echo "Downloading AutoDock Vina Linux binary..."
  mkdir -p "$WORK/bin" "$WORK/tmp"
  curl -L --fail -o "$WORK/tmp/vina.gz" \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64.gz"
  gunzip -c "$WORK/tmp/vina.gz" > "$WORK/bin/vina"
  chmod +x "$WORK/bin/vina"
fi
"$WORK/bin/vina" --version || true

python - <<'PY'
import torch, meeko
print("torch", torch.__version__, "cuda_built", torch.version.cuda, "cuda_avail", torch.cuda.is_available())
print("meeko", getattr(meeko, "__version__", "?"))
PY

test -f "$WORK/ckpts/crossdocked_fullatom_cond.ckpt"
test -d "$WORK/repo/data/raw/real100"
echo "BOOTSTRAP_OK"
