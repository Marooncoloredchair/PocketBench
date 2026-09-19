#!/bin/bash
# Bootstrap PocketBench on Unity PI work + submit Job A (and Job B if ready).
# Run from WSL: bash /mnt/d/obsfu/sbdd-robust/cluster/_unity_bootstrap_submit.sh
set -euo pipefail

KEY="${HOME}/.ssh/unity_key"
HOST="terrell_osborne_uri_edu@login.unityhpc.org"
LOCAL_REPO="/mnt/d/obsfu/sbdd-robust"
WORK="/work/pi_nzawia_uri_edu/pocketbench"

echo "=== rsync repo + PDBs to Unity (may take a few minutes) ==="
ssh -o IdentitiesOnly=yes -o BatchMode=yes -i "$KEY" "$HOST" "mkdir -p '$WORK'/{repo,runs,ckpts,DiffSBDD,bin,envs}"

# Sync code (exclude heavy local gens/results)
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'data/generations/' \
  --exclude 'data/results/figures/' \
  --exclude 'data/pocketgym_cache/' \
  --exclude 'pocketbench_out/' \
  --exclude 'cluster_runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  -e "ssh -o IdentitiesOnly=yes -i $KEY" \
  "$LOCAL_REPO/" "$HOST:$WORK/repo/"

# PDBs needed for real100 + real50 (P2M)
rsync -az \
  -e "ssh -o IdentitiesOnly=yes -i $KEY" \
  "$LOCAL_REPO/data/raw/real100/" "$HOST:$WORK/repo/data/raw/real100/"
rsync -az \
  -e "ssh -o IdentitiesOnly=yes -i $KEY" \
  "$LOCAL_REPO/data/raw/real50/" "$HOST:$WORK/repo/data/raw/real50/" 2>/dev/null || true

# Seed P2M v2 metrics for resume
rsync -az \
  -e "ssh -o IdentitiesOnly=yes -i $KEY" \
  "$LOCAL_REPO/data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv" \
  "$HOST:$WORK/repo/data/results/" 2>/dev/null || true

# Upload local vina.exe is Windows — download Linux vina on cluster instead.
# Upload DiffSBDD checkpoint from local if present (faster than Zenodo).
if [[ -f /mnt/d/obsfu/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt ]]; then
  echo "=== uploading DiffSBDD checkpoint from D: ==="
  rsync -az --progress \
    -e "ssh -o IdentitiesOnly=yes -i $KEY" \
    /mnt/d/obsfu/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
    "$HOST:$WORK/ckpts/crossdocked_fullatom_cond.ckpt"
fi

echo "=== remote bootstrap (conda env, DiffSBDD, vina) ==="
ssh -o IdentitiesOnly=yes -o BatchMode=yes -i "$KEY" "$HOST" "bash -l -s" <<REMOTE
set -euo pipefail
WORK="$WORK"
cd "\$WORK"

# env.local.sh
cat > "\$WORK/repo/cluster/env.local.sh" <<'ENV'
export POCKETBENCH_SITE=unity
export POCKETBENCH_ROOT=/work/pi_nzawia_uri_edu/pocketbench/repo
export POCKETBENCH_WORK=/work/pi_nzawia_uri_edu/pocketbench/runs
export POCKETBENCH_RESULTS=\${POCKETBENCH_WORK}/results
export POCKETBENCH_GENERATIONS=\${POCKETBENCH_WORK}/generations
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
source "\$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

# Conda env (create once)
ENV_PREFIX="\$WORK/envs/pocketbench"
if [[ ! -x "\$ENV_PREFIX/bin/python" ]]; then
  echo "Creating conda env at \$ENV_PREFIX (long)..."
  conda create -y -p "\$ENV_PREFIX" python=3.11 pip
  # shellcheck disable=SC1091
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$ENV_PREFIX"
  pip install --upgrade pip
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install rdkit biopython pyyaml omegaconf pandas numpy scipy matplotlib seaborn tqdm einops
  pip install pytorch-lightning==1.8.4
  pip install meeko
  # torch-scatter matched to torch
  TORCH_VER=\$(python -c "import torch; print(torch.__version__.split('+')[0])")
  CUDA_TAG=\$(python -c "import torch; print('cu'+torch.version.cuda.replace('.','')) if torch.version.cuda else 'cpu'")
  pip install torch-scatter -f "https://data.pyg.org/whl/torch-\${TORCH_VER}+\${CUDA_TAG}.html" || pip install torch-scatter
  pip install -e "\$WORK/repo"
else
  echo "Conda env already exists: \$ENV_PREFIX"
  source "\$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "\$ENV_PREFIX"
fi

# DiffSBDD clone
if [[ ! -d "\$WORK/DiffSBDD/.git" ]]; then
  git clone --depth 1 https://github.com/arneschneuing/DiffSBDD.git "\$WORK/DiffSBDD"
fi

# Checkpoint
if [[ ! -f "\$WORK/ckpts/crossdocked_fullatom_cond.ckpt" ]]; then
  echo "Downloading DiffSBDD checkpoint from Zenodo..."
  mkdir -p "\$WORK/ckpts"
  curl -L --fail -o "\$WORK/ckpts/crossdocked_fullatom_cond.ckpt" \
    "https://zenodo.org/records/8183747/files/crossdocked_fullatom_cond.ckpt?download=1"
fi

# Vina Linux binary
if [[ ! -x "\$WORK/bin/vina" ]]; then
  echo "Downloading AutoDock Vina..."
  mkdir -p "\$WORK/bin" "\$WORK/tmp"
  cd "\$WORK/tmp"
  curl -L --fail -o vina.tgz \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64.gz" \
    || curl -L --fail -o vina.bin \
    "https://github.com/ccsb-scripps/AutoDock-Vina/releases/download/v1.2.5/vina_1.2.5_linux_x86_64"
  if [[ -f vina.tgz ]]; then
    gunzip -c vina.tgz > "\$WORK/bin/vina" || true
  fi
  if [[ -f vina.bin ]]; then
    cp vina.bin "\$WORK/bin/vina"
  fi
  chmod +x "\$WORK/bin/vina"
  "\$WORK/bin/vina" --version || "\$WORK/bin/vina" --help | head -3
fi

# Smoke imports
python - <<'PY'
import torch, meeko
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("meeko ok", getattr(meeko, "__version__", "?"))
PY
test -x "\$WORK/bin/vina"
test -f "\$WORK/ckpts/crossdocked_fullatom_cond.ckpt"
test -d "\$WORK/repo/data/raw/real100"
echo "BOOTSTRAP_OK"
REMOTE

echo "=== submit Job A ==="
ssh -o IdentitiesOnly=yes -o BatchMode=yes -i "$KEY" "$HOST" "bash -l -s" <<'REMOTE'
set -euo pipefail
cd /work/pi_nzawia_uri_edu/pocketbench/repo
source cluster/env.local.sh
mkdir -p "$POCKETBENCH_RESULTS" "$POCKETBENCH_GENERATIONS" "$POCKETBENCH_WORK/logs" "$POCKETBENCH_WORK/slurm"
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina --dry-run
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina
echo "=== squeue ==="
squeue -u "$USER" -o '%.18i %.12P %.20j %.8u %.2t %.10M %.6D %R' | head -40
REMOTE

echo "DONE_SUBMIT_A"
