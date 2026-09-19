#!/bin/bash
# WSL entry: sync + bootstrap + submit Job A (and Job B if P2M ready).
set -euo pipefail
KEY="${HOME}/.ssh/unity_key"
HOST="terrell_osborne_uri_edu@login.unityhpc.org"
LOCAL="/mnt/d/obsfu/sbdd-robust"
WORK="/work/pi_nzawia_uri_edu/pocketbench"
SSH=(ssh -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=60 -i "$KEY")
RSYNC_SSH="ssh -o IdentitiesOnly=yes -o BatchMode=yes -i $KEY"

# LF-fix helper scripts before upload
python3 - <<'PY'
from pathlib import Path
for name in ("_unity_remote_bootstrap.sh", "_unity_bootstrap_submit.sh", "submit.sh", "run_array_task.sh", "env.example.sh"):
    p = Path("/mnt/d/obsfu/sbdd-robust/cluster") / name
    if p.is_file():
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
PY

echo "=== mkdir on Unity ==="
"${SSH[@]}" "$HOST" "mkdir -p $WORK/{repo,runs,ckpts,DiffSBDD,Pocket2Mol,bin,envs,tmp}"

echo "=== rsync PocketBench repo ==="
rsync -az --delete \
  --exclude '.git/' \
  --exclude 'data/generations/' \
  --exclude 'data/results/figures/' \
  --exclude 'data/pocketgym_cache/' \
  --exclude 'pocketbench_out/' \
  --exclude 'cluster_runs/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  -e "$RSYNC_SSH" \
  "$LOCAL/" "$HOST:$WORK/repo/"

echo "=== rsync PDBs ==="
rsync -az -e "$RSYNC_SSH" "$LOCAL/data/raw/real100/" "$HOST:$WORK/repo/data/raw/real100/"
rsync -az -e "$RSYNC_SSH" "$LOCAL/data/raw/real50/" "$HOST:$WORK/repo/data/raw/real50/"

echo "=== rsync P2M v2 metrics (resume seed) ==="
rsync -az -e "$RSYNC_SSH" \
  "$LOCAL/data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv" \
  "$HOST:$WORK/repo/data/results/"

echo "=== rsync DiffSBDD checkpoint ==="
rsync -az --progress -e "$RSYNC_SSH" \
  /mnt/d/obsfu/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt \
  "$HOST:$WORK/ckpts/crossdocked_fullatom_cond.ckpt"

echo "=== rsync Pocket2Mol checkpoint + minimal repo if present ==="
rsync -az --progress -e "$RSYNC_SSH" \
  /mnt/d/obsfu/Pocket2Mol/ckpt/pretrained_Pocket2Mol.pt \
  "$HOST:$WORK/ckpts/pretrained_Pocket2Mol.pt"
# Sync Pocket2Mol code (needed for Job B)
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude '*.pyc' --exclude 'ckpt/*.pt' \
  -e "$RSYNC_SSH" \
  /mnt/d/obsfu/Pocket2Mol/ "$HOST:$WORK/Pocket2Mol/"

echo "=== rsync DiffSBDD code ==="
rsync -az --delete \
  --exclude '.git/' --exclude '__pycache__/' --exclude 'checkpoints/*.ckpt' \
  -e "$RSYNC_SSH" \
  /mnt/d/obsfu/DiffSBDD/ "$HOST:$WORK/DiffSBDD/"

echo "=== remote bootstrap ==="
"${SSH[@]}" "$HOST" "bash -l $WORK/repo/cluster/_unity_remote_bootstrap.sh"

echo "=== submit Job A ==="
"${SSH[@]}" "$HOST" "bash -l -c '
set -euo pipefail
cd /work/pi_nzawia_uri_edu/pocketbench/repo
source cluster/env.local.sh
mkdir -p \"\$POCKETBENCH_RESULTS\" \"\$POCKETBENCH_GENERATIONS\" \"\$POCKETBENCH_WORK/logs\" \"\$POCKETBENCH_WORK/slurm\"
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina --dry-run
bash cluster/submit.sh --config configs/cluster/diffsbdd_real100_vina.yaml --job-name pb-real100-vina
'"

echo "=== seed + submit Job B (Pocket2Mol) if bridge env usable ==="
"${SSH[@]}" "$HOST" "bash -l -c '
set -euo pipefail
cd /work/pi_nzawia_uri_edu/pocketbench/repo
source cluster/env.local.sh
# Job B uses same pocketbench python for orchestrator; Pocket2Mol inference needs POCKET2MOL_PYTHON.
# If unset, adapter uses orchestrator python — may fail on old P2M deps. Try submit; failures visible in logs.
python cluster/seed_array_shards_from_panel.py \
  --panel data/results/metrics_per_condition__runpocket2mol_real47_full_v2.csv \
  --config configs/cluster/pocket2mol_real47_v2_resume.yaml \
  --results-dir \"\$POCKETBENCH_RESULTS\" \
  --run-id pocket2mol_real47_full_v2 || true
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2 --dry-run
bash cluster/submit.sh --config configs/cluster/pocket2mol_real47_v2_resume.yaml --job-name pb-p2m-real47-v2
squeue -u \"\$USER\"
'"

echo "ALL_DONE"
