#!/bin/bash
set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
SLURMDIR=$WORK/runs/slurm
LOGDIR=$WORK/runs/logs
mkdir -p "$SLURMDIR" "$LOGDIR"

SCRIPT=$SLURMDIR/pb-receptor-control.slurm
cat > "$SCRIPT" <<'EOS'
#!/usr/bin/env bash
#SBATCH --job-name=pb-receptor-control
#SBATCH --partition=cpu
#SBATCH -q long
#SBATCH --account=pi_nzawia_uri_edu
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=/work/pi_nzawia_uri_edu/pocketbench/runs/logs/%x_%j.out
#SBATCH --error=/work/pi_nzawia_uri_edu/pocketbench/runs/logs/%x_%j.err

set -euo pipefail
WORK=/work/pi_nzawia_uri_edu/pocketbench
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$WORK/envs/pocketbench"
cd "$WORK/repo"
source cluster/env.local.sh
export VINA_EXE="$WORK/bin/vina"
echo "host=$(hostname) vina=$VINA_EXE"
"$VINA_EXE" --version | head -2

python analysis/vina_receptor_control.py \
  --config configs/cluster/diffsbdd_real100_vina.yaml \
  --metrics-csv data/results/metrics_per_condition__rundiffsbdd_real100_vina.csv \
  --generations-root data/generations/run_diffsbdd_real100_vina \
  --vina-cpu 8 \
  --out-dir paper
echo RECEPTOR_CONTROL_DONE
EOS

echo "=== available partitions ==="
sinfo -h -o "%P %a %l" | head -25

# Pick a usable CPU partition; fall back to the GPU partition that already worked.
PART=""
for cand in cpu uri-cpu cpu-preempt uri-gpu; do
  if sinfo -h -p "$cand" -o '%P' 2>/dev/null | grep -q .; then
    PART="$cand"
    break
  fi
done
if [ -z "$PART" ]; then
  echo "ERROR: no usable partition found"
  exit 1
fi
echo "using partition: $PART"
sed -i "s#--partition=cpu#--partition=$PART#" "$SCRIPT"
grep -E 'partition|cpus-per-task|time' "$SCRIPT"
sbatch "$SCRIPT"
squeue -u "$USER"
echo SUBMITTED
