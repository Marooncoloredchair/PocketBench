#!/bin/bash
set +e
WORK=/work/pi_nzawia_uri_edu/pocketbench
module load conda/latest 2>/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate "$WORK/envs/pocketbench" 2>/dev/null
cd "$WORK/repo"
source cluster/env.local.sh 2>/dev/null

echo "############ PART 1: receptor-control dry run ############"
python analysis/vina_receptor_control.py \
  --config configs/cluster/diffsbdd_real100_vina.yaml \
  --metrics-csv data/results/metrics_per_condition__rundiffsbdd_real100_vina.csv \
  --generations-root data/generations/run_diffsbdd_real100_vina \
  --dry-run 2>&1 | tail -25

echo "############ PART 2: Pocket2Mol requirements ############"
P2M=$WORK/Pocket2Mol
ls "$P2M"
echo "--- requirements.txt ---"
cat "$P2M"/requirements.txt 2>/dev/null || echo "(no requirements.txt)"
echo "--- env yml files ---"
for f in "$P2M"/*.yml "$P2M"/*.yaml; do [ -f "$f" ] && { echo "== $f =="; cat "$f"; }; done
echo "--- imports used by P2M entrypoints ---"
grep -rhoE '^(import|from) [a-zA-Z0-9_\.]+' "$P2M"/sample_for_pdb.py "$P2M"/models/*.py "$P2M"/utils/*.py 2>/dev/null \
  | awk '{print $2}' | cut -d. -f1 | sort -u
echo "--- existing conda envs ---"
conda env list
echo CHECK_DONE
