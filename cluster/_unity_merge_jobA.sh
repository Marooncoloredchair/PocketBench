#!/bin/bash
set +e
WORK=/work/pi_nzawia_uri_edu/pocketbench
ENV_PREFIX=$WORK/envs/pocketbench
module load conda/latest 2>/dev/null
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate "$ENV_PREFIX" 2>/dev/null
cd "$WORK/repo"
RES="$WORK/repo/data/results"
echo "=== shard count ==="
ls "$RES"/metrics_per_condition__rundiffsbdd_real100_vina__p*.csv 2>/dev/null | wc -l
python cluster/merge_array_results.py --results-dir "$RES" --run-id diffsbdd_real100_vina --expected-pockets 100
echo "=== merged head ==="
head -3 "$RES"/metrics_per_condition__rundiffsbdd_real100_vina.csv
echo "=== has vina columns? ==="
head -1 "$RES"/metrics_per_condition__rundiffsbdd_real100_vina.csv | tr ',' '\n' | grep -in vina
echo MERGE_DONE
