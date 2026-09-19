#!/bin/bash
set +e
R=/work/pi_nzawia_uri_edu/pocketbench/runs
G=/work/pi_nzawia_uri_edu/pocketbench/repo/data/generations/run_diffsbdd_real100_vina
echo "=== squeue (live) ==="
squeue -u "$USER" 2>/dev/null
echo "=== sacct 6 days pb jobs ==="
sacct -u "$USER" --starttime now-6days -X -n -o JobID,JobName%22,State,Elapsed,End 2>/dev/null | grep -Ei 'pb-real|pb-p2m'
echo "=== Job A result shards (count) ==="
ls "$R"/results/metrics_per_condition__rundiffsbdd_real100_vina__p*.csv 2>/dev/null | wc -l
echo "=== Job A merged csv ==="
ls -l "$R"/results/metrics_per_condition__rundiffsbdd_real100_vina.csv 2>/dev/null || echo "(no merged)"
echo "=== P2M result shards (count) ==="
ls "$R"/results/metrics_per_condition__runpocket2mol_real47_full_v2__p*.csv 2>/dev/null | wc -l
echo "=== Gen: pocket dirs, sdf count ==="
ls -d "$G"/*/ 2>/dev/null | wc -l
find "$G" -name '*_gen.sdf' 2>/dev/null | wc -l
echo "=== per-condition sdf breakdown ==="
find "$G" -name '*_gen.sdf' 2>/dev/null | sed -E 's#.*/([^/]+)/[^/]+_gen.sdf#\1#' | sort | uniq -c
echo "=== newest 5 A logs ==="
ls -t "$R"/logs/pb-real100-vina_*.err 2>/dev/null | head -5
echo "=== tail newest A err ==="
NE=$(ls -t "$R"/logs/pb-real100-vina_*.err 2>/dev/null | head -1)
[ -n "$NE" ] && { echo "FILE=$NE"; tail -n 25 "$NE"; }
echo "=== any shard content sample ==="
SH=$(ls -t "$R"/results/metrics_per_condition__rundiffsbdd_real100_vina__p*.csv 2>/dev/null | head -1)
[ -n "$SH" ] && { echo "SHARD=$SH"; head -3 "$SH"; } || echo "(no diffsbdd_real100_vina shards)"
echo CHECK_DONE
