#!/bin/bash
set +e
R=/work/pi_nzawia_uri_edu/pocketbench/runs
echo "=== all result CSVs mentioning real100_vina ==="
ls -l "$R"/results/ 2>/dev/null | grep -i real100_vina
echo "=== ANY csv in results dir (names) ==="
ls "$R"/results/ 2>/dev/null | sort | tail -60
echo "=== find any *real100_vina* anywhere under runs ==="
find "$R" -iname '*real100_vina*' 2>/dev/null | head -40
echo "=== a COMPLETED task out log (task 99) ==="
cat "$R"/logs/pb-real100-vina_62848168_99.out 2>/dev/null
echo "=== a COMPLETED task err tail (task 99) ==="
tail -n 15 "$R"/logs/pb-real100-vina_62848168_99.err 2>/dev/null
echo "=== grep 'wrote' in completed logs ==="
grep -h "wrote" "$R"/logs/pb-real100-vina_62848168_*.err 2>/dev/null | head -5
echo "=== P2M failed task 0 err ==="
tail -n 30 "$R"/logs/pb-p2m-real47-v2_62848171_0.err 2>/dev/null
echo CHECK_DONE
