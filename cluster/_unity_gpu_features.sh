#!/bin/bash
set +e
echo "=== gpu partition node GPUs ==="
sinfo -p gpu -N -h -o "%N %G %f" | sort | uniq -c | sort -nr | head -40
echo "=== gres types ==="
sinfo -p gpu -h -o "%G" | tr ',' '\n' | sort | uniq -c | sort -nr
echo "=== features ==="
sinfo -p gpu -h -o "%f" | tr ',' '\n' | sort | uniq -c | sort -nr
echo FEATURES_DONE
