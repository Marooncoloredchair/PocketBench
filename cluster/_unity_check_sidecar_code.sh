#!/bin/bash
# Read-only: does the CLUSTER checkout contain the per-molecule sidecar code?
# If not, the real47 rerun would archive nothing and waste the allocation.
R=/work/pi_nzawia_uri_edu/pocketbench/repo

echo "==is it a git repo=="
if [ -d "$R/.git" ]; then
  echo "yes"
  command -v git >/dev/null && git -C "$R" rev-parse --short HEAD || echo "git not on PATH"
  command -v git >/dev/null && git -C "$R" status --porcelain | head -20
else
  echo "NO .git directory - repo was copied, not cloned"
fi

echo "==cli.py has per_molecule sidecar?=="
grep -n "per_molecule_records\|per_molecule__run" "$R/sbdd_robust/cli.py" 2>/dev/null || echo "ABSENT from cli.py"

echo "==chemistry.py has per_molecule_records?=="
grep -n "def per_molecule_records" "$R/sbdd_robust/metrics/chemistry.py" 2>/dev/null || echo "ABSENT from chemistry.py"

echo "==cli.py mtime / size=="
ls -l "$R/sbdd_robust/cli.py" "$R/sbdd_robust/metrics/chemistry.py" 2>/dev/null

echo "==diffsbdd adapter: does it keep the sdf?=="
grep -n "_gen.sdf\|def cleanup" "$R/sbdd_robust/models/diffsbdd_adapter.py" 2>/dev/null

echo "==n_samples in cluster diffsbdd_real47.yaml=="
grep -n "n_samples\|run_id\|type: diffsbdd" "$R/configs/experiments/diffsbdd_real47.yaml" 2>/dev/null

echo "==git on PATH?=="
command -v git || echo "no git"

echo "==full sacct history (no grep)=="
sacct -u "$USER" --starttime now-60days --format=JobID%16,JobName%24,Elapsed,State%12 2>/dev/null | head -40
