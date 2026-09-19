#!/bin/bash
# Read-only prereq probe for the DiffSBDD real47 submission. Changes nothing.
W=/work/pi_nzawia_uri_edu/pocketbench

echo "==WORK=="
ls -d "$W" 2>/dev/null || echo "MISSING: $W"

echo "==REPO=="
ls -d "$W/repo" 2>/dev/null || echo "MISSING repo"

echo "==CONFIGS=="
ls -l "$W/repo/configs/cluster/diffsbdd_real47.yaml" 2>/dev/null || echo "no cluster/diffsbdd_real47.yaml"
ls -l "$W/repo/configs/experiments/diffsbdd_real47.yaml" 2>/dev/null || echo "no experiments/diffsbdd_real47.yaml"

echo "==DIFFSBDD REPO/CKPT=="
ls -d "$W/DiffSBDD" 2>/dev/null || echo "no DiffSBDD repo"
ls -l "$W"/ckpts/*.ckpt 2>/dev/null || echo "no ckpts"

echo "==ENV LOCAL=="
ls -l "$W/repo/cluster/env.local.sh" 2>/dev/null || echo "no env.local.sh"

echo "==PDBS real50=="
ls "$W/repo/data/raw/real50/"*.pdb 2>/dev/null | wc -l

echo "==GENERATIONS DIRS=="
ls "$W/repo/data/generations" 2>/dev/null | head -20

echo "==PER-MOLECULE SIDECARS ALREADY PRESENT=="
ls -l "$W/repo/data/results/"per_molecule__*.csv 2>/dev/null || echo "none"

echo "==EXISTING RESULT PANELS=="
ls "$W/repo/data/results/"metrics_per_condition__*.csv 2>/dev/null | head -20

echo "==GIT HEAD=="
git -C "$W/repo" rev-parse --short HEAD 2>/dev/null
git -C "$W/repo" remote -v 2>/dev/null | head -4

echo "==RECENT JOB TIMING (for wall-clock estimate)=="
sacct -u "$USER" --starttime now-14days \
  --format=JobID%18,JobName%22,Partition%12,Elapsed,State%14,AllocTRES%38 \
  2>/dev/null | grep -Ei 'diffsbdd|p2m|pb-' | head -30

echo "==PARTITION/GPU AVAIL=="
sinfo -o "%20P %10a %10l %6D %10T %20G" 2>/dev/null | grep -Ei 'gpu' | head -12
