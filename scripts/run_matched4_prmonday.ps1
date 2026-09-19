# Pre-Monday matched 4-pocket reruns (~2-4 hours total on GPU).
# 1) Pocket2Mol: 1B0R + 1I4F x 5 conditions (fixes missing originals)
# 2) DiffSBDD: 4 pockets x 5 conditions with face_peel (needs DIFFSBDD_* env vars)
#
# After both finish:
#   python analysis/merge_matched4_metrics.py
#   python analysis/matched_isr_panel.py --pocket2mol-metrics data/results/metrics_per_condition__runpocket2mol_isr_matched4_merged.csv --diffsbdd-metrics data/results/metrics_per_condition__rundiffsbdd_isr_matched4_merged.csv
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$Py311 = "D:\Miniforge\envs\sbdd_py311\python.exe"

Set-Location $RepoRoot

Write-Host "=== Pocket2Mol matched4 rerun (1B0R, 1I4F) ==="
& $Py311 -m pip install -e $RepoRoot -q
$env:PYTHONPATH = "D:\obsfu\Pocket2Mol"
$env:PYTHONUNBUFFERED = "1"
$prev = $ErrorActionPreference; $ErrorActionPreference = "Continue"
& $Py311 -u -m sbdd_robust run --config configs/experiments/pocket2mol_isr_matched4_rerun.yaml --resume
$ErrorActionPreference = $prev
if ($LASTEXITCODE -ne 0) { Write-Error "Pocket2Mol rerun failed" }

Write-Host ""
Write-Host "=== DiffSBDD matched4 (4 pockets + face_peel) ==="
if (-not $env:DIFFSBDD_REPO) { Write-Warning "DIFFSBDD_REPO not set - skip DiffSBDD (use Colab stress CSV filter instead)" }
else {
    & $Py311 -u -m sbdd_robust run --config configs/experiments/diffsbdd_isr_matched4.yaml --resume
    if ($LASTEXITCODE -ne 0) { Write-Error "DiffSBDD matched4 failed" }
}

& $Py311 analysis/merge_matched4_metrics.py
& $Py311 analysis/matched_isr_panel.py `
    --pocket2mol-metrics data/results/metrics_per_condition__runpocket2mol_isr_matched4_merged.csv `
    --diffsbdd-metrics data/results/metrics_per_condition__rundiffsbdd_isr_matched4_merged.csv

Write-Host "Done. See paper/matched_isr_4pocket.csv"
