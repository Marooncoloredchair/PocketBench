# TargetDiff ISR smoke (5 pockets x 6 conditions = 30 generations).
# Matched panel vs pocket2mol_isr_smoke5_anchor510 (no anchor_offset for diffusion).
#
# Prerequisites:
#   - TargetDiff clone at D:\obsfu\targetdiff
#   - pretrained_models\pretrained_diffusion.pt (see TargetDiff README)
#   - pocket2mol conda env (torch 1.13 + pyg) for upstream sampling
#
# After the run:
#   python analysis/isr_minimal_compare.py --targetdiff-metrics data/results/metrics_per_condition__runtargetdiff_isr_smoke5.csv
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$WorkRoot = Split-Path $RepoRoot -Parent
$TargetDiffRoot = Join-Path $WorkRoot "targetdiff"
$Py311 = "D:\Miniforge\envs\sbdd_py311\python.exe"
$TargetDiffPy = "D:\Miniforge\envs\pocket2mol\python.exe"
$Ckpt = Join-Path $TargetDiffRoot "pretrained_models\pretrained_diffusion.pt"

if (-not (Test-Path $Py311)) {
    Write-Error "Not found: $Py311 - edit scripts/run_targetdiff_isr_smoke_local.ps1"
}
if (-not (Test-Path $TargetDiffPy)) {
    Write-Error "Not found: $TargetDiffPy"
}
if (-not (Test-Path $TargetDiffRoot)) {
    Write-Error "Not found: $TargetDiffRoot"
}
if (-not (Test-Path $Ckpt)) {
    Write-Error "Missing checkpoint: $Ckpt`nDownload from TargetDiff README (Google Drive pretrained_models)."
}

Set-Location $RepoRoot

& $Py311 analysis/make_brittleness_stress_config.py `
    --base-config configs/experiments/targetdiff_real50_base.yaml `
    --out configs/experiments/targetdiff_isr_smoke5.yaml `
    --minimal --max-pockets 5 --n-samples 8 --run-id targetdiff_isr_smoke5 --no-anchor
if ($LASTEXITCODE -ne 0) { Write-Error "config generation failed" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $RepoRoot "run_targetdiff_isr_smoke5_$ts.log"

$env:TARGETDIFF_REPO = $TargetDiffRoot
$env:TARGETDIFF_PYTHON = $TargetDiffPy
$env:TARGETDIFF_SAMPLING_YAML = Join-Path $TargetDiffRoot "configs\sampling.yml"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
# torch 1.13 (pocket2mol env) does not support expandable_segments
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

Write-Host "TargetDiff ISR smoke: 5 pockets x 5 conditions (batch_size=8)"
Write-Host "Logging to $log"

& $Py311 -m pip install -e $RepoRoot -q
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Py311 -u -m sbdd_robust run --config configs/experiments/targetdiff_isr_smoke5.yaml --resume 2>&1 |
    ForEach-Object { Add-Content -Path $log -Value $_; Write-Host $_ }
$exit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($exit -ne 0) { Get-Content $log -Tail 40 -ErrorAction SilentlyContinue; Write-Error "benchmark failed (exit $exit)" }

Write-Host ""
Write-Host "Done. Compare ISR with analysis/isr_minimal_compare.py"
