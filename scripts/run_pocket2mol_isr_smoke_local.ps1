# Minimal Pocket2Mol ISR smoke test (5 pockets x 6 conditions = 30 generations).
# Runs locally - no Colab credits needed. Expect ~1-3 hours on a mid-range GPU.
#
# Prerequisites: Pocket2Mol at D:\obsfu\Pocket2Mol, checkpoint, sbdd_py311 env.
# See POCKET2MOL_SETUP.md
#
# After the run finishes:
#   python analysis/isr_minimal_compare.py --pocket2mol-metrics data/results/metrics_per_condition__runpocket2mol_isr_smoke5.csv
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$WorkRoot = Split-Path $RepoRoot -Parent
$Pocket2MolRoot = Join-Path $WorkRoot "Pocket2Mol"
$Py311 = "D:\Miniforge\envs\sbdd_py311\python.exe"
if (-not (Test-Path $Py311)) {
    Write-Error "Not found: $Py311 - edit scripts/run_pocket2mol_isr_smoke_local.ps1"
}
if (-not (Test-Path $Pocket2MolRoot)) {
    Write-Error "Not found: $Pocket2MolRoot"
}

Set-Location $RepoRoot

& $Py311 analysis/make_brittleness_stress_config.py `
    --base-config configs/experiments/pocket2mol_real47.yaml `
    --out configs/experiments/pocket2mol_isr_smoke5.yaml `
    --minimal --max-pockets 5 --n-samples 8 --run-id pocket2mol_isr_smoke5
if ($LASTEXITCODE -ne 0) { Write-Error "config generation failed" }

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $RepoRoot "run_pocket2mol_isr_smoke5_$ts.log"

Remove-Item Env:SBDD_BRIDGE_DEBUG -ErrorAction SilentlyContinue
$env:PYTHONPATH = $Pocket2MolRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "Minimal ISR smoke: 5 pockets x 6 conditions"
Write-Host "Logging to $log"

& $Py311 -m pip install -e $RepoRoot -q
& $Py311 -u -m sbdd_robust run --config configs/experiments/pocket2mol_isr_smoke5.yaml --resume *>&1 | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { Write-Error "benchmark failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "Done. Run isr_minimal_compare.py on the metrics CSV in data/results/"
