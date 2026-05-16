# Run pocket2mol_real47 on Windows with GPU-friendly env + immediate logging (Tee-Object).
# Layout: sbdd-robust and Pocket2Mol are siblings under the same parent (e.g. D:\obsfu).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$WorkRoot = Split-Path $RepoRoot -Parent
$Pocket2MolRoot = Join-Path $WorkRoot "Pocket2Mol"
$Py311 = "D:\Miniforge\envs\sbdd_py311\python.exe"
if (-not (Test-Path $Py311)) {
    Write-Error "Not found: $Py311 — edit scripts/run_pocket2mol_real47_local.ps1"
}
if (-not (Test-Path $Pocket2MolRoot)) {
    Write-Error "Not found: $Pocket2MolRoot — set sibling Pocket2Mol next to sbdd-robust"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $RepoRoot "run_pocket2mol_real47_$ts.log"
Set-Location $RepoRoot

Remove-Item Env:SBDD_BRIDGE_DEBUG -ErrorAction SilentlyContinue
$env:PYTHONPATH = $Pocket2MolRoot
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

Write-Host "PYTHONPATH=$($env:PYTHONPATH)"
Write-Host "Logging to $log"
Write-Host "Tip: close extra Python/jobs hammering CUDA, then rely on this single run."

& $Py311 -m pip install -e $RepoRoot -q
if ($LASTEXITCODE -ne 0) { Write-Error "pip install -e sbdd-robust failed (exit $LASTEXITCODE)" }

& $Py311 -u -m sbdd_robust run --config configs\pocket2mol_real47.yaml *>&1 | Tee-Object -FilePath $log
