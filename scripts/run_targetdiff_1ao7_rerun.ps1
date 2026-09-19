# Re-run TargetDiff ISR stress on pocket 1AO7 only (~30 min).
$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path $PSScriptRoot -Parent
$TargetDiffRoot = Join-Path (Split-Path $RepoRoot -Parent) "targetdiff"
$Py311 = "D:\Miniforge\envs\sbdd_py311\python.exe"
$TargetDiffPy = "D:\Miniforge\envs\pocket2mol\python.exe"
$Ckpt = Join-Path $TargetDiffRoot "pretrained_models\pretrained_diffusion.pt"

if (-not (Test-Path $Ckpt)) { Write-Error "Missing checkpoint: $Ckpt" }

Set-Location $RepoRoot
$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$log = Join-Path $RepoRoot "run_targetdiff_1ao7_rerun_$ts.log"

$env:TARGETDIFF_REPO = $TargetDiffRoot
$env:TARGETDIFF_PYTHON = $TargetDiffPy
$env:TARGETDIFF_SAMPLING_YAML = Join-Path $TargetDiffRoot "configs\sampling.yml"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:KMP_DUPLICATE_LIB_OK = "TRUE"
Remove-Item Env:PYTORCH_CUDA_ALLOC_CONF -ErrorAction SilentlyContinue

Write-Host "TargetDiff 1AO7 rerun: 1 pocket x 5 conditions"
Write-Host "Logging to $log"

& $Py311 -m pip install -e $RepoRoot -q
$prevEap = $ErrorActionPreference
$ErrorActionPreference = "Continue"
& $Py311 -u -m sbdd_robust run --config configs/experiments/targetdiff_isr_1ao7_rerun.yaml --resume 2>&1 |
    ForEach-Object { Add-Content -Path $log -Value $_; Write-Host $_ }
$exit = $LASTEXITCODE
$ErrorActionPreference = $prevEap
if ($exit -ne 0) { Write-Error "benchmark failed (exit $exit)" }

Write-Host "Done. Merge 1AO7 rows into metrics_per_condition__runtargetdiff_isr_smoke5.csv if needed."
