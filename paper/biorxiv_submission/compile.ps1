# Portable MiKTeX on D: (adjust $MikTeXBin if your install differs).
$MikTeXBin = "D:\MIKETeX\miktex\bin\x64"
if (-not (Test-Path "$MikTeXBin\pdflatex.exe")) {
    Write-Error "pdflatex not found at $MikTeXBin — edit compile.ps1 or add MiKTeX to PATH."
    exit 1
}
$env:Path = "$MikTeXBin;$env:Path"

Set-Location $PSScriptRoot
pdflatex -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
bibtex main
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pdflatex -interaction=nonstopmode main.tex
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
pdflatex -interaction=nonstopmode main.tex
exit $LASTEXITCODE
