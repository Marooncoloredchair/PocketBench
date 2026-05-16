#!/usr/bin/env bash
# Optional docking deps for sbdd_robust (Meeko + gemmi everywhere; Vina binary on Windows).
set -euo pipefail

python -m pip install --upgrade pip
python -m pip install "gemmi" "meeko>=0.5.0"
if uname -s 2>/dev/null | grep -qiv mingw && [[ "$(uname -s)" != *MINGW* ]] && [[ "$(uname -s)" != *MSYS* ]]; then
  python -m pip install "vina>=1.2.2" || true
fi

echo "Windows: download AutoDock Vina from https://vina.scripps.edu/download.html"
echo "  and add vina.exe to PATH or set VINA_EXE to its full path."

# Conda (Linux/macOS): try conda-forge if pip vina fails to build
# conda install -y -c conda-forge boost-cpp
# python -m pip install "vina>=1.2.2"
