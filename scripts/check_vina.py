#!/usr/bin/env python3
"""Print AutoDock Vina / Meeko availability; on Windows use ``vina.exe`` on PATH or ``VINA_EXE`` (pip ``vina`` often won't build)."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _vina_exe() -> str | None:
    env = os.environ.get("VINA_EXE", "").strip()
    if env and Path(env).is_file():
        return env
    return shutil.which("vina") or shutil.which("vina.exe")


def main() -> int:
    ok_meeko = False
    ok_vina = False
    notes: list[str] = []

    try:
        import meeko  # type: ignore

        ver = getattr(meeko, "__version__", "unknown")
        print(f"meeko: {ver}")
        ok_meeko = True
    except Exception as e:
        print(f"meeko: not importable ({e.__class__.__name__})")

    try:
        import vina  # type: ignore

        ver = getattr(vina, "__version__", "unknown")
        print(f"vina (Python package): {ver}")
        ok_vina = True
    except Exception as e:
        notes.append(f"Python package autodock-vina not importable ({e.__class__.__name__})")

    exe = _vina_exe()
    if exe:
        try:
            proc = subprocess.run(
                [exe, "--help"],
                capture_output=True,
                timeout=15,
                text=True,
            )
            # vina exits 0 or 1 for --help on some builds; accept either if no crash
            print(f"vina (executable): {exe}")
            ok_vina = True
        except Exception as e:
            print(f"vina (executable): {exe} failed to run ({e.__class__.__name__})")
    else:
        raw = os.environ.get("VINA_EXE", "").strip()
        if raw and not Path(raw).is_file():
            print(f"vina (executable): VINA_EXE set but file not found: {raw}")
        else:
            print("vina (executable): not on PATH (download vina.exe or set VINA_EXE)")

    if notes and not ok_vina:
        for n in notes:
            print(f"note: {n}")

    if ok_meeko and ok_vina:
        print()
        print("Ready for docking: Meeko (PDBQT prep) + Vina (Python API or vina.exe).")
        return 0

    print()
    print("Install hints:")
    print("  pip install gemmi meeko   # gemmi is required for meeko>=0.7")
    print("  pip install -e \".[vina]\"  # meeko+gemmi; PyPI vina is skipped on Windows")
    print("  # Linux/macOS: pip install vina   (needs Boost dev headers if no wheel)")
    print("  # Windows: pip install vina usually fails (Boost); download AutoDock Vina from")
    print("  #   https://vina.scripps.edu/download.html , put vina.exe on PATH or set VINA_EXE.")
    print("  Optional: pip install -e \".[vina]\"  (meeko + gemmi; vina pip omits on Windows)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
