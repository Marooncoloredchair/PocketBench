import json, runpy, sys
from pathlib import Path

_wd = Path(__file__).resolve().parent
_REPO = "D:\\obsfu\\DiffSBDD"
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
sys.argv = json.loads((_wd / "_sbdd_robust_diffsbdd_argv.json").read_text(encoding="utf-8"))
runpy.run_path("D:\\obsfu\\DiffSBDD\\generate_ligands.py", run_name="__main__")
