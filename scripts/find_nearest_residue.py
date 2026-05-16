#!/usr/bin/env python3
"""Find the protein residue (chain:resseq) whose CA (or residue centroid) is nearest the ligand centroid."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from sbdd_robust.datasets.find_nearest_residue import find_nearest_residue_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", type=Path, required=True)
    ap.add_argument("--ligand-chain", required=True)
    ap.add_argument("--ligand-resseq", type=int, required=True)
    ap.add_argument("--json", action="store_true", help='Print {"nearest_residue_id": ...}')
    args = ap.parse_args()
    rid = find_nearest_residue_id(args.pdb, args.ligand_chain, args.ligand_resseq)
    if args.json:
        print(json.dumps({"nearest_residue_id": rid}))
    else:
        print(rid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
