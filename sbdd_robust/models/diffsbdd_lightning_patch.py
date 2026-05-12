"""Patch vanilla DiffSBDD ``lightning_modules.py`` in-place (Colab-safe).

Upstream indexes pocket residues as ``(' ', resseq, ' ')``, which raises
``KeyError`` for PDBs that use insertion codes. This module inserts helpers
and rewrites the ``pocket_ids`` block once per resolved ``repo_root``.
"""

from __future__ import annotations

from pathlib import Path

_OLD_POCKET_BLOCK = (
    "        if pocket_ids is not None:\n"
    "            # define pocket with list of residues\n"
    "            residues = [\n"
    "                pdb_struct[x.split(':')[0]][(' ', int(x.split(':')[1]), ' ')]\n"
    "                for x in pocket_ids]\n\n"
    "        else:"
)

_NEW_POCKET_BLOCK = (
    "        if pocket_ids is not None:\n"
    "            residues = []\n"
    "            for x in pocket_ids:\n"
    "                parts = str(x).split(':', 1)\n"
    "                if len(parts) != 2:\n"
    "                    raise ValueError(\n"
    "                        f\"Bad pocket id {x!r}; expected chain:resseq\")\n"
    "                chain = _diffsbdd_chain(pdb_struct, parts[0])\n"
    "                residues.append(_diffsbdd_residue(chain, int(parts[1])))\n\n"
    "        else:"
)

_HELPERS_BEFORE_CLASS = (
    "\n\n"
    "def _diffsbdd_chain(pdb_model, chain_key: str):\n"
    "    ck = (chain_key or \"\").strip()\n"
    "    if ck in pdb_model:\n"
    "        return pdb_model[ck]\n"
    "    for cid in pdb_model:\n"
    "        if str(cid).strip() == ck:\n"
    "            return pdb_model[cid]\n"
    "    raise KeyError(f\"chain {chain_key!r} not in structure\")\n\n\n"
    "def _diffsbdd_residue(chain, resseq: int):\n"
    "    rid0 = (\" \", int(resseq), \" \")\n"
    "    if rid0 in chain:\n"
    "        return chain[rid0]\n"
    "    hits = [rid for rid in chain.child_dict if rid[1] == int(resseq)]\n"
    "    if not hits:\n"
    "        raise KeyError(f\"resseq {resseq} not in chain {chain.id!r}\")\n"
    "    if len(hits) == 1:\n"
    "        return chain[hits[0]]\n"
    "    hits.sort(key=lambda r: (str(r[0]), str(r[2])))\n"
    "    return chain[hits[0]]\n\n"
)

_MARK = "def _diffsbdd_chain"

_patched_roots: set[str] = set()


def ensure_lightning_resi_patch(repo_root: Path) -> None:
    """Idempotent: fix ``--resi_list`` / pocket_ids handling in DiffSBDD."""
    root = Path(repo_root).resolve()
    key = str(root)
    if key in _patched_roots:
        return

    lm = root / "lightning_modules.py"
    if not lm.is_file():
        _patched_roots.add(key)
        return

    txt = lm.read_text(encoding="utf-8")
    if _MARK in txt:
        _patched_roots.add(key)
        return

    if _OLD_POCKET_BLOCK not in txt:
        _patched_roots.add(key)
        return

    marker = "class LigandPocketDDPM"
    idx = txt.find(marker)
    if idx == -1:
        _patched_roots.add(key)
        return

    txt = txt[:idx] + _HELPERS_BEFORE_CLASS + txt[idx:]
    txt = txt.replace(_OLD_POCKET_BLOCK, _NEW_POCKET_BLOCK, 1)
    lm.write_text(txt, encoding="utf-8")
    _patched_roots.add(key)
