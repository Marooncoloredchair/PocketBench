"""Git provenance helpers for reproducible benchmark runs."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Optional


def git_commit(repo: Path, timeout_s: float = 15.0) -> Optional[str]:
    """Return ``git rev-parse HEAD`` for ``repo``, or ``None`` if unavailable."""
    repo = Path(repo).resolve()
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            return None
        h = proc.stdout.strip()
        return h or None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def git_is_dirty(repo: Path, timeout_s: float = 15.0) -> Optional[bool]:
    """True if working tree has uncommitted changes; ``None`` if unknown."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        if proc.returncode != 0:
            return None
        return bool(proc.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def collect_git_provenance(
    sbdd_robust_repo: Path,
    diffsbdd_repo: Optional[Path] = None,
    pocket2mol_repo: Optional[Path] = None,
    targetdiff_repo: Optional[Path] = None,
) -> dict[str, Any]:
    sbdd_robust_repo = Path(sbdd_robust_repo).resolve()
    out: dict[str, Any] = {
        "sbdd_robust_repo_root": str(sbdd_robust_repo),
        "sbdd_robust_commit": git_commit(sbdd_robust_repo),
        "sbdd_robust_dirty": git_is_dirty(sbdd_robust_repo),
    }
    if diffsbdd_repo is not None:
        dr = Path(diffsbdd_repo).resolve()
        out["diffsbdd_repo_root"] = str(dr)
        out["diffsbdd_commit"] = git_commit(dr)
        out["diffsbdd_dirty"] = git_is_dirty(dr)
    else:
        out["diffsbdd_repo_root"] = None
        out["diffsbdd_commit"] = None
        out["diffsbdd_dirty"] = None
    if pocket2mol_repo is not None:
        pr = Path(pocket2mol_repo).resolve()
        out["pocket2mol_repo_root"] = str(pr)
        out["pocket2mol_commit"] = git_commit(pr)
        out["pocket2mol_dirty"] = git_is_dirty(pr)
    else:
        out["pocket2mol_repo_root"] = None
        out["pocket2mol_commit"] = None
        out["pocket2mol_dirty"] = None
    if targetdiff_repo is not None:
        tr = Path(targetdiff_repo).resolve()
        out["targetdiff_repo_root"] = str(tr)
        out["targetdiff_commit"] = git_commit(tr)
        out["targetdiff_dirty"] = git_is_dirty(tr)
    else:
        out["targetdiff_repo_root"] = None
        out["targetdiff_commit"] = None
        out["targetdiff_dirty"] = None
    return out
