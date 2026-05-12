from pathlib import Path

from sbdd_robust import provenance


def test_collect_git_provenance_shape():
    root = Path(__file__).resolve().parents[1]
    d = provenance.collect_git_provenance(root, diffsbdd_repo=None)
    assert d["sbdd_robust_repo_root"]
    assert "sbdd_robust_commit" in d
    assert d.get("diffsbdd_repo_root") is None
