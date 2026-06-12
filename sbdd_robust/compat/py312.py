"""Small shims for Python 3.12+ on Colab (DiffSBDD / PL 1.8 stack)."""

from __future__ import annotations


def ensure_pkgutil_impimporter_shim() -> None:
    """Restore ``pkgutil.ImpImporter`` removed in Python 3.12.

    Old ``pytorch-lightning==1.8`` pulls ``lightning_lite``, which imports
    system ``pkg_resources`` that still references ``ImpImporter``.
    """
    import pkgutil

    if hasattr(pkgutil, "ImpImporter"):
        return

    class ImpImporter:  # noqa: N801 — legacy name expected by old pkg_resources
        def find_module(self, fullname, path=None):
            return None

    pkgutil.ImpImporter = ImpImporter  # type: ignore[attr-defined]
