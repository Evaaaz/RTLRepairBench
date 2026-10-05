"""Shared, extraction-safe path resolution for the research code.

The workshop code currently lives inside a larger checkout, but it is also
intended to run as a standalone repository.  Callers must therefore not encode
any enclosing monorepo nesting depth.  ``RTLREPAIR_ROOT`` is
the explicit override; otherwise we choose the nearest ancestor that owns one
of the repository-level data/config/toolchain directories.  A bare extraction
with none of those directories yet present falls back to the directory that
contains ``code/``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Optional


CODE_ROOT = Path(__file__).resolve().parent
BUNDLE_ROOT = CODE_ROOT.parent
_ROOT_MARKERS = ("data", "configs", "docker")


def _resolved(path: Path) -> Path:
    return path.expanduser().resolve()


def discover_project_root(
    *,
    code_root: Path = CODE_ROOT,
    environ: Optional[Mapping[str, str]] = None,
) -> Path:
    """Return the repository root without relying on a fixed parent depth.

    ``code_root`` is injectable so the discovery rule can be tested in an
    isolated temporary tree.  The nearest marked ancestor wins, which supports
    both ``<standalone>/code`` and this tree's historical nested location.
    """

    env = os.environ if environ is None else environ
    override = (env.get("RTLREPAIR_ROOT") or "").strip()
    if override:
        return _resolved(Path(override))

    code = _resolved(code_root)
    bundle = code.parent
    for candidate in (bundle, *bundle.parents):
        if any((candidate / marker).is_dir() for marker in _ROOT_MARKERS):
            return candidate
    return bundle


def project_root() -> Path:
    return discover_project_root()


def data_root() -> Path:
    override = (os.environ.get("RTLREPAIR_DATA") or "").strip()
    return _resolved(Path(override)) if override else project_root() / "data"


def output_root() -> Path:
    override = (os.environ.get("RTLREPAIR_OUT") or "").strip()
    return _resolved(Path(override)) if override else project_root() / "generated"


if __name__ == "__main__":
    print(project_root())
