"""
arcade_mfa_aluminum.paths
-------------------------
Filesystem helpers that keep output writes working on Windows.

Output paths such as `runs/<run_name>/node_balance_diagnostics.csv`
can exceed 260 characters when the package is run from a deeply nested working
directory, tripping the legacy Windows MAX_PATH limit. The failure mode is
misleading: `os.makedirs` succeeds because the directory path is shorter, and
then `open()` raises `FileNotFoundError` on a file whose parent exists.

`long_path` converts a path to the Win32 extended-length form (`\\\\?\\C:\\...`),
which bypasses MAX_PATH regardless of the `LongPathsEnabled` registry setting.
`prepare_output` additionally creates the parent directory. On POSIX both are
plain `os.path.abspath`.

Use `prepare_output(p)` at the point of writing and keep the original,
human-readable path for logging and for values returned to callers -- the
extended-length form is ugly in printed output.
"""

from __future__ import annotations

import os

_EXTENDED_PREFIX = "\\\\?\\"


def long_path(path: str | os.PathLike) -> str:
    """Absolute path, in extended-length form on Windows."""
    p = os.path.abspath(os.fspath(path))
    if os.name != "nt" or p.startswith(_EXTENDED_PREFIX):
        return p
    if p.startswith("\\\\"):                      # UNC share: \\server\share
        return _EXTENDED_PREFIX + "UNC" + p[1:]
    return _EXTENDED_PREFIX + p


def ensure_dir(path: str | os.PathLike) -> str:
    """Create `path` as a directory (parents included) and return a writable form."""
    p = long_path(path)
    os.makedirs(p, exist_ok=True)
    return p


def prepare_output(path: str | os.PathLike) -> str:
    """Create the parent directory of `path` and return a writable form of it."""
    p = long_path(path)
    parent = os.path.dirname(p)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return p
