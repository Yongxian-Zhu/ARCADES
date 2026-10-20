#!/usr/bin/env python3
"""
arcade_aluminum_package/scripts/validate_2017_source.py

Standalone pre-flight check: given a run config (YAML), verify its declared
2017 `source_name` is authoritative BEFORE any expensive loading/sampling
happens. Exits non-zero and prints a clear error if not.

Usage:
    python scripts/validate_2017_source.py configs/aluminum_2017.yaml
"""

from __future__ import annotations

import sys

import yaml

sys.path.insert(0, "src")
from arcade_mfa_aluminum.adapters.aluminum.adapter import (  # noqa: E402
    AUTHORITATIVE_2017_SOURCES,
    DEPRECATED_2017_SOURCES,
    NonAuthoritative2017SourceError,
    assert_authoritative_2017,
)


def main(config_path: str) -> int:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    year = cfg.get("run", {}).get("year")
    source_name = cfg.get("data", {}).get("source_name")

    if year != 2017:
        print(f"[skip] config year is {year}, not 2017 -- no authoritative-source check needed.")
        return 0

    print(f"Checking 2017 source_name='{source_name}' against authoritative registry...")
    print(f"  Authoritative sources: {sorted(AUTHORITATIVE_2017_SOURCES)}")
    print(f"  Deprecated sources:    {sorted(DEPRECATED_2017_SOURCES)}")

    try:
        assert_authoritative_2017(source_name)
    except NonAuthoritative2017SourceError as e:
        print(f"\n[FAIL] {e}", file=sys.stderr)
        return 1

    print(f"\n[OK] '{source_name}' is an authoritative 2017 source. Safe to proceed.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_2017_source.py <config.yaml>", file=sys.stderr)
        sys.exit(2)
    sys.exit(main(sys.argv[1]))
