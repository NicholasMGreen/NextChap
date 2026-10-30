#!/usr/bin/env python3
"""Deprecated 1.x entry point. Use `nextchap books|clean|state|index|export ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nextchap.cli.legacy import book_manager

if __name__ == "__main__":
    raise SystemExit(book_manager(sys.argv[1:]))
