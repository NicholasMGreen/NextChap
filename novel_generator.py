#!/usr/bin/env python3
"""Deprecated 1.x entry point. Use `nextchap import <file>` then `nextchap generate <book-id> -n N`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nextchap.cli.legacy import novel_generator

if __name__ == "__main__":
    raise SystemExit(novel_generator(sys.argv[1:]))
