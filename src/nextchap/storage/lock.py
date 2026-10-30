"""One writer per book."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from filelock import FileLock, Timeout

from nextchap.errors import BookLocked


@contextmanager
def book_lock(lock_path: Path) -> Iterator[None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(lock_path), timeout=0)
    try:
        lock.acquire()
    except Timeout as exc:
        raise BookLocked(f"Book is locked: {lock_path.parent.name}") from exc
    try:
        yield
    finally:
        lock.release()
