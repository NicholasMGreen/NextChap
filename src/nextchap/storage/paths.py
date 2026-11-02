"""Book IDs and the on-disk layout of a book."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from nextchap.errors import InvalidBookId

BOOK_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


def validate_book_id(book_id: str) -> str:
    if not BOOK_ID_RE.fullmatch(book_id or ""):
        raise InvalidBookId(f"Invalid book ID: {book_id!r}")
    return book_id


def slugify(name: str) -> str:
    """Turn a title or filename into a valid book ID."""
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "_", s).strip("_-")
    s = s[:64].rstrip("_-")
    return s or "book"


@dataclass(frozen=True)
class BookPaths:
    root: Path

    @property
    def book_id(self) -> str:
        return self.root.name

    @property
    def meta(self) -> Path:
        return self.root / "book.json"

    @property
    def master_book(self) -> Path:
        return self.root / "master_book.md"

    @property
    def summaries(self) -> Path:
        return self.root / "chapter_summaries.json"

    @property
    def arcs(self) -> Path:
        return self.root / "arc_summaries.json"

    @property
    def world_state(self) -> Path:
        return self.root / "world_state.json"

    @property
    def world_history(self) -> Path:
        return self.root / "world_state_history"

    @property
    def bible(self) -> Path:
        return self.root / "master_bible.md"

    @property
    def style(self) -> Path:
        return self.root / "style.md"

    @property
    def generated(self) -> Path:
        return self.root / "generated_chapters"

    @property
    def combined(self) -> Path:
        return self.root / "generated_chapters_combined.md"

    @property
    def index(self) -> Path:
        return self.root / "index"

    @property
    def template_epub(self) -> Path:
        return self.root / "template.epub"

    @property
    def journal(self) -> Path:
        return self.root / "journal.json"

    @property
    def staging(self) -> Path:
        return self.root / "staging"

    @property
    def usage(self) -> Path:
        return self.root / "usage.jsonl"

    @property
    def lock(self) -> Path:
        return self.root / ".lock"

    def staging_for(self, chapter: int) -> Path:
        return self.staging / f"ch{chapter:05d}"

    def export_epub(self, stem: str) -> Path:
        return self.root / f"{stem}_with_generated.epub"


def books_dir(data_root: Path) -> Path:
    """Books live in <data_root>/novel_books/<book_id> (same layout as 1.x)."""
    return data_root / "novel_books"


def book_paths(data_root: Path, book_id: str) -> BookPaths:
    return BookPaths(books_dir(data_root) / validate_book_id(book_id))


def all_book_ids(data_root: Path) -> list[str]:
    root = books_dir(data_root)
    if not root.exists():
        return []
    return sorted(p.name for p in root.iterdir() if p.is_dir() and BOOK_ID_RE.fullmatch(p.name))
