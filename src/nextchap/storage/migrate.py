"""Move books from the 1.x layout (./novel_books, ./chroma_books) into the data root.

World state, arc summaries and book metadata are upgraded lazily on first open; the
old Chroma index is not carried over because the embedding model changed.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from nextchap.ingest import master_book
from nextchap.storage.paths import books_dir, slugify

LEGACY_BOOKS = "novel_books"
LEGACY_CHROMA = "chroma_books"
_SKIP = {"recent_chapters.json"}


@dataclass
class MigrationResult:
    legacy_id: str
    book_id: str
    status: str  # "migrated" | "exists" | "empty"


def find_legacy(project_dir: Path) -> list[Path]:
    root = project_dir / LEGACY_BOOKS
    if not root.is_dir():
        return []
    return sorted(
        p
        for p in root.iterdir()
        if p.is_dir()
        and ((p / "master_book.md").exists() or (p / "chapter_summaries.json").exists())
    )


def migrate(project_dir: Path, data_root: Path, *, copy: bool = True) -> list[MigrationResult]:
    results = []
    for src in find_legacy(project_dir):
        book_id = slugify(src.name)
        dest = books_dir(data_root) / book_id
        if dest.exists():
            results.append(MigrationResult(src.name, book_id, "exists"))
            continue
        if not (src / "master_book.md").exists():
            results.append(MigrationResult(src.name, book_id, "empty"))
            continue
        dest.mkdir(parents=True)
        for item in src.iterdir():
            if item.name in _SKIP:
                continue
            target = dest / item.name
            if item.is_dir():
                shutil.copytree(item, target)
            else:
                shutil.copy2(item, target)
        source = master_book.load(dest / "master_book.md").frontmatter.get("source", "")
        candidate = project_dir / source if source else None
        if candidate and candidate.suffix.lower() == ".epub" and candidate.is_file():
            shutil.copy2(candidate, dest / "template.epub")
        if not copy:
            shutil.rmtree(src)
            shutil.rmtree(project_dir / LEGACY_CHROMA / src.name, ignore_errors=True)
        results.append(MigrationResult(src.name, book_id, "migrated"))
    return results
