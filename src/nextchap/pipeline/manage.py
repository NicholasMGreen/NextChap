"""Book maintenance that needs no model calls."""

from __future__ import annotations

import datetime as dt
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nextchap.errors import NextChapError
from nextchap.ingest import master_book
from nextchap.ingest.cleanup import CleanupReport, NoiseFilter, clean_master_markdown
from nextchap.ingest.master_book import Chapter
from nextchap.memory.summaries import ArcSummary, ChapterSummary
from nextchap.pipeline.book import Book
from nextchap.pipeline.reporter import NullReporter, Reporter
from nextchap.retrieval.store import VectorStore
from nextchap.storage.atomic import atomic_write_text


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _last_modified(path: Path) -> str:
    stamps = [f.stat().st_mtime for f in path.rglob("*") if f.is_file()] if path.exists() else []
    if not stamps:
        return "never"
    return dt.datetime.fromtimestamp(max(stamps)).strftime("%Y-%m-%d %H:%M")


def book_overview(book: Book) -> dict[str, Any]:
    meta = book.meta()
    mb = book.master() if book.paths.master_book.exists() else None
    state = book.world.load() if book.world.exists() else None
    return {
        "book_id": book.id,
        "title": meta.title,
        "source": meta.source_name,
        "chapters": len(mb.chapters) if mb else 0,
        "last_chapter": mb.last_number if mb else 0,
        "generated": len(meta.generated),
        "state_chapter": state.chapter_count if state else 0,
        "size_mb": round(_dir_size(book.paths.root) / 1_048_576, 2),
        "modified": _last_modified(book.paths.root),
        "path": str(book.paths.root),
    }


def clean_book(
    book: Book, *, backup: bool = True, extra_patterns: list[str] | None = None
) -> CleanupReport:
    path = book.paths.master_book
    original = path.read_text(encoding="utf-8")
    cleaned, report = clean_master_markdown(original, NoiseFilter(extra_patterns or []))
    if report.changed:
        if backup:
            atomic_write_text(path.with_suffix(".md.bak"), original)
        atomic_write_text(path, cleaned)
    return report


@dataclass
class RebuildResult:
    chapters: int
    renumbered: int
    index_reset: bool


def rebuild_state(
    book: Book, *, renumber: bool = True, dry_run: bool = False, backup: bool = True
) -> RebuildResult:
    """Re-derive bookkeeping from master_book.md (fixes chapter-number drift)."""
    path = book.paths.master_book
    original = path.read_text(encoding="utf-8")
    mb = master_book.parse(original)
    if not mb.chapters:
        raise NextChapError("master_book.md has no chapters.")
    mapping = {c.number: (i if renumber else c.number) for i, c in enumerate(mb.chapters, 1)}
    changed = sum(1 for old, new in mapping.items() if old != new)
    if dry_run:
        return RebuildResult(len(mb.chapters), changed, changed > 0)

    if changed:
        if backup:
            atomic_write_text(path.with_suffix(".md.pre_rebuild.bak"), original)
        mb.chapters = [Chapter(mapping[c.number], c.title, c.content) for c in mb.chapters]
        master_book.save(path, mb)

    old = book.summaries.load()
    book.summaries.save(
        {
            mapping[n]: ChapterSummary(mapping[n], s.title, s.summary, s.content_hash)
            for n, s in old.items()
            if n in mapping
        }
    )
    arcs = [
        ArcSummary(a.index, mapping.get(a.start, a.start), mapping.get(a.end, a.end), a.summary)
        for a in book.arcs.load()
    ]
    book.arcs.save(arcs)
    if book.world.exists():
        state = book.world.load()
        book.world.save(state.model_copy(update={"chapter_count": mb.chapters[-1].number}))
    meta = book.meta()
    meta.generated = sorted({mapping[n] for n in meta.generated if n in mapping})
    book.save_meta(meta)
    if changed:
        VectorStore.reset(book.paths.index)
    return RebuildResult(len(mb.chapters), changed, changed > 0)


def rollback(book: Book, *, to: int, truncate: bool) -> list[int]:
    """Restore the world state saved after chapter `to`; with truncate, drop later chapters."""
    snapshots = book.world.snapshots()
    if to not in snapshots:
        near = ", ".join(map(str, snapshots[-10:])) or "none"
        raise NextChapError(
            f"No world-state snapshot for chapter {to}.", hint=f"Available snapshots: {near}"
        )
    book.world.rollback(to)
    removed: list[int] = []
    if truncate:
        mb = book.master()
        removed = [c.number for c in mb.chapters if c.number > to]
        if removed:
            atomic_write_text(
                book.paths.master_book.with_suffix(".md.pre_rollback.bak"),
                book.paths.master_book.read_text(encoding="utf-8"),
            )
            mb.chapters = [c for c in mb.chapters if c.number <= to]
            master_book.save(book.paths.master_book, mb)
            summaries = book.summaries.load()
            book.summaries.save({n: s for n, s in summaries.items() if n <= to})
            book.arcs.save([a for a in book.arcs.load() if a.end <= to])
            meta = book.meta()
            meta.generated = [n for n in meta.generated if n <= to]
            book.save_meta(meta)
            VectorStore.reset(book.paths.index)
    return removed


def rebuild_index(
    book: Book, store_factory: Callable[[], VectorStore], reporter: Reporter | None = None
) -> int:
    ui = reporter or NullReporter()
    VectorStore.reset(book.paths.index)
    store = store_factory()
    chapters = book.master().chapters
    ui.phase("Indexing chapters", len(chapters))
    total = 0
    for i in range(0, len(chapters), 10):
        batch = chapters[i : i + 10]
        total += store.upsert_chapters(batch)
        ui.advance(len(batch))
    ui.end_phase()
    return total


def delete_book(book: Book) -> None:
    shutil.rmtree(book.paths.root)
