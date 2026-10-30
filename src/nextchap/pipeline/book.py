"""A book on disk: metadata plus typed access to every per-book file."""

from __future__ import annotations

import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field

from nextchap.config import Settings
from nextchap.errors import BookNotFound
from nextchap.ingest import master_book
from nextchap.ingest.master_book import MasterBook
from nextchap.memory.summaries import ArcStore, SummaryStore
from nextchap.memory.world_state import WorldStateStore
from nextchap.prompts import raw
from nextchap.storage.atomic import atomic_write_json, atomic_write_text, read_json
from nextchap.storage.paths import BookPaths, book_paths

META_VERSION = 2


class BookMeta(BaseModel):
    schema_version: int = META_VERSION
    book_id: str
    source_name: str = ""
    source_kind: Literal["epub", "txt", "md"] = "md"
    title: str = ""
    author: str = ""
    language: str = "en"
    created_at: str = Field(
        default_factory=lambda: dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
    )
    source_chapters: int = 0
    generated: list[int] = Field(default_factory=list)


class Book:
    def __init__(self, paths: BookPaths, settings: Settings) -> None:
        self.paths = paths
        self.settings = settings
        self.summaries = SummaryStore(paths.summaries)
        self.arcs = ArcStore(paths.arcs, settings.generation.arc_size)
        self.world = WorldStateStore(paths.world_state, paths.world_history)

    @classmethod
    def open(cls, settings: Settings, book_id: str, *, must_exist: bool = True) -> Book:
        book = cls(book_paths(settings.data_root, book_id), settings)
        if must_exist and not book.exists():
            raise BookNotFound(f"Book '{book_id}' not found in {settings.data_root}.")
        return book

    @property
    def id(self) -> str:
        return self.paths.book_id

    def exists(self) -> bool:
        return self.paths.meta.exists() or self.paths.master_book.exists()

    # ── metadata ──────────────────────────────────────────────────────────────

    def meta(self) -> BookMeta:
        data = read_json(self.paths.meta, default=None)
        if data is not None:
            return BookMeta.model_validate(data)
        # Legacy book (pre-2.0): synthesize from master_book frontmatter.
        mb = self.master() if self.paths.master_book.exists() else MasterBook()
        source = mb.frontmatter.get("source", "")
        kind: Literal["epub", "txt", "md"] = (
            "epub" if source.lower().endswith(".epub") else "txt" if source else "md"
        )
        meta = BookMeta(
            book_id=self.id, source_name=source, source_kind=kind, source_chapters=len(mb.chapters)
        )
        self.save_meta(meta)
        return meta

    def save_meta(self, meta: BookMeta) -> None:
        atomic_write_json(self.paths.meta, meta.model_dump(mode="json"))

    # ── text artifacts ────────────────────────────────────────────────────────

    def master(self) -> MasterBook:
        return master_book.load(self.paths.master_book)

    def bible(self) -> str:
        p = self.paths.bible
        return p.read_text(encoding="utf-8") if p.exists() else ""

    def save_bible(self, text: str) -> None:
        atomic_write_text(self.paths.bible, text.strip() + "\n")

    def style(self) -> str:
        p = self.paths.style
        if not p.exists():
            atomic_write_text(p, raw("style_default"))
        return p.read_text(encoding="utf-8")
