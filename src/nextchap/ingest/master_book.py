"""`master_book.md`: the canonical text of every chapter.

Format (backward compatible with v1):

    ---
    source: book.epub
    chapters: 54
    ---

    ## Chapter 1: Title

    Prose...
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

from nextchap.ingest.cleanup import MD_HEADER_RE
from nextchap.storage.atomic import atomic_write_text

_INLINE_HEADER_RE = re.compile(r"^\s*##\s+Chapter\s+\d+", re.IGNORECASE)


@dataclass
class Chapter:
    number: int
    title: str
    content: str

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def display_title(self) -> str:
        if re.match(
            r"^(chapter|prologue|epilogue|interlude|side story|afterword)\b", self.title, re.I
        ):
            return self.title
        return f"Chapter {self.number}: {self.title}"

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.content.encode("utf-8")).hexdigest()[:16]


@dataclass
class MasterBook:
    frontmatter: dict[str, str] = field(default_factory=dict)
    chapters: list[Chapter] = field(default_factory=list)

    def by_number(self) -> dict[int, Chapter]:
        return {c.number: c for c in self.chapters}

    @property
    def last_number(self) -> int:
        return self.chapters[-1].number if self.chapters else 0


def sanitize_body(text: str) -> str:
    """Keep chapter bodies from containing lines that would parse as a new chapter header."""
    lines = []
    for line in text.splitlines():
        if _INLINE_HEADER_RE.match(line):
            line = line.lstrip().lstrip("#").strip()
        lines.append(line)
    return "\n".join(lines).strip()


def parse(text: str) -> MasterBook:
    book = MasterBook()
    lines = text.splitlines()
    i = 0
    if lines and lines[0].strip() == "---":
        i = 1
        while i < len(lines) and lines[i].strip() != "---":
            key, sep, value = lines[i].partition(":")
            if sep:
                book.frontmatter[key.strip()] = value.strip()
            i += 1
        i += 1

    cur_num: int | None = None
    cur_title = ""
    cur_lines: list[str] = []

    def flush() -> None:
        if cur_num is None:
            return
        body = "\n".join(cur_lines).strip()
        if body:
            book.chapters.append(Chapter(cur_num, cur_title, body))

    for line in lines[i:]:
        m = MD_HEADER_RE.match(line.strip())
        if m:
            flush()
            cur_num = int(m.group(1))
            cur_title = (m.group(2) or "").strip() or f"Chapter {cur_num}"
            cur_lines = []
        elif cur_num is not None:
            cur_lines.append(line)
    flush()
    return book


def render(book: MasterBook) -> str:
    fm = dict(book.frontmatter)
    fm["chapters"] = str(len(book.chapters))
    parts = ["---", *(f"{k}: {v}" for k, v in fm.items()), "---", ""]
    for ch in book.chapters:
        parts += [f"## Chapter {ch.number}: {ch.title}", "", ch.content.strip(), ""]
    return "\n".join(parts).rstrip() + "\n"


def load(path: Path) -> MasterBook:
    return parse(path.read_text(encoding="utf-8"))


def save(path: Path, book: MasterBook) -> None:
    atomic_write_text(path, render(book))


def upsert_chapter(path: Path, chapter: Chapter) -> bool:
    """Append (or replace) one chapter. Returns False when the identical chapter is present."""
    book = load(path) if path.exists() else MasterBook()
    chapter = Chapter(chapter.number, chapter.title, sanitize_body(chapter.content))
    existing = book.by_number().get(chapter.number)
    if existing is not None:
        if existing.content_hash == chapter.content_hash and existing.title == chapter.title:
            return False
        book.chapters = [chapter if c.number == chapter.number else c for c in book.chapters]
    else:
        book.chapters.append(chapter)
        book.chapters.sort(key=lambda c: c.number)
    save(path, book)
    return True
