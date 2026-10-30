"""EPUB input and output.

Parsing uses `html.parser`: EPUB content is XHTML, but "lxml" triggers
XMLParsedAsHTMLWarning and "xml" breaks tag lookup on namespaced documents.
"""

from __future__ import annotations

import html
import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import ebooklib
from bs4 import BeautifulSoup, Tag
from ebooklib import epub

from nextchap.errors import ParseError
from nextchap.ingest.cleanup import NoiseFilter
from nextchap.ingest.master_book import Chapter
from nextchap.ingest.text import Block, DetectionReport, detect_chapters

_BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote", "pre"]
_HEADING_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = ["script", "style", "head", "nav", "aside", "figure", "figcaption"]
_FRONT_MATTER_RE = re.compile(r"cover|title", re.IGNORECASE)
_SCENE_BREAK_RE = re.compile(r"^\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:#\s*){3}|~{3,}|◇+|◆+)\s*$")
_HEADING_MARK_RE = re.compile(r"^#{1,6}\s+")
_STRONG_RE = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
_EM_RE = re.compile(r"(?<![\w*])\*(?=[^\s*])(.+?)(?<=[^\s*])\*(?![\w*])")


@dataclass
class EpubMeta:
    title: str = ""
    author: str = ""
    language: str = "en"


def _first_meta(book: epub.EpubBook, name: str) -> str:
    values = book.get_metadata("DC", name)
    if values and values[0] and values[0][0]:
        return str(values[0][0])
    return ""


def _is_nav(item: Any) -> bool:
    return isinstance(item, epub.EpubNav) or "nav" in (getattr(item, "properties", None) or [])


def _ordered_documents(book: epub.EpubBook) -> list[Any]:
    docs = {item.id: item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
    ordered: list[Any] = []
    seen: set[str] = set()
    for item_id, _ in book.spine:
        item = docs.get(item_id)
        if item is not None and item.id not in seen and not _is_nav(item):
            ordered.append(item)
            seen.add(item.id)
    ordered.extend(i for i in docs.values() if i.id not in seen and not _is_nav(i))
    return ordered


def _blocks_from_document(content: bytes) -> list[Block]:
    soup = BeautifulSoup(content, "html.parser")
    for tag in soup(_SKIP_TAGS):
        tag.decompose()
    body = soup.find("body") or soup
    blocks: list[Block] = []
    for el in body.find_all(_BLOCK_TAGS):
        if not isinstance(el, Tag) or el.find(_BLOCK_TAGS):
            continue
        text = el.get_text(" ", strip=True).replace("\ufffd", "")
        if text:
            blocks.append(Block(text, heading=el.name in _HEADING_TAGS))
    if not blocks:
        raw = body.get_text("\n", strip=True).replace("\ufffd", "")
        blocks = [Block(line.strip()) for line in raw.splitlines() if line.strip()]
    for i, b in enumerate(blocks):
        b.strong = b.heading or i == 0
    return blocks


def read_epub(path: Path, noise: NoiseFilter | None = None) -> tuple[DetectionReport, EpubMeta]:
    try:
        book = epub.read_epub(str(path), options={"ignore_ncx": True})
    except Exception as exc:
        raise ParseError(f"Could not read EPUB {path.name}: {exc}") from exc
    blocks: list[Block] = []
    for item in _ordered_documents(book):
        blocks.extend(_blocks_from_document(item.get_content()))
    report = detect_chapters(blocks, joiner="\n\n", noise=noise)
    if not report.chapters:
        text = "\n\n".join(b.text for b in blocks).strip()
        if not text:
            raise ParseError(f"No readable text found in {path.name}.")
        report.chapters = [Chapter(1, "Chapter 1", text)]
        report.anomalies.append(
            "No chapter headings found; treating the whole book as one chapter."
        )
    meta = EpubMeta(
        title=_first_meta(book, "title"),
        author=_first_meta(book, "creator"),
        language=_first_meta(book, "language") or "en",
    )
    return report, meta


# ── Output ────────────────────────────────────────────────────────────────────


def _inline(text: str) -> str:
    s = html.escape(text, quote=False)
    s = _STRONG_RE.sub(r"<strong>\1</strong>", s)
    s = _EM_RE.sub(r"<em>\1</em>", s)
    return s


def _paragraphs(text: str) -> list[str]:
    paras = [p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    if len(paras) == 1 and "\n" in paras[0]:
        paras = [line for line in paras[0].splitlines() if line.strip()]
    return paras


def prose_to_html(title: str, text: str) -> str:
    out = [f"<h2>{html.escape(title, quote=False)}</h2>"]
    for para in _paragraphs(text):
        if _SCENE_BREAK_RE.match(para):
            out.append('<hr class="scene-break"/>')
            continue
        lines = [_inline(_HEADING_MARK_RE.sub("", ln.strip())) for ln in para.splitlines()]
        out.append("<p>" + "<br/>".join(ln for ln in lines if ln) + "</p>")
    return "\n".join(out)


def _book_from_template(template: Path) -> tuple[epub.EpubBook, list[Any]]:
    book = epub.read_epub(str(template), options={"ignore_ncx": True})
    spine_order = {item_id: i for i, (item_id, _) in enumerate(book.spine)}
    front: list[Any] = []
    for item in list(book.get_items()):
        is_doc = item.get_type() == ebooklib.ITEM_DOCUMENT
        if item.get_type() == ebooklib.ITEM_NAVIGATION or (is_doc and _is_nav(item)):
            book.items.remove(item)
        elif is_doc:
            if _FRONT_MATTER_RE.search(f"{item.id} {item.file_name}"):
                front.append(item)
            else:
                book.items.remove(item)
    front.sort(key=lambda it: spine_order.get(it.id, 10_000))
    book.toc = []
    book.guide = []
    return book, front


def write_epub(
    chapters: list[Chapter], out_path: Path, template: Path | None, meta: EpubMeta
) -> None:
    if template is not None and template.exists():
        book, front = _book_from_template(template)
    else:
        book, front = epub.EpubBook(), []
        book.set_identifier(f"urn:uuid:{uuid.uuid4()}")
        book.set_title(meta.title or "Untitled")
        book.set_language(meta.language or "en")
        if meta.author:
            book.add_author(meta.author)

    lang = meta.language or "en"
    items = []
    for ch in chapters:
        item = epub.EpubHtml(
            uid=f"nc-ch{ch.number:05d}",
            file_name=f"text/nc-ch{ch.number:05d}.xhtml",
            title=ch.display_title,
            lang=lang,
        )
        item.content = prose_to_html(ch.display_title, ch.content)
        book.add_item(item)
        items.append(item)

    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.toc = [epub.Link(it.file_name, it.title, it.id) for it in items]
    book.spine = [*front, "nav", *items]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out_path.parent) as tmpdir:
        tmp = Path(tmpdir) / out_path.name
        epub.write_epub(str(tmp), book, {})
        tmp.replace(out_path)


def check_epub(path: Path) -> tuple[bool, str] | None:
    """Run epubcheck if installed. Returns None when it is not available."""
    exe = shutil.which("epubcheck")
    if exe is None:
        return None
    proc = subprocess.run(  # noqa: S603 - fixed executable from PATH, no shell
        [exe, str(path)], capture_output=True, text=True, check=False
    )
    return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
