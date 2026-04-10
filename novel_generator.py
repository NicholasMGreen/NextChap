#!/usr/bin/env python3
"""
Novel Chapter Generator
========================
Generates new web novel chapters using:
  - LlamaIndex + ChromaDB for RAG
  - Hierarchical summarization (chapter -> arc -> master bible)
  - Structured world state tracking (characters, plot threads, lore)
  - Agentic two-pass generation (plan -> write)
  - DeepSeek V3 via OpenRouter for all LLM calls
  - Native EPUB input/output via ebooklib (guaranteed Unicode fidelity)
  - Per-book isolated file structure (switch books freely)

Usage:
  python novel_generator.py <novel.epub> <num_chapters> [--api-key KEY] [--book-id SLUG]
  python novel_generator.py <novel.txt>  <num_chapters> [--api-key KEY] [--book-id SLUG]

Set OPENROUTER_API_KEY env var or pass --api-key.

Continuing a book (same --book-id): for EPUB, if novel_books/<id>/<stem>_with_generated.epub
exists it is used as the source instead of the file you pass, so new chapters you add to that
merged EPUB are summarised and indexed before the next generation run.
"""

import os
import re
import json
import time
import copy
import argparse
import unicodedata
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# ── Dependency check ─────────────────────────────────────────────────────────

def _check_deps():
    missing = []
    for pkg, imp in [
        ("chromadb",                            "chromadb"),
        ("llama-index-core",                    "llama_index.core"),
        ("llama-index-vector-stores-chroma",    "llama_index.vector_stores.chroma"),
        ("llama-index-embeddings-huggingface",  "llama_index.embeddings.huggingface"),
        ("openai",                              "openai"),
        ("ebooklib",                            "ebooklib"),
        ("beautifulsoup4",                      "bs4"),
    ]:
        try:
            __import__(imp)
        except ImportError:
            missing.append(pkg)
    if missing:
        print("Missing dependencies. Run:\n")
        print(f"  pip install {' '.join(missing)} sentence-transformers torch\n")
        raise SystemExit(1)

_check_deps()

import chromadb
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core import StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from openai import OpenAI
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_API_KEY  = os.environ.get("OPENROUTER_API_KEY", "")
DEEPSEEK_MODEL      = "deepseek/deepseek-chat"
EMBED_MODEL         = "BAAI/bge-large-en-v1.5"

ARC_SIZE            = 20
RECENT_FULL_COUNT   = 3
BIBLE_REFRESH_EVERY = 5
TARGET_WORDS        = (1500, 2000)

# Root directories -- all per-book data lives inside subdirs of these
BOOKS_ROOT  = Path("./novel_books")   # novel_books/<book_id>/...
CHROMA_ROOT = Path("./chroma_books")  # chroma_books/<book_id>/


# ── Per-book path helpers ─────────────────────────────────────────────────────

def book_paths(book_id: str) -> dict:
    root    = BOOKS_ROOT / book_id
    gen_dir = root / "generated_chapters"
    return {
        "root":         root,
        "gen_dir":      gen_dir,
        "state":        root / "world_state.json",
        "bible":        root / "master_bible.md",
        "combined":     root / "generated_chapters_combined.md",
        "chroma":       CHROMA_ROOT / book_id,
        # Resume-state files written after first run
        "ch_summaries": root / "chapter_summaries.json",   # all chapter {number,title,summary}
        "arc_summaries":root / "arc_summaries.json",       # list of arc summary strings
        "recent_chs":   root / "recent_chapters.json",     # last N chapters with full content
        "master_book":  root / "master_book.md",           # structured markdown of entire book
    }


def slug(name: str) -> str:
    """Convert a book title or filename to a safe directory name."""
    s = unicodedata.normalize("NFKD", name)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_-]+", "_", s)
    return s or "book"


# ── Markdown book parser ───────────────────────────────────────────────────────

_MD_CH_RE = re.compile(r"^##\s+Chapter\s+(\d+)[:\s\-]*(.*)$", re.IGNORECASE)
_PROMO_LINE_RE = re.compile(
    r"(patre\.?on|patreon\.com|chapters?\s+in\s+advance|^\s*\[?\d+\s+chapters?\s+in\s+advance\]?\s*$)",
    re.IGNORECASE,
)
_DOTS_RE = re.compile(r"^\s*\.{3,}\s*$")


def parse_markdown_book(md_text: str) -> list:
    """
    Parse a structured markdown book file into Chapter objects.

    Expected format:
        ---
        source: book.epub
        chapters: 54
        ---

        ## Chapter 1: Title

        Chapter content...

        ## Chapter 2: Title

        Chapter content...
    """
    chapters = []
    cur_num  = None
    cur_title = ""
    cur_lines = []

    for line in md_text.splitlines():
        m = _MD_CH_RE.match(line.strip())
        if m:
            if cur_num is not None:
                body = "\n".join(cur_lines).strip()
                if body:
                    chapters.append(Chapter(cur_num, cur_title, body))
            cur_num   = int(m.group(1))
            cur_title = m.group(2).strip() or f"Chapter {cur_num}"
            cur_lines = []
        elif cur_num is not None:
            cur_lines.append(line)

    if cur_num is not None:
        body = "\n".join(cur_lines).strip()
        if body:
            chapters.append(Chapter(cur_num, cur_title, body))

    return chapters


def clean_master_markdown_text(md_text: str) -> tuple[str, int, int]:
    """
    Remove common author promo/noise lines and duplicate chapter headers.
    Returns (cleaned_text, removed_noise_lines, removed_duplicate_headers).
    """
    lines = md_text.splitlines()
    out = []
    removed_noise = 0
    removed_dupes = 0

    last_header_key = None
    saw_substantive_since_header = False

    for line in lines:
        s = line.strip()
        if s and (_PROMO_LINE_RE.search(s) or _DOTS_RE.match(s)):
            removed_noise += 1
            continue

        m = _MD_CH_RE.match(s)
        if m:
            key = f"{int(m.group(1))}:{re.sub(r'\\s+', ' ', (m.group(2) or '').strip().lower())}"
            if key == last_header_key and not saw_substantive_since_header:
                removed_dupes += 1
                continue
            out.append(line)
            last_header_key = key
            saw_substantive_since_header = False
            continue

        out.append(line)
        if s:
            saw_substantive_since_header = True

    cleaned = "\n".join(out)
    if md_text.endswith("\n"):
        cleaned += "\n"
    return cleaned, removed_noise, removed_dupes


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Chapter:
    number:     int
    title:      str
    content:    str
    summary:    str = ""
    word_count: int = 0

    def __post_init__(self):
        self.word_count = len(self.content.split())


@dataclass
class WorldState:
    characters:             dict = field(default_factory=dict)
    active_plot_threads:    list = field(default_factory=list)
    resolved_plot_threads:  list = field(default_factory=list)
    world_rules:            list = field(default_factory=list)
    locations:              dict = field(default_factory=dict)
    factions:               dict = field(default_factory=dict)
    key_items:              list = field(default_factory=list)
    current_arc:            str  = ""
    story_tone:             str  = ""
    narrative_pov:          str  = ""
    last_chapter_hook:      str  = ""
    chapter_count:          int  = 0


# ── EPUB handler ──────────────────────────────────────────────────────────────

class EpubHandler:
    """
    Reads an EPUB file into Chapter objects and writes new chapters back
    as a properly formed EPUB, preserving all original metadata and content.

    Library choice: ebooklib + lxml + BeautifulSoup4.
    - ebooklib reads/writes EPUB2 and EPUB3 with full Unicode via lxml XML
      parser -- zero character corruption risk.
    - BeautifulSoup4 with html.parser strips HTML/XHTML tags to clean prose
      while preserving every Unicode codepoint (em-dashes, curly quotes, etc.)
    - No intermediate encoding steps that could mangle characters.

    Parser note: EPUB content files are XHTML but are best parsed with
    html.parser (Python stdlib). Using "lxml" triggers XMLParsedAsHTMLWarning
    and using "xml" mangles tag names with namespace prefixes, breaking
    find_all("p"). html.parser handles both HTML and XHTML correctly with
    zero warnings and no namespace issues.
    """

    _HEADER_PATTERNS = [
        r"^(?:Chapter|CHAPTER|Ch\.?)\s*(\d+)[:\s\-\u2013\u2014]*(.*)$",
        r"^(\d+)[.:]\s+(.+)$",          # matches "3: Go! Go! Go!" and "3. Title"
        r"^(\d+):\s*(.*)$",             # matches bare "3:" with optional title
        r"^(?:Chapter|CHAPTER)\s+([IVXLCDM]+)[:\s\-\u2013\u2014]*(.*)$",
    ]

    def read(self, filepath: str) -> tuple:
        """
        Returns (chapters: list[Chapter], original_book: epub.EpubBook).

        Strategy: flatten the entire EPUB into one plain-text string in spine
        order, then hand that string to TextParser._split() -- the same chapter
        detection logic used for .txt files.  This is more robust than trying
        to detect chapters per EPUB item, because many EPUBs pack multiple
        chapters per HTML file or use non-standard item structures.
        """
        book = epub.read_epub(filepath, options={"ignore_ncx": True})

        # ── Collect all document items in spine order ─────────────────────────
        all_docs  = {item.id: item for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)}
        spine_ids = [item_id for item_id, _ in book.spine]

        ordered = []
        seen    = set()
        for item_id in spine_ids:
            item = all_docs.get(item_id)
            if item and item.id not in seen:
                ordered.append(item)
                seen.add(item.id)
        # Catch any document items not referenced by the spine
        for item in all_docs.values():
            if item.id not in seen:
                ordered.append(item)

        print(f"[EpubHandler] Found {len(ordered)} document item(s) -- flattening to text ...")

        # ── Dump every item to plain text and concatenate ─────────────────────
        # A blank line between items ensures chapter headers sitting at the
        # start of a new HTML file land on their own line in the combined text.
        parts = []
        for item in ordered:
            text = self._html_to_text(item.get_content()).strip()
            if text:
                parts.append(text)

        full_text = "\n\n".join(parts)

        # ── Re-use TextParser's chapter splitting logic ───────────────────────
        chapters = TextParser()._split(full_text)

        if not chapters:
            print("[EpubHandler] WARNING: no chapter headers found in flattened text.")
            print("              Treating entire book as one chapter.")
            chapters = [Chapter(1, "Chapter 1", full_text)]

        print(f"[EpubHandler] Extracted {len(chapters)} chapter(s).")
        return chapters, book

    def extract_to_markdown(self, filepath: str) -> tuple:
        """
        Read an EPUB and return (markdown_string, original_book).

        The markdown uses '## Chapter N: Title' headers so it can be
        reliably re-parsed by parse_markdown_book() with exact chapter
        title preservation.
        """
        chapters, book = self.read(filepath)

        lines = [
            "---",
            f"source: {Path(filepath).name}",
            f"chapters: {len(chapters)}",
            "---",
            "",
        ]
        for ch in chapters:
            lines.append(f"## Chapter {ch.number}: {ch.title}")
            lines.append("")
            lines.append(ch.content)
            lines.append("")

        md_text = "\n".join(lines)
        md_text, removed_noise, removed_dupes = clean_master_markdown_text(md_text)

        # Keep frontmatter chapter count aligned with cleaned markdown.
        parsed_count = len(parse_markdown_book(md_text))
        md_text = re.sub(
            r"^chapters:\s*\d+\s*$",
            f"chapters: {parsed_count}",
            md_text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if removed_noise or removed_dupes:
            print(
                f"[EpubHandler] Cleanup applied: removed {removed_noise} promo/noise lines, "
                f"{removed_dupes} duplicate chapter headers."
            )
        print(f"[EpubHandler] Converted to structured markdown ({len(md_text)} chars).")
        return md_text, book

    def write(self, original_book: epub.EpubBook, all_chapters: list, output_path: str):
        """
        Write ALL chapters (original + generated) into a fresh EPUB.
        Strips markdown artifacts (* and #) before conversion.
        The original file is never modified.
        """
        book = copy.deepcopy(original_book)

        # Remove existing document items from the book copy
        existing_docs = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        for item in existing_docs:
            book.items.remove(item)
        book.spine = []
        book.toc   = []

        for ch in all_chapters:
            safe_id  = f"ch_{ch.number}"
            filename = f"{safe_id}.xhtml"
            title    = ch.title or f"Chapter {ch.number}"

            item = epub.EpubHtml(
                uid       = safe_id,
                file_name = filename,
                title     = title,
                lang      = book.language or "en",
            )
            clean_content = self._clean_for_epub(ch.content)
            item.content = self._text_to_html(title, clean_content).encode("utf-8")
            book.add_item(item)

            book.spine.append((safe_id, "yes"))
            book.toc.append(epub.Link(filename, title, safe_id))

        epub.write_epub(output_path, book)
        print(f"[EpubHandler] Saved full EPUB ({len(all_chapters)} chapters) -> {output_path}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _clean_for_epub(self, text: str) -> str:
        """Remove markdown artifacts (* and #) before EPUB export."""
        return text.replace("*", "").replace("#", "")

    def _html_to_text(self, html_bytes: bytes) -> str:
        """
        Converts EPUB XHTML to plain Unicode text.

        Uses html.parser (Python stdlib) which:
        - Handles both HTML and XHTML without namespace issues
        - Never triggers XMLParsedAsHTMLWarning
        - Returns native Python str (Unicode) -- no encoding mangling possible
        - Finds <p> tags by simple name, not namespace-qualified name

        Text is collected from <p> and heading tags only (NOT <div>) to avoid
        the duplication bug where a <div> containing <p> elements would return
        the same text twice.  If a document has no <p> tags at all (unusual
        but valid EPUB), we fall back to get_text() on the body element.
        """
        soup = BeautifulSoup(html_bytes, "html.parser")

        # Remove non-prose elements
        for tag in soup(["script", "style", "head", "nav", "aside", "figure", "figcaption"]):
            tag.decompose()

        body = soup.find("body") or soup

        # Primary strategy: collect <p> and heading tags in document order.
        # Deliberately excludes <div> to prevent nesting duplication.
        prose_tags = body.find_all(["p", "h1", "h2", "h3", "h4", "h5", "h6"])
        lines = [tag.get_text(" ", strip=True) for tag in prose_tags]
        lines = [l for l in lines if l]  # drop empty strings

        # Fallback: no <p> tags found -- grab all visible text from body
        if not lines:
            raw = body.get_text(separator="\n", strip=True)
            lines = [l.strip() for l in raw.splitlines() if l.strip()]

        # Drop U+FFFD (replacement); often appears when source bytes/entities are invalid
        return "\n\n".join(lines).replace("\ufffd", "")

    def _text_to_html(self, title: str, text: str) -> str:
        """Converts plain prose to clean well-formed XHTML."""
        def esc(s):
            return (s.replace("&", "&amp;").replace("<", "&lt;")
                     .replace(">", "&gt;").replace('"', "&quot;"))

        paras = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
        body  = "\n".join(f"    <p>{esc(p)}</p>" for p in paras)
        return (
            '<?xml version="1.0" encoding="utf-8"?>\n'
            '<!DOCTYPE html PUBLIC "-//W3C//DTD XHTML 1.1//EN"\n'
            '  "http://www.w3.org/TR/xhtml11/DTD/xhtml11.dtd">\n'
            '<html xmlns="http://www.w3.org/1999/xhtml" xml:lang="en">\n'
            "<head>\n"
            f'  <title>{esc(title)}</title>\n'
            '  <meta http-equiv="Content-Type" content="text/html; charset=utf-8"/>\n'
            "</head>\n"
            "<body>\n"
            f"  <h2>{esc(title)}</h2>\n"
            f"{body}\n"
            "</body>\n"
            "</html>"
        )


# ── Plain-text parser (for .txt input) ───────────────────────────────────────

class TextParser:
    _PATTERNS = [
        r"^(?:Chapter|CHAPTER|Ch\.?)\s*(\d+)[:\s\-\u2013\u2014]*(.*)$",
        r"^(\d+)[.:]\s+(.+)$",          # matches "3: Go! Go! Go!" and "3. Title"
        r"^(\d+):\s*(.*)$",             # matches bare "3:" with optional title
        r"^\*{1,3}\s*Chapter\s+(\d+)\s*\*{1,3}",
    ]

    def parse(self, filepath: str) -> list:
        text = Path(filepath).read_text(encoding="utf-8", errors="replace")
        chapters = self._split(text)
        if not chapters:
            chapters = [Chapter(1, "Chapter 1", text.strip())]
        print(f"[TextParser] Detected {len(chapters)} chapters.")
        return chapters

    def _split(self, text: str) -> list:
        chapters, cur_num, cur_title, cur_lines = [], None, "", []
        for line in text.splitlines():
            hit = self._match(line.strip())
            if hit:
                if cur_num is not None and cur_lines:
                    body = "\n".join(cur_lines).strip()
                    if body:
                        chapters.append(Chapter(cur_num, cur_title, body))
                cur_num, cur_title, cur_lines = hit[0], hit[1], []
            elif cur_num is not None:
                cur_lines.append(line)
        if cur_num is not None and cur_lines:
            body = "\n".join(cur_lines).strip()
            if body:
                chapters.append(Chapter(cur_num, cur_title, body))
        return chapters

    def _match(self, line: str) -> Optional[tuple]:
        for pat in self._PATTERNS:
            m = re.match(pat, line, re.IGNORECASE)
            if m:
                try:
                    num   = int(m.group(1))
                    title = (m.group(2).strip() if len(m.groups()) > 1 else "") or f"Chapter {num}"
                    return (num, title)
                except (ValueError, IndexError):
                    continue
        return None


# ── LLM client ────────────────────────────────────────────────────────────────

class LLMClient:
    def __init__(self, api_key: str):
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

    def complete(
        self,
        system: str,
        user: str,
        temperature=0.7,
        max_tokens=4096,
        top_p: Optional[float] = None,
    ) -> str:
        for attempt in range(3):
            try:
                kwargs = dict(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": user},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    extra_headers={
                        "HTTP-Referer": "https://novel-generator.local",
                        "X-Title":      "Novel Generator",
                    },
                )
                if top_p is not None:
                    kwargs["top_p"] = top_p
                resp = self.client.chat.completions.create(**kwargs)
                return resp.choices[0].message.content.strip()
            except Exception as exc:
                if attempt == 2:
                    raise
                wait = 2 ** attempt
                print(f"    [retry {attempt+1}/3] {exc} -- waiting {wait}s")
                time.sleep(wait)
        return ""


# ── Summarization engine ──────────────────────────────────────────────────────

class SummarizationEngine:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def summarize_chapter(self, ch: Chapter) -> str:
        if ch.summary:
            return ch.summary
        result = self.llm.complete(
            system=(
                "You are a precise story analyst for web novels. "
                "Create dense, information-rich summaries that miss nothing narratively significant."
            ),
            user=(
                f"Summarize Chapter {ch.number}: '{ch.title}' in 150-250 words.\n\n"
                "Include ALL of:\n"
                "- Named characters and their actions\n"
                "- Ability/power usage or progression\n"
                "- Key dialogue revelations\n"
                "- Location changes\n"
                "- Plot threads opened or advanced\n"
                "- Emotional beats and relationship shifts\n"
                "- Ending hook or cliffhanger\n\n"
                f"Chapter:\n{ch.content[:5000]}"
            ),
            temperature=0.2,
            max_tokens=500,
        )
        ch.summary = result
        return result

    def summarize_arc(self, chapters: list, arc_num: int) -> str:
        block = "\n\n".join(
            f"Chapter {c.number}: {c.summary}" for c in chapters if c.summary
        )
        return self.llm.complete(
            system="You are a story analyst creating arc-level summaries.",
            user=(
                f"Write a 500-700 word summary for Arc {arc_num} "
                f"(chapters {chapters[0].number}-{chapters[-1].number}).\n\n"
                "Cover: central conflict, character developments, power milestones, "
                "key world-building, relationship changes, threads carried forward.\n\n"
                f"Chapter summaries:\n{block}"
            ),
            temperature=0.3,
            max_tokens=1000,
        )

    def build_master_bible(self, arc_summaries: list, world_state: WorldState) -> str:
        recent_arcs = "\n\n".join(arc_summaries[-3:]) if arc_summaries else "None yet."
        ws_json     = json.dumps(asdict(world_state), indent=2)[:3500]
        return self.llm.complete(
            system="You are writing the master story bible for a web novel.",
            user=(
                "Write a 1000-1800 word master story bible with these sections:\n\n"
                "1. STORY PREMISE & GENRE\n"
                "2. PROTAGONIST(S)\n"
                "3. KEY SUPPORTING CHARACTERS\n"
                "4. ANTAGONISTS\n"
                "5. ACTIVE PLOT THREADS (numbered)\n"
                "6. WORLD RULES & POWER SYSTEM\n"
                "7. CURRENT LOCATION & SETTING\n"
                "8. NARRATIVE TRAJECTORY\n"
                "9. TONE & STYLE NOTES\n\n"
                f"World State:\n{ws_json}\n\n"
                f"Recent Arc Summaries:\n{recent_arcs}"
            ),
            temperature=0.2,
            max_tokens=2200,
        )


# ── World state manager ───────────────────────────────────────────────────────

class WorldStateManager:
    _FIELDS = set(WorldState.__dataclass_fields__.keys())

    _SCHEMA = """{
  "characters": {"name": {"role":"...","status":"...","location":"...","power_level":"...","goals":"...","relationships":"...","notes":"..."}},
  "active_plot_threads":   ["thread description (introduced ch.X)"],
  "resolved_plot_threads": [],
  "world_rules":           ["rule or power-system mechanic"],
  "locations":             {"name": "description"},
  "factions":              {"name": "description and status"},
  "key_items":             ["item and significance"],
  "current_arc":           "arc name or description",
  "story_tone":            "dark/lighthearted/action/etc.",
  "narrative_pov":         "first/third person, close/omniscient",
  "last_chapter_hook":     "ending hook of the last chapter",
  "chapter_count":         0
}"""

    def __init__(self, llm: LLMClient):
        self.llm   = llm
        self.state = WorldState()

    def _parse_json(self, raw: str) -> dict:
        raw = re.sub(r"^```(?:json)?\n?", "", raw.strip())
        raw = re.sub(r"\n?```$", "", raw).strip()
        return json.loads(raw)

    def _apply(self, data: dict, chapter_count: int):
        safe = {k: v for k, v in data.items() if k in self._FIELDS}
        self.state = WorldState(**safe)
        self.state.chapter_count = chapter_count

    def _merge_apply(self, data: dict, chapter_count: int):
        """
        Merge update data into existing state instead of replacing.
        Preserves existing non-empty values when the update returns empty ones.
        """
        old = asdict(self.state)
        merged = {}
        for k in self._FIELDS:
            new_val = data.get(k, old.get(k))
            # Preserve old value if new value is empty/blank
            old_val = old.get(k)
            if new_val in (None, "", [], {}) and old_val not in (None, "", [], {}):
                merged[k] = old_val
            else:
                merged[k] = new_val
        self.state = WorldState(**merged)
        self.state.chapter_count = chapter_count

    def _validate_state_update(self, old_state: WorldState, new_state: WorldState) -> bool:
        """
        Sanity-check that the update didn't silently wipe critical data.
        Returns True if the update looks valid.
        """
        # Characters shouldn't disappear unless explicitly killed
        if old_state.characters and not new_state.characters:
            print("  [WorldState] Validation: characters wiped — rejecting update.")
            return False
        # Active threads shouldn't vanish without resolution
        if old_state.active_plot_threads and not new_state.active_plot_threads:
            print("  [WorldState] Validation: all plot threads cleared — rejecting update.")
            return False
        # last_chapter_hook should never go blank if it was set
        if old_state.last_chapter_hook and not new_state.last_chapter_hook:
            print("  [WorldState] Validation: last_chapter_hook went blank — rejecting update.")
            return False
        return True

    def initialize(self, chapters: list) -> WorldState:
        block = "\n\n".join(
            f"Chapter {c.number}: {c.summary}" for c in chapters[-60:] if c.summary
        )
        raw = self.llm.complete(
            system="You extract structured world state data from novel summaries. Respond with ONLY valid JSON.",
            user=f"Extract current world state as JSON matching this schema:\n{self._SCHEMA}\n\nSummaries:\n{block[:6000]}",
            temperature=0.1,
            max_tokens=2200,
        )
        try:
            self._apply(self._parse_json(raw), len(chapters))
        except Exception as exc:
            print(f"  [WorldState] Init parse error: {exc} -- using empty state.")
        return self.state

    def update(self, chapter_content: str, chapter_num: int) -> WorldState:
        """Full JSON rewrite update — improved prompt with merge-safe apply."""
        raw = self.llm.complete(
            system=(
                "You update a web novel's world state JSON after a new chapter. "
                "Respond with ONLY the complete updated JSON. CRITICAL: you MUST preserve "
                "ALL existing characters, plot threads, world rules, locations, factions, "
                "and key items unless the chapter explicitly resolves or removes them. "
                "NEVER return empty lists or empty dicts for fields that currently have data."
            ),
            user=(
                f"Update state after Chapter {chapter_num}.\n\n"
                f"Current state (preserve all non-empty fields):\n"
                f"{json.dumps(asdict(self.state), indent=2)[:3000]}\n\n"
                f"New chapter content:\n{chapter_content[:6000]}\n\n"
                "Return the COMPLETE updated JSON. For each field:\n"
                "- characters: ADD new characters, UPDATE existing ones, keep all others\n"
                "- active_plot_threads: ADD new threads, REMOVE only if resolved, keep all others\n"
                "- resolved_plot_threads: append any newly resolved threads\n"
                "- world_rules: add new rules, keep existing ones\n"
                "- locations: add/update locations mentioned in this chapter\n"
                "- last_chapter_hook: MUST set to the ending hook of this chapter\n"
                "- current_arc: set if a new arc starts, otherwise keep existing\n"
                "- chapter_count: will be set programmatically (you can omit it)"
            ),
            temperature=0.1,
            max_tokens=2200,
        )
        try:
            data = self._parse_json(raw)
            old_state = asdict(self.state)
            self._merge_apply(data, chapter_num)
            if not self._validate_state_update(
                WorldState(**{k: v for k, v in old_state.items() if k in self._FIELDS}),
                self.state
            ):
                print("  [WorldState] Falling back to previous state (validation failed).")
                self.state = WorldState(**{k: v for k, v in old_state.items() if k in self._FIELDS})
                self.state.chapter_count = chapter_num
        except Exception as exc:
            print(f"  [WorldState] Update parse error: {exc} -- state unchanged.")
            self.state.chapter_count = chapter_num
        return self.state

    def update_delta(self, chapter_content: str, chapter_num: int) -> WorldState:
        """
        Request a delta/diff from the LLM instead of a full rewrite.
        The LLM returns only what changed, and we merge programmatically.
        """
        delta_schema = """{
  "add_plot_threads":    ["new thread (introduced ch.X)"],
  "remove_plot_threads": ["resolved thread text"],
  "add_characters":      {"Name": {"role":"...","status":"...","location":"...","power_level":"...","goals":"...","relationships":"...","notes":"..."}},
  "update_characters":   {"Name": {"status": "new status", "notes": "new notes"}},
  "update_locations":    {"Place": "updated description"},
  "add_world_rules":     ["new rule"],
  "add_key_items":       ["item and significance"],
  "update_factions":     {"Faction": "updated status"},
  "update_current_arc":  "new arc name (only if arc changed)",
  "update_story_tone":   "new tone (only if changed)",
  "update_last_hook":    "ending hook of this chapter (REQUIRED)",
  "update_narrative_pov": "new POV (only if changed)"
}"""
        raw = self.llm.complete(
            system=(
                "You track world state changes for a web novel. "
                "Respond with ONLY a JSON object containing ONLY the fields that changed. "
                "Do NOT include fields that did not change. "
                "Always include 'update_last_hook' with the ending hook of the chapter."
            ),
            user=(
                f"What changed after Chapter {chapter_num}?\n\n"
                f"Current state:\n{json.dumps(asdict(self.state), indent=2)[:2500]}\n\n"
                f"Chapter content:\n{chapter_content[:5000]}\n\n"
                f"Return a JSON delta (only changed fields) matching this schema:\n{delta_schema}\n\n"
                "Omit fields that did not change. Use empty arrays only for additions."
            ),
            temperature=0.1,
            max_tokens=1800,
        )
        try:
            delta = self._parse_json(raw)
        except Exception as exc:
            print(f"  [WorldState] Delta parse error: {exc}.")
            raise

        # Apply delta programmatically
        s = self.state

        for thread in delta.get("add_plot_threads", []):
            if thread not in s.active_plot_threads:
                s.active_plot_threads.append(thread)

        for thread in delta.get("remove_plot_threads", []):
            if thread in s.active_plot_threads:
                s.active_plot_threads.remove(thread)
                s.resolved_plot_threads.append(thread)

        for name, info in delta.get("add_characters", {}).items():
            if name not in s.characters:
                s.characters[name] = info

        for name, updates in delta.get("update_characters", {}).items():
            if name in s.characters:
                s.characters[name].update(updates)

        for loc, desc in delta.get("update_locations", {}).items():
            s.locations[loc] = desc

        for rule in delta.get("add_world_rules", []):
            if rule not in s.world_rules:
                s.world_rules.append(rule)

        for item in delta.get("add_key_items", []):
            if item not in s.key_items:
                s.key_items.append(item)

        for faction, status in delta.get("update_factions", {}).items():
            s.factions[faction] = status

        if delta.get("update_current_arc"):
            s.current_arc = delta["update_current_arc"]
        if delta.get("update_story_tone"):
            s.story_tone = delta["update_story_tone"]
        if delta.get("update_narrative_pov"):
            s.narrative_pov = delta["update_narrative_pov"]
        if delta.get("update_last_hook"):
            s.last_chapter_hook = delta["update_last_hook"]

        s.chapter_count = chapter_num
        print(f"  [WorldState] Delta update applied (ch.{chapter_num}).")
        return s

    def update_safely(self, chapter_content: str, chapter_num: int) -> WorldState:
        """
        Try delta update first; fall back to full rewrite if delta fails.
        """
        old_state = asdict(self.state)
        try:
            return self.update_delta(chapter_content, chapter_num)
        except Exception as exc:
            print(f"  [WorldState] Delta update failed ({exc}), trying full rewrite...")
            return self.update(chapter_content, chapter_num)

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self.state), indent=2))

    def load(self, path: Path):
        data = json.loads(path.read_text())
        self._apply(data, data.get("chapter_count", 0))


# ── Vector store ──────────────────────────────────────────────────────────────

class VectorStoreManager:
    def __init__(self, chroma_dir: str):
        print("[VectorStore] Loading embedding model (BAAI/bge-large-en-v1.5) ...")
        Settings.embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL)
        Settings.llm         = None

        self._chroma  = chromadb.PersistentClient(path=chroma_dir)
        self._col     = self._chroma.get_or_create_collection("novel_chapters")
        vs            = ChromaVectorStore(chroma_collection=self._col)
        ctx           = StorageContext.from_defaults(vector_store=vs)
        self._index   = VectorStoreIndex([], storage_context=ctx)

    def index_chapters(self, chapters: list):
        existing = set(self._col.get()["ids"])
        to_add = [
            Document(
                text     = ch.content,
                metadata = {"chapter_number": ch.number, "chapter_title": ch.title, "summary": ch.summary},
                id_      = f"chapter_{ch.number}",
            )
            for ch in chapters if f"chapter_{ch.number}" not in existing
        ]
        if to_add:
            for doc in to_add:
                self._index.insert(doc)
            print(f"  [VectorStore] Indexed {len(to_add)} new chapter(s).")

    def query(self, query_text: str, top_k: int = 6) -> list:
        retriever = self._index.as_retriever(similarity_top_k=top_k)
        nodes = retriever.retrieve(query_text)
        return [
            {
                "chapter_number": n.metadata.get("chapter_number"),
                "chapter_title":  n.metadata.get("chapter_title"),
                "summary":        n.metadata.get("summary", ""),
                "excerpt":        n.text[:650],
                "score":          round(n.score or 0, 4),
            }
            for n in nodes
        ]


# ── Chapter planner ───────────────────────────────────────────────────────────

class ChapterPlanner:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def plan(self, chapter_num, master_bible, world_state, recent_summaries, rag_results, arc_summary="") -> str:
        ws       = world_state
        rag_ctx  = "\n\n".join(
            f"[Related scene - Ch.{r['chapter_number']}]:\n{r['excerpt']}" for r in rag_results[:3]
        )
        recent_block = "\n".join(f"- {s}" for s in recent_summaries[-6:])
        return self.llm.complete(
            system=(
                "You are a master web novel story planner. You create focused, scene-dense chapter plans "
                "that DWELL in each location rather than jumping between many places. "
                "You prioritize depth over breadth: fewer scenes, more beats per scene, "
                "more dialogue and character interaction within each setting. "
                "Every scene must move story, character, or mystery forward with specific on-page events."
            ),
            user=(
                f"Plan Chapter {chapter_num} in detail.\n\n"
                f"STORY BIBLE:\n{master_bible[:8000]}\n\n"
                f"CURRENT STATE:\n"
                f"  Last hook:      {ws.last_chapter_hook}\n"
                f"  Current arc:    {ws.current_arc}\n"
                f"  Active threads: {json.dumps(ws.active_plot_threads[:8])}\n"
                f"  Key characters: {json.dumps(list(ws.characters.keys())[:8])}\n\n"
                f"RECENT SUMMARIES:\n{recent_block}\n\n"
                f"RELEVANT PAST SCENES:\n{rag_ctx}\n\n"
                "CRITICAL PACING RULES:\n"
                "- Use only 2-3 scenes maximum. STAY in each location long enough for the reader "
                "to feel grounded. Do not hop between 5+ locations in one chapter.\n"
                "- Each scene must have 4-6 concrete plot beats — not vague 'tension builds' but "
                "specific dialogue exchanges, decisions, reveals, and actions.\n"
                "- Advance at least TWO distinct threads or open questions this chapter.\n"
                "- Each scene needs a complication, reversal, reveal, or decision.\n"
                "- Specify on-page dialogue and conflict; name who says what and what is at stake.\n\n"
                "Produce a plan with ALL these sections:\n"
                "1. OPENING HOOK (concrete situation + why the reader cares)\n"
                "2. SCENE BREAKDOWN — 2-3 scenes ONLY. For EACH scene list:\n"
                "   - Location\n"
                "   - Narrative function (what changes because of this scene)\n"
                "   - 4-6 PLOT BEATS (specific events, choices, or information — not 'tension builds')\n"
                "   - Rough prose target for that scene (e.g. 300-500 words) so the full chapter totals ~1000 words\n"
                "3. THREAD ADVANCEMENT (map each active thread to scenes where it moves)\n"
                "4. CHARACTER MOMENT (interiority + relationship shift with evidence)\n"
                "5. CONFLICT/ACTION (physical, social, or ideological — beat by beat)\n"
                "6. REVELATIONS (what the reader or cast learns, and consequences)\n"
                "7. ENDING HOOK\n"
                "8. TONE GUIDANCE\n"
                "9. CONTINUITY TRAPS (lore/voice risks to avoid)\n"
                "10. ON-PAGE REQUIREMENTS — bullet list of moments that MUST be dramatized "
                "(e.g. a full exchange, a confrontation, a discovery), not summarized away\n"
            ),
            temperature=0.75,
            max_tokens=2400,
        )

    def plan_part_two(
        self,
        chapter_num,
        master_bible,
        world_state,
        recent_summaries,
        rag_results,
        first_plan,
    ) -> str:
        """
        Create the continuation plan for "Chapter N Part 2".
        Prompted as if planning the next chapter beat progression, but it is
        explicitly framed as the second half/part of the same chapter number.
        """
        ws = world_state
        rag_ctx = "\n\n".join(
            f"[Related scene - Ch.{r['chapter_number']}]:\n{r['excerpt']}" for r in rag_results[:3]
        )
        recent_block = "\n".join(f"- {s}" for s in recent_summaries[-6:])
        next_ch_num = chapter_num + 1
        return self.llm.complete(
            system=(
                "You are a master web novel story planner. "
                "You are creating a continuation plan that reads like the next chapter's escalation, "
                "but it must be written as PART 2 of the same chapter number."
            ),
            user=(
                f"Create a continuation plan titled 'Chapter {chapter_num} Part 2'.\n"
                f"Treat progression/intensity as if you are outlining Chapter {next_ch_num}, "
                f"but this must continue directly from Chapter {chapter_num} and stay in one continuous flow.\n\n"
                f"STORY BIBLE:\n{master_bible[:8000]}\n\n"
                f"CURRENT STATE:\n"
                f"  Last hook:      {ws.last_chapter_hook}\n"
                f"  Current arc:    {ws.current_arc}\n"
                f"  Active threads: {json.dumps(ws.active_plot_threads[:8])}\n"
                f"  Key characters: {json.dumps(list(ws.characters.keys())[:8])}\n\n"
                f"RECENT SUMMARIES:\n{recent_block}\n\n"
                f"RELEVANT PAST SCENES:\n{rag_ctx}\n\n"
                f"CHAPTER {chapter_num} PLAN (PART 1 CONTEXT):\n{first_plan[:3000]}\n\n"
                "Requirements:\n"
                "- Continue directly from where Part 1 should leave off; no recap intro.\n"
                "- Keep scene density high and beats concrete.\n"
                "- Use 2-3 scenes maximum with 4-6 specific beats each.\n"
                "- End with a strong hook that can close the full chapter cleanly.\n\n"
                "Produce the same section structure as the main chapter plan:\n"
                "1. OPENING HOOK\n"
                "2. SCENE BREAKDOWN\n"
                "3. THREAD ADVANCEMENT\n"
                "4. CHARACTER MOMENT\n"
                "5. CONFLICT/ACTION\n"
                "6. REVELATIONS\n"
                "7. ENDING HOOK\n"
                "8. TONE GUIDANCE\n"
                "9. CONTINUITY TRAPS\n"
                "10. ON-PAGE REQUIREMENTS\n"
            ),
            temperature=0.75,
            max_tokens=2400,
        )


# ── Chapter writer ────────────────────────────────────────────────────────────

class ChapterWriter:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def write(self, chapter_num, plan, master_bible, world_state, recent_chapters, rag_results) -> str:
        ws          = world_state
        recent_prose = "".join(
            f"\n\n--- Chapter {c.number}: {c.title} ---\n{c.content[:4500]}"
            for c in recent_chapters[-RECENT_FULL_COUNT:]
        )
        rag_ctx = "\n\n".join(
            f"[Relevant past scene - Ch.{r['chapter_number']}]:\n{r['excerpt']}"
            for r in rag_results[:3]
        )
        min_w, max_w = TARGET_WORDS
        return self.llm.complete(
            system=(
                "You are an expert web novel author. Your prose is immersive, character-driven, "
                "and stylistically consistent. You write with varied sentence rhythm, sensory detail, "
                "and genuine emotional depth. Your work never sounds robotic or formulaic.\n\n"
                "ABSOLUTE RULES:\n"
                f"- Narrative POV: {ws.narrative_pov or 'match existing style'}\n"
                f"- Tone: {ws.story_tone or 'match existing style'}\n"
                f"- HARD MINIMUM LENGTH: at least 700 words. "
                f"If you are short, extend with dialogue, interior thought, blocking, and secondary beats—"
                f"do not pad with repetition or meta commentary.\n"
                f"- Target band: {min_w}-{max_w} words.\n"
                "- DWELL in each scene. Do not rush through locations. Give the reader time to "
                "sink into the setting, hear the dialogue, feel the atmosphere.\n"
                "- Execute every scene and beat from the chapter plan on the page—no skipped, summarized, "
                "or off-screen resolutions.\n"
                "- Never contradict established lore or character abilities\n"
                "- No recapping -- dive straight into the scene\n"
                "- Show, don't tell -- action and dialogue over description dumps\n"
                "- End on the planned hook\n"
                "- Do not output markdown format — no *text* for italics, write only plain text\n"
                "- Each character speaks and acts consistently with their established personality\n"
            ),
            user=(
                f"Write Chapter {chapter_num} in full.\n\n"
                f"STORY BIBLE:\n{master_bible[:1800]}\n\n"
                f"CHAPTER PLAN:\n{plan}\n\n"
                f"WORLD STATE:\n"
                f"  Last hook:      {ws.last_chapter_hook}\n"
                f"  Active threads: {json.dumps(ws.active_plot_threads[:6])}\n"
                f"  World rules:    {json.dumps(ws.world_rules[:5])}\n\n"
                f"RECENT CHAPTERS (continuity):\n{recent_prose[:5000]}\n\n"
                f"RELEVANT PAST SCENES (lore consistency):\n{rag_ctx[:1800]}\n\n"
                f"Deliver complete chapter prose now. Minimum 900 words; "
                f"target {min_w}-{max_w}. Stay in each scene long — do not rush between locations. "
                f"Fully dramatize each planned scene with dialogue, action, and consequences on the page. "
                f"Begin immediately with the story (no preamble)."
            ),
            temperature=0.85,
            max_tokens=8000,
            top_p=0.92,
        )


# ── Main orchestrator ─────────────────────────────────────────────────────────

class NovelGenerator:
    def __init__(self, api_key: str, book_id: str):
        print("=" * 62)
        print(f"  Novel Generator  |  book: {book_id}")
        print("=" * 62)

        self.book_id = book_id
        self.paths   = book_paths(book_id)
        self.paths["root"].mkdir(parents=True, exist_ok=True)
        self.paths["gen_dir"].mkdir(exist_ok=True)
        self.paths["chroma"].mkdir(parents=True, exist_ok=True)

        self.llm          = LLMClient(api_key)
        self.epub_handler = EpubHandler()
        self.summarizer   = SummarizationEngine(self.llm)
        self.world_mgr    = WorldStateManager(self.llm)
        self.vec_store    = VectorStoreManager(str(self.paths["chroma"]))
        self.planner      = ChapterPlanner(self.llm)
        self.writer       = ChapterWriter(self.llm)

        self.chapters:       list           = []
        self.arc_summaries:  list           = []
        self.master_bible:   str            = ""
        self._original_epub                 = None
        self._input_is_epub: bool           = False

    # ── Resume helpers ────────────────────────────────────────────────────────

    def _merged_epub_path(self, user_path: str) -> Optional[Path]:
        """`{stem}_with_generated.epub` in the book folder, if present (EPUB only)."""
        if Path(user_path).suffix.lower() != ".epub":
            return None
        p = self.paths["root"] / f"{Path(user_path).stem}_with_generated.epub"
        return p if p.exists() else None

    def _rebuild_arcs_from_index(self, first_dirty_arc_idx: int):
        """Re-summarise arcs from first_dirty_arc_idx onward (0-based)."""
        n = len(self.chapters)
        if n == 0:
            return
        num_arcs = (n + ARC_SIZE - 1) // ARC_SIZE
        first = max(0, first_dirty_arc_idx)
        for arc_idx in range(first, num_arcs):
            sl = self.chapters[arc_idx * ARC_SIZE : (arc_idx + 1) * ARC_SIZE]
            if not sl:
                break
            print(f"  Rebuilding arc summary {arc_idx + 1}/{num_arcs} (ch.{sl[0].number}-{sl[-1].number}) ...")
            sm = self.summarizer.summarize_arc(sl, arc_idx + 1)
            if arc_idx < len(self.arc_summaries):
                self.arc_summaries[arc_idx] = sm
            else:
                self.arc_summaries.append(sm)
        if len(self.arc_summaries) > num_arcs:
            self.arc_summaries[:] = self.arc_summaries[:num_arcs]

    def _ingest_new_chapters_after_resume(self, fresh: list, n_saved: int) -> None:
        """
        Append chapters that exist in the source file but not in saved session state.
        Assumes document order matches: fresh[:n_saved] aligns with prior session.
        """
        extra = fresh[n_saved:]
        if not extra:
            return
        first_arc = n_saved // ARC_SIZE
        print(f"  Ingesting {len(extra)} new source chapter(s) (arc rebuild from arc {first_arc + 1}) ...")
        for ch in extra:
            print(f"    New Ch.{ch.number}: {ch.title[:50]}..." if len(ch.title) > 50 else f"    New Ch.{ch.number}: {ch.title}")
            self.summarizer.summarize_chapter(ch)
            self.vec_store.index_chapters([ch])
            self.world_mgr.update_safely(ch.content, ch.number)
            self.world_mgr.save(self.paths["state"])
            self._append_chapter_for_session(ch)
        self._rebuild_arcs_from_index(first_arc)
        print("  Refreshing master bible after new source material ...")
        self.master_bible = self.summarizer.build_master_bible(
            self.arc_summaries, self.world_mgr.state
        )
        self.paths["bible"].write_text(self.master_bible)
        self._save_progress()
        print(f"  Session updated — {len(self.chapters)} total chapter(s) in canon.")

    def _append_chapter_for_session(self, ch: Chapter) -> None:
        """Append to session canon and drop full text from chapters outside the recent window."""
        self.chapters.append(ch)
        n = len(self.chapters)
        if n > RECENT_FULL_COUNT:
            for j in range(n - RECENT_FULL_COUNT):
                c = self.chapters[j]
                if c.content:
                    self.chapters[j] = Chapter(c.number, c.title, "", c.summary)

    def _append_to_master_book(self, ch: Chapter) -> None:
        """Append a new chapter to the master book markdown file."""
        md_path = self.paths["master_book"]
        new_block = f"\n\n## Chapter {ch.number}: {ch.title}\n\n{ch.content}\n"
        with open(md_path, "a") as f:
            f.write(new_block)

    def _save_progress(self):
        """Persist everything needed to resume without re-processing."""
        # Chapter summaries (lightweight -- no full content except recent)
        ch_data = [{"number": c.number, "title": c.title, "summary": c.summary}
                   for c in self.chapters]
        self.paths["ch_summaries"].write_text(json.dumps(ch_data, indent=2))

        # Arc summaries
        self.paths["arc_summaries"].write_text(json.dumps(self.arc_summaries, indent=2))

        # Last RECENT_FULL_COUNT chapters with full content (needed by writer)
        recent = [{"number": c.number, "title": c.title,
                   "content": c.content, "summary": c.summary}
                  for c in self.chapters[-RECENT_FULL_COUNT:]]
        self.paths["recent_chs"].write_text(json.dumps(recent, indent=2))

    def _load_resume(self, filepath: str):
        """
        Restore state from disk. Uses master_book.md as the source of truth
        if it exists, otherwise falls back to the original epub/txt file.
        """
        print("  Existing session detected -- loading saved state ...")

        # Chapter summaries -> lightweight Chapter stubs
        ch_data = json.loads(self.paths["ch_summaries"].read_text())
        self.chapters = [Chapter(number=d["number"], title=d["title"],
                                 content="", summary=d["summary"])
                         for d in ch_data]

        recent_data = json.loads(self.paths["recent_chs"].read_text())
        recent_map  = {d["number"]: d for d in recent_data}
        for i, ch in enumerate(self.chapters):
            if ch.number in recent_map:
                d = recent_map[ch.number]
                self.chapters[i] = Chapter(number=d["number"], title=d["title"],
                                           content=d["content"], summary=d["summary"])

        self.arc_summaries = json.loads(self.paths["arc_summaries"].read_text())
        self.world_mgr.load(self.paths["state"])
        self.master_bible = self.paths["bible"].read_text()

        n_saved = len(self.chapters)

        # Check for new source material from master_book.md or original file
        master_md = self.paths["master_book"]
        if master_md.exists():
            print("  Using master_book.md as source of truth.")
            fresh = parse_markdown_book(master_md.read_text())
            self._input_is_epub = Path(filepath).suffix.lower() == ".epub"
            if self._input_is_epub:
                try:
                    _, self._original_epub = self.epub_handler.read(filepath)
                except Exception:
                    self._original_epub = None
            else:
                self._original_epub = None
            if len(fresh) > n_saved:
                self._ingest_new_chapters_after_resume(fresh, n_saved)
            elif len(fresh) < n_saved:
                print(
                    f"  [WARN] master_book.md has {len(fresh)} chapter(s) but session has {n_saved} — "
                    "keeping saved metadata."
                )
        else:
            # Legacy path: no master_book.md yet, re-parse from source
            ext = Path(filepath).suffix.lower()
            self._input_is_epub = (ext == ".epub")
            if self._input_is_epub:
                merged = self._merged_epub_path(filepath)
                epub_read = str(merged) if merged else filepath
                if merged:
                    print(f"  Using merged EPUB: {merged.name}")
                    print("    (includes earlier generated chapters; add new source chapters here to continue the story)")
                fresh, self._original_epub = self.epub_handler.read(epub_read)
                if len(fresh) > n_saved:
                    self._ingest_new_chapters_after_resume(fresh, n_saved)
                elif len(fresh) < n_saved:
                    print(
                        f"  [WARN] EPUB has {len(fresh)} chapter(s) but session has {n_saved} — "
                        "keeping saved metadata. Fix the EPUB or delete session JSON to re-index."
                    )
            else:
                self._original_epub = None
                fresh = TextParser().parse(filepath)
                if len(fresh) > n_saved:
                    self._ingest_new_chapters_after_resume(fresh, n_saved)
                elif len(fresh) < n_saved:
                    print(
                        f"  [WARN] Text file has {len(fresh)} chapter(s) but session has {n_saved} — "
                        "keeping saved metadata."
                    )

        total = len(self.chapters)
        print(f"  Resumed: {total} chapter(s) in canon.")
        print(f"  Last world state: chapter {self.world_mgr.state.chapter_count}")
        print(f"  Generating from Chapter {self.world_mgr.state.chapter_count + 1}.")

    # ── Phase 1: load novel ───────────────────────────────────────────────────

    def load_novel(self, filepath: str):
        # ── Resume check: if saved state exists, skip all LLM processing ─────
        if self.paths["ch_summaries"].exists():
            print(f"\n[RESUME] Book '{self.book_id}' has existing session data.")
            self._load_resume(filepath)
            print(f"{'_'*62}\n")
            return

        # ── First run: full Phase 1 pipeline ─────────────────────────────────
        ext = Path(filepath).suffix.lower()
        self._input_is_epub = (ext == ".epub")

        print(f"\n[1/7] Parsing novel ({ext}): {filepath}")
        if self._input_is_epub:
            md_text, self._original_epub = self.epub_handler.extract_to_markdown(filepath)
            self.paths["master_book"].write_text(md_text)
            print(f"  Saved master_book.md -> {self.paths['master_book']}")
            self.chapters = parse_markdown_book(md_text)
        else:
            self.chapters = TextParser().parse(filepath)
            # Build master_book.md from parsed chapters
            lines = ["---", f"source: {Path(filepath).name}", f"chapters: {len(self.chapters)}", "---", ""]
            for ch in self.chapters:
                lines.append(f"## Chapter {ch.number}: {ch.title}")
                lines.append("")
                lines.append(ch.content)
                lines.append("")
            self.paths["master_book"].write_text("\n".join(lines))
            print(f"  Saved master_book.md -> {self.paths['master_book']}")

        print(f"\n[2/7] Summarising {len(self.chapters)} chapters ...")
        for ch in self.chapters:
            print(f"  Ch.{ch.number}/{self.chapters[-1].number} ...", end="\r")
            self.summarizer.summarize_chapter(ch)
        print(f"  All {len(self.chapters)} chapters summarised.           ")

        print(f"\n[3/7] Building arc summaries ...")
        arcs = [self.chapters[i:i + ARC_SIZE] for i in range(0, len(self.chapters), ARC_SIZE)]
        for idx, arc in enumerate(arcs, 1):
            print(f"  Arc {idx}/{len(arcs)} (ch.{arc[0].number}-{arc[-1].number}) ...")
            self.arc_summaries.append(self.summarizer.summarize_arc(arc, idx))

        print(f"\n[4/7] Extracting world state ...")
        self.world_mgr.initialize(self.chapters)
        self.world_mgr.save(self.paths["state"])
        print(f"  Saved -> {self.paths['state']}")

        print(f"\n[5/7] Building master bible ...")
        self.master_bible = self.summarizer.build_master_bible(
            self.arc_summaries, self.world_mgr.state
        )
        self.paths["bible"].write_text(self.master_bible)
        print(f"  Saved -> {self.paths['bible']}")

        print(f"\n[6/7] Indexing chapters in ChromaDB ...")
        self.vec_store.index_chapters(self.chapters)
        print(f"  Vector index ready.")

        # ── Save everything needed for future resume ──────────────────────────
        print(f"\n[7/7] Saving session state ...")
        self._save_progress()
        print(f"  Session state saved.")

        print(f"\n{'_'*62}")
        print(f"  Novel loaded. Generating from Chapter {len(self.chapters)+1}.")
        print(f"{'_'*62}\n")

    # ── Phase 2: generate chapters ────────────────────────────────────────────

    def generate_chapters(self, num_chapters: int, source_path: str):
        # Use world_state.chapter_count as the authoritative "last processed"
        # number. On a fresh run this equals len(self.chapters). On a resume
        # it reflects all previously generated chapters, not just the source.
        start_num    = self.world_mgr.state.chapter_count + 1
        all_chapters = list(self.chapters)
        new_chapters = []

        print(f"Generating {num_chapters} chapter(s) starting from Chapter {start_num}\n")

        for i in range(num_chapters):
            ch_num = start_num + i
            print(f"-- Chapter {ch_num} " + "-" * 44)

            ws        = self.world_mgr.state
            rag_query = " ".join(filter(None, [
                ws.last_chapter_hook, ws.current_arc, *list(ws.characters.keys())[:4]
            ]))

            print("  [1/5] RAG retrieval ...")
            rag_results = self.vec_store.query(rag_query, top_k=6)

            recent_summaries = [
                f"Ch.{c.number}: {c.summary}" for c in all_chapters[-8:] if c.summary
            ]

            print("  [2/5] Planning chapter ...")
            plan = self.planner.plan(
                chapter_num      = ch_num,
                master_bible     = self.master_bible,
                world_state      = ws,
                recent_summaries = recent_summaries,
                rag_results      = rag_results,
                arc_summary      = self.arc_summaries[-1] if self.arc_summaries else "",
            )
            (self.paths["gen_dir"] / f"chapter_{ch_num:04d}_plan.md").write_text(
                f"# Chapter {ch_num} - Plan\n\n{plan}"
            )
            part2_plan = self.planner.plan_part_two(
                chapter_num      = ch_num,
                master_bible     = self.master_bible,
                world_state      = ws,
                recent_summaries = recent_summaries,
                rag_results      = rag_results,
                first_plan       = plan,
            )
            (self.paths["gen_dir"] / f"chapter_{ch_num:04d}_part2_plan.md").write_text(
                f"# Chapter {ch_num} Part 2 - Plan\n\n{part2_plan}"
            )

            print("  [3/5] Writing chapter prose ...")
            part1_prose = self.writer.write(
                chapter_num     = ch_num,
                plan            = plan,
                master_bible    = self.master_bible,
                world_state     = ws,
                recent_chapters = all_chapters[-RECENT_FULL_COUNT:],
                rag_results     = rag_results,
            )
            part1_stub = Chapter(number=ch_num, title=f"Chapter {ch_num}", content=part1_prose)
            part2_prose = self.writer.write(
                chapter_num     = f"{ch_num} Part 2",
                plan            = part2_plan,
                master_bible    = self.master_bible,
                world_state     = ws,
                recent_chapters = all_chapters[-(RECENT_FULL_COUNT - 1):] + [part1_stub],
                rag_results     = rag_results,
            )
            prose = (part1_prose.rstrip() + "\n\n" + part2_prose.lstrip()).strip()

            new_ch = Chapter(number=ch_num, title=f"Chapter {ch_num}", content=prose)

            print("  [4/5] Appending to master book & updating world state ...")
            self._append_to_master_book(new_ch)
            self.summarizer.summarize_chapter(new_ch)
            self.world_mgr.update_safely(prose, ch_num)
            self.world_mgr.save(self.paths["state"])
            self.vec_store.index_chapters([new_ch])

            all_chapters.append(new_ch)
            new_chapters.append(new_ch)
            self._append_chapter_for_session(new_ch)

            if ch_num % ARC_SIZE == 0:
                print("  [~]   Arc boundary -- summarising new arc ...")
                arc_idx = (ch_num // ARC_SIZE) - 1
                sm = self.summarizer.summarize_arc(
                    all_chapters[-ARC_SIZE:], arc_idx + 1
                )
                if arc_idx < len(self.arc_summaries):
                    self.arc_summaries[arc_idx] = sm
                else:
                    self.arc_summaries.append(sm)

            (self.paths["gen_dir"] / f"chapter_{ch_num:04d}.md").write_text(
                f"# Chapter {ch_num}\n\n{prose}\n\n---\n*{new_ch.word_count} words*"
            )
            (self.paths["gen_dir"] / f"chapter_{ch_num:04d}_part2.md").write_text(
                f"# Chapter {ch_num} Part 2\n\n{part2_prose}"
            )

            # Update resume state so next session knows about this chapter
            self._save_progress()

            if ch_num % BIBLE_REFRESH_EVERY == 0:
                print("  [~]   Refreshing master bible ...")
                self.master_bible = self.summarizer.build_master_bible(
                    self.arc_summaries, self.world_mgr.state
                )
                self.paths["bible"].write_text(self.master_bible)

            print(f"  [5/5] Done  ({new_ch.word_count} words)")

        # ── Write combined markdown (new chapters only) ───────────────────────
        combined = "\n\n---\n\n".join(
            f"# Chapter {c.number}\n\n{c.content}" for c in new_chapters
        )
        self.paths["combined"].write_text(combined)

        # ── Write EPUB output from full master book ───────────────────────────
        out_epub = None
        if self._input_is_epub and self._original_epub is not None:
            stem     = Path(source_path).stem
            out_epub = self.paths["root"] / f"{stem}_with_generated.epub"
            # Parse ALL chapters from master_book.md for a complete EPUB
            master_md = self.paths["master_book"]
            if master_md.exists():
                all_for_epub = parse_markdown_book(master_md.read_text())
            else:
                all_for_epub = all_chapters
            self.epub_handler.write(self._original_epub, all_for_epub, str(out_epub))

        print(f"\n{'='*62}")
        print(f"  {num_chapters} chapter(s) generated for book '{self.book_id}'")
        print(f"  Book folder : {self.paths['root'].resolve()}")
        if out_epub:
            print(f"  EPUB output : {out_epub.name}")
        print(f"  Markdown    : {self.paths['combined'].name}")
        print(f"{'='*62}\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Generate web novel chapters -- EPUB/TXT input, EPUB output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python novel_generator.py novel.epub 3\n"
            "  python novel_generator.py novel.epub 5 --book-id my_story\n"
            "  python novel_generator.py novel.txt  10 --api-key sk-or-xxxx\n"
        ),
    )
    ap.add_argument("novel_file",   help=".epub or .txt novel file")
    ap.add_argument("num_chapters", type=int, help="Number of new chapters to generate")
    ap.add_argument("--api-key",    help="OpenRouter API key (overrides env var)")
    ap.add_argument(
        "--book-id",
        help=(
            "Unique slug for this book (default: derived from filename). "
            "Use the same slug across sessions to continue a book."
        ),
    )
    args = ap.parse_args()

    api_key = args.api_key or OPENROUTER_API_KEY
    if not api_key:
        ap.error(
            "OpenRouter API key required.\n"
            "  export OPENROUTER_API_KEY=sk-or-xxxx\n"
            "  or: --api-key sk-or-xxxx"
        )

    if not Path(args.novel_file).exists():
        ap.error(f"File not found: {args.novel_file}")

    if args.num_chapters < 1:
        ap.error("num_chapters must be >= 1")

    book_id = args.book_id or slug(Path(args.novel_file).stem)
    print(f"Book ID resolved to: {book_id}")

    gen = NovelGenerator(api_key, book_id)
    gen.load_novel(args.novel_file)
    gen.generate_chapters(args.num_chapters, args.novel_file)


if __name__ == "__main__":
    main()
