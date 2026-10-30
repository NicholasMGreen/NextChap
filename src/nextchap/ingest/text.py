"""Chapter detection shared by TXT and EPUB input.

The input is a sequence of blocks (lines or paragraphs). A block is "strong" when its
position suggests a heading: an HTML heading, the first block of an EPUB document, or a
text line preceded by a blank line. Bare-number headers ("12: Title") are only accepted on
strong blocks and must continue the numbering, so prose like "3. He ran to the door." or
"1984: the year it began" is not mistaken for a chapter break.
"""

from __future__ import annotations

import itertools
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from nextchap.ingest.cleanup import NoiseFilter, clean_chapter_text
from nextchap.ingest.master_book import Chapter

MAX_KEYWORD_HEADER_CHARS = 120
MAX_BARE_HEADER_CHARS = 80
MAX_BARE_TITLE_WORDS = 12
SHORT_CHAPTER_WORDS = 150


@dataclass
class Block:
    text: str
    strong: bool = False
    heading: bool = False


@dataclass
class DetectionReport:
    chapters: list[Chapter]
    anomalies: list[str] = field(default_factory=list)
    renumbered: bool = False
    preamble_words: int = 0
    removed_noise: int = 0


# ── Number parsing ────────────────────────────────────────────────────────────

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100, "d": 500, "m": 1000}
_ROMAN_RE = re.compile(r"^m{0,3}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})$", re.IGNORECASE)

_UNITS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
}
_TENS = {
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
}

_CN_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CN_UNITS = {"十": 10, "百": 100, "千": 1000}


def roman_to_int(s: str) -> int | None:
    if not s or not _ROMAN_RE.fullmatch(s):
        return None
    s = s.lower()
    total = 0
    for i, ch in enumerate(s):
        v = _ROMAN[ch]
        if i + 1 < len(s) and _ROMAN[s[i + 1]] > v:
            total -= v
        else:
            total += v
    return total or None


def words_to_int(s: str) -> int | None:
    tokens = [t for t in re.split(r"[\s-]+", s.lower().strip()) if t and t != "and"]
    if not tokens:
        return None
    total, current = 0, 0
    for t in tokens:
        if t in _UNITS:
            current += _UNITS[t]
        elif t in _TENS:
            current += _TENS[t]
        elif t == "hundred":
            current = (current or 1) * 100
        elif t == "thousand":
            total += (current or 1) * 1000
            current = 0
        else:
            return None
    value = total + current
    return value if value > 0 else None


def chinese_to_int(s: str) -> int | None:
    if s.isdigit():
        return int(s)
    total, digit = 0, 0
    for ch in s:
        if ch in _CN_DIGITS:
            digit = _CN_DIGITS[ch]
        elif ch in _CN_UNITS:
            total += (digit or 1) * _CN_UNITS[ch]
            digit = 0
        else:
            return None
    value = total + digit
    return value or None


# ── Header matching ───────────────────────────────────────────────────────────

_KEYWORD_RE = re.compile(r"^(?:chapter|ch\.)\s*(.+)$", re.IGNORECASE)
_SPECIAL_RE = re.compile(
    r"^(prologue|epilogue|interlude|side\s+story|afterword)\b[\s:.\-\u2013\u2014]*(.*)$",
    re.IGNORECASE,
)
_BARE_RE = re.compile(r"^(\d{1,4})\s*(?:[.:)\-\u2013\u2014]\s*(.*))?$")
_CN_RE = re.compile(r"^第\s*([0-9零〇一二三四五六七八九十百千两]+)\s*[章回节]\s*(.*)$")
_TITLE_LEAD_RE = re.compile(r"^[\s:.\-\u2013\u2014|]+")


@dataclass
class _Header:
    number: int | None
    title: str
    bare: bool = False


def _clean_title(t: str) -> str:
    return _TITLE_LEAD_RE.sub("", t).strip()


def _strip_decoration(line: str) -> str:
    return line.strip().strip("*#_= ").strip()


def _match_keyword(s: str) -> _Header | None:
    m = _KEYWORD_RE.match(s)
    if not m:
        return None
    rest = m.group(1).strip()
    num = re.match(r"^(\d+)\b(.*)$", rest)
    if num:
        return _Header(int(num.group(1)), _clean_title(num.group(2)))
    roman = re.match(r"^([ivxlcdm]+)\b(.*)$", rest, re.IGNORECASE)
    if roman and (value := roman_to_int(roman.group(1))):
        return _Header(value, _clean_title(roman.group(2)))
    words = rest.split()
    for k in range(min(5, len(words)), 0, -1):
        candidate = " ".join(words[:k]).rstrip(":.-\u2013\u2014,")
        value = words_to_int(candidate)
        if value:
            return _Header(value, _clean_title(" ".join(words[k:])))
    return None


def _looks_like_sentence(title: str, strong: bool) -> bool:
    words = len(title.split())
    if title.endswith((".", ",", ";")) and words > 6:
        return True
    return not strong and words > 8


def match_header(block: Block, last_number: int | None) -> _Header | None:
    s = _strip_decoration(block.text)
    if not s or "\n" in s:
        return None

    if len(s) <= MAX_KEYWORD_HEADER_CHARS:
        hit = _match_keyword(s)
        if hit and not _looks_like_sentence(hit.title, block.strong):
            return hit
        cn = _CN_RE.match(s)
        if cn and (value := chinese_to_int(cn.group(1))):
            return _Header(value, _clean_title(cn.group(2)))

    if not block.strong or len(s) > MAX_BARE_HEADER_CHARS:
        return None

    special = _SPECIAL_RE.match(s)
    if special and len(s.split()) <= MAX_BARE_TITLE_WORDS:
        label = special.group(1).title()
        sub = _clean_title(special.group(2))
        return _Header(None, f"{label}: {sub}" if sub else label)

    bare = _BARE_RE.match(s)
    if not bare:
        return None
    number = int(bare.group(1))
    title = _clean_title(bare.group(2) or "")
    if len(title.split()) > MAX_BARE_TITLE_WORDS:
        return None
    if title.endswith(".") and len(title.split()) > 4:
        return None
    expected_ok = number == 1 if last_number is None else last_number < number <= last_number + 2
    if not expected_ok and not (block.heading and last_number is None):
        return None
    return _Header(number, title, bare=True)


# ── Detection ─────────────────────────────────────────────────────────────────


@dataclass
class _Draft:
    number: int | None
    title: str
    body: list[str] = field(default_factory=list)


def detect_chapters(
    blocks: Iterable[Block], joiner: str = "\n\n", noise: NoiseFilter | None = None
) -> DetectionReport:
    drafts: list[_Draft] = []
    preamble_words = 0
    last_number: int | None = None

    for block in blocks:
        hit = match_header(block, last_number)
        if hit is None:
            if drafts:
                drafts[-1].body.append(block.text)
            else:
                preamble_words += len(block.text.split())
            continue

        cur = drafts[-1] if drafts else None
        if cur is not None and hit.number is not None and hit.number == cur.number:
            # Repeated header (e.g. <h1>Chapter 3</h1> then "Chapter 3: Title"), or a
            # running header inside the chapter. Keep the more descriptive title.
            if not "".join(cur.body).strip() and hit.title and not cur.title:
                cur.title = hit.title
            continue
        drafts.append(_Draft(hit.number, hit.title))
        if hit.number is not None:
            last_number = hit.number
        elif last_number is not None:
            last_number += 1

    noise = noise or NoiseFilter()
    report = DetectionReport(chapters=[], preamble_words=preamble_words)
    kept: list[_Draft] = []
    for d in drafts:
        body, removed = clean_chapter_text(joiner.join(d.body), noise)
        report.removed_noise += removed
        if body:
            d.body = [body]
            kept.append(d)

    numbers = [d.number for d in kept]
    increasing = all(n is not None for n in numbers) and all(
        a is not None and b is not None and b > a for a, b in itertools.pairwise(numbers)
    )
    if not increasing and kept:
        report.renumbered = True
        report.anomalies.append(
            "Chapter numbers were missing or not increasing (special chapters or a numbering "
            f"restart); renumbered sequentially 1-{len(kept)}."
        )
    for idx, d in enumerate(kept, 1):
        number = idx if report.renumbered else int(d.number or idx)
        title = d.title or f"Chapter {number}"
        report.chapters.append(Chapter(number, title, d.body[0]))

    if not report.renumbered:
        for a, b in zip(report.chapters, report.chapters[1:], strict=False):
            if b.number - a.number > 1:
                report.anomalies.append(
                    f"Gap in numbering between chapter {a.number} and {b.number}."
                )
    short = [c.number for c in report.chapters if c.word_count < SHORT_CHAPTER_WORDS]
    if short:
        listed = ", ".join(map(str, short[:8])) + ("…" if len(short) > 8 else "")
        report.anomalies.append(
            f"{len(short)} very short chapter(s) (<{SHORT_CHAPTER_WORDS} words): {listed}"
        )
    if preamble_words:
        report.anomalies.append(f"Skipped {preamble_words} words before the first chapter heading.")
    return report


def blocks_from_text(text: str) -> list[Block]:
    blocks: list[Block] = []
    prev_blank = True
    for line in text.splitlines():
        if not line.strip():
            prev_blank = True
            blocks.append(Block("", strong=False))
            continue
        blocks.append(Block(line.rstrip(), strong=prev_blank))
        prev_blank = False
    return blocks


def parse_text(text: str, noise: NoiseFilter | None = None) -> DetectionReport:
    text = text.replace("\ufffd", "")
    report = detect_chapters(blocks_from_text(text), joiner="\n", noise=noise)
    if not report.chapters and text.strip():
        body, removed = clean_chapter_text(text, noise)
        report.chapters = [Chapter(1, "Chapter 1", body)]
        report.removed_noise = removed
        report.anomalies.append(
            "No chapter headings found; treating the whole file as one chapter."
        )
    return report


def parse_text_file(path: Path, noise: NoiseFilter | None = None) -> DetectionReport:
    return parse_text(path.read_text(encoding="utf-8", errors="replace"), noise)
