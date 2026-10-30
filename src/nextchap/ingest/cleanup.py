"""Remove author promo lines, separator noise, and repeated chapter headers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

DEFAULT_NOISE_PATTERNS: tuple[str, ...] = (
    r"patre\.?on",
    r"patreon\.com",
    r"ko-?fi\.com",
    r"\bchapters?\s+in\s+advance\b",
    r"^\s*\[?\d+\s+chapters?\s+(?:in\s+)?advance\]?\s*$",
    r"^\s*(?:support|join)\s+(?:me|us)\s+on\b",
)
DOTS_RE = re.compile(r"^\s*\.{3,}\s*$")
# Promo lines are short; long lines matching a keyword are usually prose.
MAX_NOISE_LINE_CHARS = 200

MD_HEADER_RE = re.compile(r"^##\s+Chapter\s+(\d+)[:\s\-\u2013\u2014]*(.*)$", re.IGNORECASE)


@dataclass
class CleanupReport:
    removed_noise: int = 0
    removed_duplicate_headers: int = 0

    @property
    def changed(self) -> bool:
        return bool(self.removed_noise or self.removed_duplicate_headers)


class NoiseFilter:
    def __init__(self, extra_patterns: Iterable[str] = ()) -> None:
        patterns = [*DEFAULT_NOISE_PATTERNS, *extra_patterns]
        self._res = [re.compile(p, re.IGNORECASE) for p in patterns]

    def is_noise(self, line: str) -> bool:
        s = line.strip()
        if not s:
            return False
        if DOTS_RE.match(s):
            return True
        if len(s) > MAX_NOISE_LINE_CHARS:
            return False
        return any(r.search(s) for r in self._res)


def header_key(line: str) -> str | None:
    m = MD_HEADER_RE.match(line.strip())
    if not m:
        return None
    title = re.sub(r"\s+", " ", (m.group(2) or "").strip().lower())
    return f"{int(m.group(1))}:{title}"


def clean_master_markdown(text: str, noise: NoiseFilter | None = None) -> tuple[str, CleanupReport]:
    noise = noise or NoiseFilter()
    report = CleanupReport()
    out: list[str] = []
    last_key: str | None = None
    saw_text_since_header = False

    for line in text.splitlines():
        if noise.is_noise(line):
            report.removed_noise += 1
            continue
        key = header_key(line)
        if key is not None:
            if key == last_key and not saw_text_since_header:
                report.removed_duplicate_headers += 1
                continue
            out.append(line)
            last_key = key
            saw_text_since_header = False
            continue
        out.append(line)
        if line.strip():
            saw_text_since_header = True

    cleaned = "\n".join(out)
    if text.endswith("\n"):
        cleaned += "\n"
    return cleaned, report


def clean_chapter_text(text: str, noise: NoiseFilter | None = None) -> tuple[str, int]:
    noise = noise or NoiseFilter()
    kept: list[str] = []
    removed = 0
    for line in text.splitlines():
        if noise.is_noise(line):
            removed += 1
            continue
        kept.append(line)
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    return cleaned, removed
