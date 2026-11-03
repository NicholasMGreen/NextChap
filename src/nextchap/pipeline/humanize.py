"""Deterministic checks for machine-sounding prose: stock phrasing and flat rhythm.

The rules themselves live in `prompts/humanize.md` (sent to the writer, and echoed to the
critic) with the long rationale in `docs/humanize.md`. This module measures only what a model
does not have to: banned stock phrases and sentence-length spread. The critic gets evidence
instead of a hunch, and a chapter can be re-run because a count fired.
"""

from __future__ import annotations

import re

# Stock phrases that read as machine-written in fiction. Mirrors the list in
# prompts/humanize.md and docs/humanize.md; keep the three in sync.
TELLS: tuple[str, ...] = (
    "a testament to",
    "little did",
    "unbeknownst",
    "only time would tell",
    "the air was thick with",
    "sent shivers",
    "shivers down",
    "in the grand scheme",
    "at the end of the day",
    "turns out",
    "it is important to note",
    "furthermore",
    "moreover",
    "palpable",
    "multifaceted",
    "vibrant",
    "tapestry",
    "delve",
)

MIN_SENTENCES = 8
MIN_SPREAD = 10
MID_BAND = (10, 20)
MAX_MID_SHARE = 0.6

_PIVOT = re.compile(r"\bnot\s+(?:just|only)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*")
# A scene separator line ("* * *", "---") is not a sentence boundary worth trusting.
_SEPARATOR = re.compile(r"^\s*(?:[*_~#-]{2,}\s*)+$", re.MULTILINE)


def sentence_lengths(prose: str) -> list[int]:
    """Word count of every sentence, in order."""
    lengths: list[int] = []
    for sentence in _SENTENCE.findall(_SEPARATOR.sub(" ", prose)):
        n = sum(1 for word in sentence.split() if word.strip("*-\u2013\u2014#_~"))
        if n:
            lengths.append(n)
    return lengths


def flat_rhythm(prose: str) -> str | None:
    """A finding when sentence lengths cluster, else None. Short text is not judged."""
    lengths = sentence_lengths(prose)
    if len(lengths) < MIN_SENTENCES:
        return None
    spread = max(lengths) - min(lengths)
    lo, hi = MID_BAND
    mid = sum(1 for n in lengths if lo <= n <= hi)
    if spread < MIN_SPREAD:
        return (
            f"Flat sentence rhythm: {len(lengths)} sentences with only {spread} words between "
            "the longest and the shortest."
        )
    if mid / len(lengths) >= MAX_MID_SHARE:
        return (
            f"Flat sentence rhythm: {mid} of {len(lengths)} sentences are between {lo} and "
            f"{hi} words long."
        )
    return None


def tell_hits(prose: str) -> list[str]:
    """Stock phrases present in the prose, in TELLS order."""
    low = prose.lower()
    hits = [t for t in TELLS if t in low]
    if _PIVOT.search(prose):
        hits.append("not just / not only")
    return hits


def humanize_findings(prose: str) -> list[str]:
    """Machine-sounding prose as check lines the critic can act on."""
    findings: list[str] = []
    hits = tell_hits(prose)
    if hits:
        findings.append("Machine-sounding phrasing: " + ", ".join(f'"{h}"' for h in hits) + ".")
    rhythm = flat_rhythm(prose)
    if rhythm:
        findings.append(rhythm)
    return findings
