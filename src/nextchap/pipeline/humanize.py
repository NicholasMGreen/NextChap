"""Deterministic checks for machine-sounding prose.

The rules themselves live in `prompts/humanize.md` (sent to the writer, and echoed to the
critic) with the long rationale in `docs/humanize.md`. This module measures only what a model
does not have to: banned stock phrases, sentence-length spread, and a repeated sentence opener
(the crutch that replaces the one you banned). The critic gets evidence instead of a hunch.
"""

from __future__ import annotations

import re
from collections import Counter

# Stock phrases that read as machine-written in fiction. prompts/humanize.md names the
# worst of them in the system message; this list is the broader set the checks enforce.
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
    "meticulous",
    "intricate",
    "a symphony of",
    "labyrinth of",
    "undoubtedly",
    "embark",
    "bolster",
    "garner",
)

MIN_SENTENCES = 8
MIN_SPREAD = 10
MID_BAND = (10, 20)
MAX_MID_SHARE = 0.6

MIN_OPENER_SENTENCES = 15
MAX_OPENER_SHARE = 0.3
# Function words carry no stylistic signature: a run of these is normal prose. "And" and
# "But" are also what the humanize rules ask for, so they must never be a finding.
OPENER_ALLOWED = frozenset(
    {
        "and",
        "but",
        "so",
        "then",
        "the",
        "a",
        "an",
        "he",
        "she",
        "it",
        "they",
        "i",
        "you",
        "we",
        "his",
        "her",
        "their",
        "its",
        "my",
        "your",
        "there",
        "this",
        "that",
    }
)

_PIVOT = re.compile(r"\bnot\s+(?:just|only)\b", re.IGNORECASE)
_SENTENCE = re.compile(r"[^.!?…]+[.!?…]*")
_FIRST_WORD = re.compile(r"[A-Za-z']+")
# A scene separator line ("* * *", "---") is not a sentence boundary worth trusting.
_SEPARATOR = re.compile(r"^\s*[*_~#-](?:[\s*_~#-]*[*_~#-])+\s*$", re.MULTILINE)


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


def sentence_openers(prose: str) -> list[str]:
    """First word of every sentence, lowercased."""
    openers: list[str] = []
    for sentence in _SENTENCE.findall(_SEPARATOR.sub(" ", prose)):
        match = _FIRST_WORD.search(sentence)
        if match:
            openers.append(match.group(0).lower())
    return openers


def dominant_opener(prose: str) -> str | None:
    """A finding when one word keeps starting sentences: the swapped-in crutch."""
    openers = [w for w in sentence_openers(prose) if w not in OPENER_ALLOWED]
    if len(openers) < MIN_OPENER_SENTENCES:
        return None
    word, count = Counter(openers).most_common(1)[0]
    if count / len(openers) < MAX_OPENER_SHARE:
        return None
    return f'Repeated sentence opener: {count} of {len(openers)} sentences start with "{word}".'


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
    opener = dominant_opener(prose)
    if opener:
        findings.append(opener)
    return findings
