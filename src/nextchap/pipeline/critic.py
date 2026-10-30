"""Continuity review: cheap local checks plus an independent model pass."""

from __future__ import annotations

import re
from collections import Counter
from typing import Literal

from pydantic import BaseModel

from nextchap.llm.base import LLM
from nextchap.prompts import render
from nextchap.retrieval.context import fit_text

NGRAM = 6
REPEAT_THRESHOLD = 3


class CriticIssue(BaseModel):
    severity: Literal["minor", "major"]
    scene_number: int | None
    description: str
    suggestion: str


class CriticReport(BaseModel):
    verdict: Literal["ok", "minor", "major"]
    plan_followed: bool
    issues: list[CriticIssue]


def banned_phrases(style_guide: str) -> list[str]:
    """Quoted phrases on style-guide lines that mention avoiding/banning."""
    out: list[str] = []
    for line in style_guide.splitlines():
        if re.search(r"\b(avoid|ban|never use|don't use|do not use)\b", line, re.IGNORECASE):
            out += [p.strip() for p in re.findall(r'"([^"]{3,80})"', line)]
    return out


def local_checks(
    prose: str, *, min_words: int, target: tuple[int, int], style_guide: str
) -> list[str]:
    findings: list[str] = []
    words = prose.split()
    n = len(words)
    if n < min_words:
        findings.append(f"Length {n} words is below the minimum of {min_words}.")
    elif n > int(target[1] * 1.6):
        findings.append(f"Length {n} words is far above the target of {target[0]}-{target[1]}.")
    low = [re.sub(r"\W+", "", w.lower()) for w in words]
    grams = Counter(tuple(low[i : i + NGRAM]) for i in range(max(0, n - NGRAM + 1)))
    repeats = [" ".join(g) for g, c in grams.most_common(5) if c >= REPEAT_THRESHOLD and all(g)]
    if repeats:
        findings.append("Repeated phrasing: " + "; ".join(f'"{r}"' for r in repeats[:3]))
    text_low = prose.lower()
    hits = [p for p in banned_phrases(style_guide) if p.lower() in text_low]
    if hits:
        findings.append("Uses phrases the style guide bans: " + ", ".join(f'"{h}"' for h in hits))
    return findings


def critique(
    llm: LLM,
    *,
    chapter: int,
    plan_markdown: str,
    prose: str,
    recent: str,
    checks: list[str],
    prefix: str,
) -> CriticReport:
    return llm.structured(
        "critic",
        render("critic_system"),
        render(
            "critic",
            chapter=chapter,
            plan=plan_markdown,
            recent=recent or "(none)",
            checks="\n".join(f"- {c}" for c in checks) or "- none",
            prose=fit_text(prose, 20_000, keep="head"),
        ),
        CriticReport,
        prefix=prefix,
        label=f"critic:ch{chapter}",
    )
