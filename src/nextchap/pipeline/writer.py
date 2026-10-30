"""Scene-by-scene chapter writing with streaming previews."""

from __future__ import annotations

import re
from collections.abc import Callable

from nextchap.llm.base import LLM
from nextchap.pipeline.plan import PlanDoc
from nextchap.prompts import render
from nextchap.retrieval.context import fit_text

_LEADING_LABEL_RE = re.compile(
    r"^\s*(?:\*\*|#+\s*)?(?:scene\s+\d+|chapter\s+\d+)\b[^\n]*\n+", re.IGNORECASE
)
_PREAMBLE_RE = re.compile(
    r"^\s*(?:here(?:'s| is) (?:the|scene|chapter)[^\n]*|sure[,!][^\n]*)\n+", re.IGNORECASE
)
_TRAILER_RE = re.compile(
    r"\n+\s*(?:-{3,}|\*{3,}|\(?end of (?:scene|chapter)[^\n]*)\s*$", re.IGNORECASE
)


def clean_prose(text: str) -> str:
    s = text.strip()
    for _ in range(2):
        s = _PREAMBLE_RE.sub("", s)
        s = _LEADING_LABEL_RE.sub("", s)
    s = _TRAILER_RE.sub("", s)
    lines = [ln for ln in s.splitlines() if not re.match(r"^\s*#{1,6}\s", ln)]
    return "\n".join(lines).strip()


def word_count(text: str) -> int:
    return len(text.split())


def scene_words(plan: PlanDoc, index: int, target_total: int) -> int:
    planned = plan.scene_words[index] if index < len(plan.scene_words) else None
    return planned or max(200, target_total // max(1, len(plan.scenes)))


def join_scenes(scenes: list[str], separator: str) -> str:
    sep = f"\n\n{separator}\n\n" if separator else "\n\n"
    return sep.join(s.strip() for s in scenes if s.strip())


def write_scene(
    llm: LLM,
    *,
    chapter: int,
    plan: PlanDoc,
    index: int,
    system: str,
    prefix: str,
    rag: str,
    so_far: str,
    so_far_tokens: int,
    target_total: int,
    last_hook: str,
    on_preview: Callable[[str], None] | None = None,
) -> str:
    n = len(plan.scenes)
    if index == 0:
        position = f"Open in-scene, continuing directly from the last chapter's hook: {last_hook or '(see plan)'}"
    elif index == n - 1:
        position = f"This is the final scene: end the chapter on this hook: {plan.ending_hook or '(see plan)'}"
    else:
        position = "Continue the chapter; do not wrap it up yet."
    user = render(
        "writer_scene",
        chapter=chapter,
        title=plan.title,
        plan=plan.markdown,
        scene_number=index + 1,
        scene_count=n,
        scene_words=scene_words(plan, index, target_total),
        position=position,
        scene=plan.scenes[index],
        rag=rag or "(none)",
        so_far=fit_text(so_far, so_far_tokens, keep="tail")
        if so_far
        else "(nothing yet: this is the opening)",
    )
    comp = llm.text(
        "writer",
        system,
        user,
        prefix=prefix,
        label=f"write:ch{chapter}:s{index + 1}",
        on_delta=on_preview,
    )
    return clean_prose(comp.text)


def extend_chapter(
    llm: LLM,
    *,
    chapter: int,
    plan: PlanDoc,
    system: str,
    prefix: str,
    prose: str,
    min_words: int,
    on_preview: Callable[[str], None] | None = None,
) -> str:
    user = render(
        "writer_extend",
        chapter=chapter,
        title=plan.title,
        words=word_count(prose),
        min_words=min_words,
        ending_hook=plan.ending_hook or "(see plan)",
        tail=fit_text(prose, 1500, keep="tail"),
    )
    comp = llm.text(
        "writer", system, user, prefix=prefix, label=f"extend:ch{chapter}", on_delta=on_preview
    )
    return clean_prose(comp.text)


def rewrite_scene(
    llm: LLM,
    *,
    chapter: int,
    plan: PlanDoc,
    index: int,
    scenes: list[str],
    issues: list[str],
    system: str,
    prefix: str,
    target_total: int,
) -> str:
    user = render(
        "rewrite_scene",
        chapter=chapter,
        title=plan.title,
        scene_number=index + 1,
        issues="\n".join(f"- {i}" for i in issues),
        scene=plan.scenes[min(index, len(plan.scenes) - 1)],
        scene_words=scene_words(plan, index, target_total),
        before=fit_text(scenes[index - 1], 800, keep="tail") if index > 0 else "(chapter start)",
        current=scenes[index],
        after=fit_text(scenes[index + 1], 800, keep="head")
        if index + 1 < len(scenes)
        else "(chapter end)",
    )
    comp = llm.text(
        "writer", system, user, prefix=prefix, label=f"rewrite:ch{chapter}:s{index + 1}"
    )
    return clean_prose(comp.text)
