"""Plan one chapter as structured JSON."""

from __future__ import annotations

from nextchap.llm.base import LLM
from nextchap.pipeline.plan import ChapterPlan
from nextchap.prompts import render


def plan_chapter(
    llm: LLM,
    *,
    chapter: int,
    prefix: str,
    arc: str,
    recent: str,
    rag: str,
    target_words: int,
    feedback: str = "",
) -> ChapterPlan:
    note = (
        f"\nAuthor feedback on the previous plan (apply it):\n{feedback}\n"
        if feedback.strip()
        else ""
    )
    return llm.structured(
        "planner",
        render("planner_system"),
        render(
            "planner",
            chapter=chapter,
            target_words=target_words,
            arc=arc or "(none yet)",
            recent=recent or "(none)",
            rag=rag or "(none)",
            feedback=note,
        ),
        ChapterPlan,
        prefix=prefix,
        label=f"plan:ch{chapter}",
    )
