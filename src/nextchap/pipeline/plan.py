"""Chapter plans: the structured planner output and its editable markdown form.

The writer always works from the markdown form, so a plan edited by hand in review mode
is used exactly as edited.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import BaseModel, Field


class Scene(BaseModel):
    location: str
    characters: list[str]
    purpose: str = Field(description="What changes because of this scene")
    beats: list[str] = Field(
        description="4-6 concrete beats: dialogue, decisions, reveals, actions"
    )
    target_words: int


class ChapterPlan(BaseModel):
    title: str
    opening_hook: str
    scenes: list[Scene]
    threads_advanced: list[str]
    character_moment: str
    revelations: list[str]
    ending_hook: str
    tone: str
    continuity_traps: list[str]
    must_dramatize: list[str]


def _bullets(items: list[str]) -> str:
    return "\n".join(f"- {i}" for i in items) if items else "- (none)"


def render_plan(plan: ChapterPlan) -> str:
    parts = [f"# {plan.title.strip() or 'Untitled'}", "", "## Opening hook", plan.opening_hook, ""]
    for i, s in enumerate(plan.scenes, 1):
        parts += [
            f"## Scene {i}: {s.location}",
            f"Characters: {', '.join(s.characters)}",
            f"Purpose: {s.purpose}",
            f"Target words: {s.target_words}",
            "Beats:",
            _bullets(s.beats),
            "",
        ]
    parts += [
        "## Threads advanced",
        _bullets(plan.threads_advanced),
        "",
        "## Character moment",
        plan.character_moment,
        "",
        "## Revelations",
        _bullets(plan.revelations),
        "",
        "## Ending hook",
        plan.ending_hook,
        "",
        "## Tone",
        plan.tone,
        "",
        "## Continuity traps",
        _bullets(plan.continuity_traps),
        "",
        "## Must dramatize",
        _bullets(plan.must_dramatize),
    ]
    return "\n".join(parts).strip() + "\n"


@dataclass
class PlanDoc:
    markdown: str
    title: str
    opening_hook: str = ""
    ending_hook: str = ""
    scenes: list[str] = field(default_factory=list)
    scene_words: list[int | None] = field(default_factory=list)


_SECTION_RE = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_TARGET_RE = re.compile(r"^\s*target words:\s*(\d+)", re.IGNORECASE | re.MULTILINE)


def parse_plan(markdown: str) -> PlanDoc:
    title_m = re.search(r"^#\s+(.+?)\s*$", markdown, re.MULTILINE)
    doc = PlanDoc(
        markdown=markdown.strip() + "\n", title=title_m.group(1).strip() if title_m else ""
    )
    heads = list(_SECTION_RE.finditer(markdown))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(markdown)
        name = m.group(1).strip()
        body = markdown[m.end() : end].strip()
        low = name.lower()
        if low.startswith("scene"):
            doc.scenes.append(f"{name}\n{body}".strip())
            t = _TARGET_RE.search(body)
            doc.scene_words.append(int(t.group(1)) if t else None)
        elif low == "opening hook":
            doc.opening_hook = body
        elif low == "ending hook":
            doc.ending_hook = body
    if not doc.scenes:
        doc.scenes = [markdown.strip()]
        doc.scene_words = [None]
    return doc
