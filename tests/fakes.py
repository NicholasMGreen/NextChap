"""Scripted stand-ins for the model client (no network in tests)."""

from __future__ import annotations

import itertools
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from nextchap.llm.base import Completion, Usage
from nextchap.llm.cost import CallRecord, CostMeter
from nextchap.memory.world_state import (
    CharacterDraft,
    CharacterPatch,
    NamedEntry,
    WorldStateDelta,
    WorldStateExtraction,
)
from nextchap.pipeline.critic import CriticIssue, CriticReport
from nextchap.pipeline.plan import ChapterPlan, Scene

WORDS = [
    "the",
    "wind",
    "moved",
    "across",
    "the",
    "courtyard",
    "while",
    "Lin",
    "Feng",
    "weighed",
    "every",
    "word",
]


def prose(n: int, tag: str) -> str:
    cycle = itertools.cycle(WORDS)
    words = [next(cycle) for _ in range(n)]
    paras = [" ".join(words[i : i + 60]) + "." for i in range(0, n, 60)]
    return f"[{tag}] " + "\n\n".join(paras)


class FakeLLM:
    def __init__(
        self,
        meter: CostMeter | None = None,
        *,
        scene_words: int = 450,
        critic_verdict: str = "ok",
        cost_per_call: float = 0.001,
    ) -> None:
        self.meter = meter or CostMeter()
        self.calls: list[tuple[str, str]] = []
        self.scene_words = scene_words
        self.critic_verdict = critic_verdict
        self.cost_per_call = cost_per_call
        self.before_call: Callable[[str, str], None] | None = None

    def _record(self, role: str, label: str) -> None:
        if self.before_call:
            self.before_call(role, label)
        self.meter.check_budget()
        self.calls.append((role, label))
        self.meter.record(
            CallRecord(
                role=role,
                model="fake",
                label=label,
                prompt_tokens=100,
                completion_tokens=50,
                cost=self.cost_per_call,
            )
        )

    def labels(self, prefix: str) -> list[str]:
        return [lbl for _, lbl in self.calls if lbl.startswith(prefix)]

    def text(
        self,
        role: str,
        system: str,
        user: str,
        *,
        prefix: str = "",
        label: str = "",
        on_delta: Any = None,
        max_tokens: int | None = None,
    ) -> Completion:
        self._record(role, label)
        if label.startswith(("write:", "extend:", "rewrite:")):
            out = prose(self.scene_words, label)
            if on_delta:
                on_delta(out[: len(out) // 2])
                on_delta(out)
        elif label.startswith("summary:"):
            out = f"Summary for {label}: Lin Feng trains and faces the Azure Sect."
        elif label.startswith("arc:"):
            out = f"Arc summary {label}."
        elif label == "bible":
            out = "# Story bible\n\nLin Feng is a cultivator."
        else:
            out = f"text for {label}"
        return Completion(out, "fake", "stop", Usage(100, 50, 0, self.cost_per_call))

    def structured(
        self,
        role: str,
        system: str,
        user: str,
        schema: type[Any],
        *,
        prefix: str = "",
        label: str = "",
    ) -> BaseModel:
        self._record(role, label)
        if schema is WorldStateExtraction:
            return WorldStateExtraction(
                characters=[
                    CharacterDraft(
                        name="Lin Feng",
                        role="protagonist",
                        status="alive",
                        location="Azure Peak",
                        power_level="Qi 3",
                        goals="revenge",
                        relationships="",
                        notes="",
                    )
                ],
                active_threads=["Find the sect traitor"],
                resolved_threads=[],
                world_rules=["Qi realms have nine stages"],
                locations=[NamedEntry(name="Azure Peak", description="mountain")],
                factions=[],
                key_items=[],
                current_arc="Outer disciple",
                story_tone="serious",
                narrative_pov="third person limited",
                last_chapter_hook="A letter arrives.",
            )
        if schema is WorldStateDelta:
            n = label.split("ch")[-1]
            return WorldStateDelta(
                upsert_characters=[
                    CharacterPatch(
                        id=None,
                        name=f"Elder {n}",
                        role="mentor",
                        status=None,
                        location=None,
                        power_level=None,
                        goals=None,
                        relationships=None,
                        notes=None,
                    )
                ],
                new_threads=[f"Mystery {n}"],
                resolved_thread_ids=[],
                new_world_rules=[],
                upsert_locations=[],
                upsert_factions=[],
                new_key_items=[],
                current_arc=None,
                story_tone=None,
                narrative_pov=None,
                last_chapter_hook=f"Hook after chapter {n}",
            )
        if schema is ChapterPlan:
            n = label.split("ch")[-1]
            return ChapterPlan(
                title=f"Generated {n}",
                opening_hook="The letter burns.",
                scenes=[
                    Scene(
                        location=f"Hall {i}",
                        characters=["Lin Feng"],
                        purpose="advance",
                        beats=["beat a", "beat b", "beat c", "beat d"],
                        target_words=500,
                    )
                    for i in range(1, 4)
                ],
                threads_advanced=["t001"],
                character_moment="doubt",
                revelations=["traitor"],
                ending_hook=f"Ending hook {n}",
                tone="tense",
                continuity_traps=[],
                must_dramatize=["duel"],
            )
        if schema is CriticReport:
            if self.critic_verdict == "major":
                return CriticReport(
                    verdict="major",
                    plan_followed=False,
                    issues=[
                        CriticIssue(
                            severity="major",
                            scene_number=2,
                            description="Lin Feng uses a lost arm.",
                            suggestion="Use the other arm.",
                        ),
                    ],
                )
            return CriticReport(verdict="ok", plan_followed=True, issues=[])
        raise AssertionError(f"unexpected schema {schema}")
