"""Pre-run token and cost estimates for --dry-run and import confirmation."""

from __future__ import annotations

from dataclasses import dataclass

from nextchap.config import Settings
from nextchap.ingest.master_book import Chapter
from nextchap.models import ModelRegistry
from nextchap.retrieval.context import estimate_tokens

TOKENS_PER_WORD = 1.35


@dataclass
class EstimateRow:
    role: str
    model: str
    calls: float
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    cost: float
    step: str = ""


def _row(
    settings: Settings,
    registry: ModelRegistry,
    role: str,
    calls: float,
    prompt: int,
    completion: int,
    cached: int = 0,
    step: str = "",
) -> EstimateRow:
    model = settings.models.roles()[role].model
    cost = registry.estimate_cost(model, int(prompt), int(completion), int(cached))
    return EstimateRow(role, model, calls, int(prompt), int(completion), int(cached), cost, step)


def estimate_import(
    settings: Settings, registry: ModelRegistry, chapters: list[Chapter]
) -> list[EstimateRow]:
    n = len(chapters)
    arcs = -(-n // settings.generation.arc_size)
    summary_in = sum(min(estimate_tokens(c.content), 12_000) + 250 for c in chapters)
    return [
        _row(
            settings,
            registry,
            "analyst",
            n,
            summary_in,
            n * 350,
            step="chapter summaries",
        ),
        _row(
            settings,
            registry,
            "analyst",
            arcs,
            arcs * (settings.generation.arc_size * 350 + 250),
            arcs * 900,
            step="arc summaries",
        ),
        _row(
            settings,
            registry,
            "analyst",
            1,
            min(n, 60) * 350 + 600,
            2500,
            step="master bible",
        ),
        _row(settings, registry, "analyst", 1, 8000, 2500, step="world state"),
    ]


def estimate_generation(
    settings: Settings,
    registry: ModelRegistry,
    count: int,
    base_prefix_tokens: int,
    writer_prefix_tokens: int,
) -> list[EstimateRow]:
    g = settings.generation
    words = (g.target_words[0] + g.target_words[1]) / 2
    prose_tokens = int(words * TOKENS_PER_WORD)
    scenes = 3
    caches = settings.models.writer.model.startswith("anthropic/")
    writer_in = scenes * (writer_prefix_tokens + 1500 + 1500 + prose_tokens // 2)
    writer_cached = (scenes - 1) * writer_prefix_tokens if caches else 0
    rows = [
        _row(
            settings,
            registry,
            "planner",
            count,
            count * (base_prefix_tokens + 3500),
            count * 2500,
            step="chapter plans",
        ),
        _row(
            settings,
            registry,
            "writer",
            count * scenes,
            count * writer_in,
            count * prose_tokens,
            count * writer_cached,
            step="chapter scenes",
        ),
        _row(
            settings,
            registry,
            "analyst",
            count * 2,
            count * (2 * prose_tokens + 2500),
            count * 1150,
            step="summary + world-state delta",
        ),
    ]
    if g.critic:
        rows.append(
            _row(
                settings,
                registry,
                "critic",
                count,
                count * (base_prefix_tokens + prose_tokens + 2000),
                count * 600,
                step="critic pass",
            )
        )
    bible_calls = count / max(1, g.bible_refresh_every)
    rows.append(
        _row(
            settings,
            registry,
            "analyst",
            bible_calls,
            int(bible_calls * 8000),
            int(bible_calls * 2500),
            step="bible refresh",
        )
    )
    return rows


def total(rows: list[EstimateRow]) -> float:
    return sum(r.cost for r in rows)
