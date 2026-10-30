"""Token-budgeted prompt assembly (model-agnostic ~4 chars/token estimate)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN + 1


def fit_text(text: str, max_tokens: int, keep: Literal["head", "tail"] = "head") -> str:
    limit = max(0, max_tokens) * CHARS_PER_TOKEN
    if len(text) <= limit:
        return text
    if keep == "head":
        cut = text[:limit]
        cut = cut[: cut.rfind("\n")] if "\n" in cut[limit // 2 :] else cut
        return cut.rstrip() + "\n[…]"
    cut = text[-limit:]
    nl = cut.find("\n")
    cut = cut[nl + 1 :] if 0 <= nl < limit // 2 else cut
    return "[…]\n" + cut.lstrip()


def fit_items(
    items: list[str], max_tokens: int, keep: Literal["head", "tail"] = "tail"
) -> list[str]:
    """Keep whole items (newest when keep='tail') within the budget."""
    out: list[str] = []
    used = 0
    seq = reversed(items) if keep == "tail" else iter(items)
    for item in seq:
        cost = estimate_tokens(item)
        if used + cost > max_tokens:
            break
        out.append(item)
        used += cost
    return list(reversed(out)) if keep == "tail" else out


@dataclass(frozen=True)
class ContextBudget:
    bible: int
    style: int
    world: int
    recent_prose: int
    recent_summaries: int
    rag: int
    plan: int
    so_far: int

    @classmethod
    def from_total(cls, total: int) -> ContextBudget:
        return cls(
            bible=int(total * 0.08),
            style=int(total * 0.02),
            world=int(total * 0.06),
            recent_prose=int(total * 0.25),
            recent_summaries=int(total * 0.05),
            rag=int(total * 0.06),
            plan=int(total * 0.05),
            so_far=int(total * 0.10),
        )
