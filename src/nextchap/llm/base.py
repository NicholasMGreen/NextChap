"""The interface the pipeline depends on. Tests substitute a fake implementation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar

from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)

Role = str  # "writer" | "planner" | "analyst" | "critic"
DeltaCallback = Callable[[str], None]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float | None = None


@dataclass
class Completion:
    text: str
    model: str
    finish_reason: str
    usage: Usage
    continued: int = 0


class LLM(Protocol):
    def text(
        self,
        role: Role,
        system: str,
        user: str,
        *,
        prefix: str = "",
        label: str = "",
        on_delta: DeltaCallback | None = None,
        max_tokens: int | None = None,
    ) -> Completion: ...

    def structured(
        self,
        role: Role,
        system: str,
        user: str,
        schema: type[T],
        *,
        prefix: str = "",
        label: str = "",
    ) -> T: ...


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...
