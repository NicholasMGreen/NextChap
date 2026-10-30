"""How the engine talks to the UI. The CLI implements this with Rich; tests use NullReporter."""

from __future__ import annotations

from typing import Literal, Protocol

ReviewAction = Literal["accept", "regenerate", "quit"]


class Reporter(Protocol):
    def phase(self, name: str, total: int | None = None) -> None: ...
    def advance(self, n: int = 1, detail: str = "") -> None: ...
    def end_phase(self) -> None: ...
    def step(self, label: str) -> None: ...
    def preview(self, text: str) -> None: ...
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def chapter_done(self, number: int, title: str, words: int, cost: float) -> None: ...


class ReviewHooks(Protocol):
    def review_plan(self, chapter: int, plan_markdown: str) -> tuple[ReviewAction, str]:
        """Return (action, text): edited plan for accept, feedback note for regenerate."""
        ...

    def review_chapter(self, chapter: int, prose: str) -> tuple[ReviewAction, str]:
        """Return (action, text): possibly edited prose for accept, feedback for regenerate."""
        ...


class NullReporter:
    def phase(self, name: str, total: int | None = None) -> None:
        pass

    def advance(self, n: int = 1, detail: str = "") -> None:
        pass

    def end_phase(self) -> None:
        pass

    def step(self, label: str) -> None:
        pass

    def preview(self, text: str) -> None:
        pass

    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    def chapter_done(self, number: int, title: str, words: int, cost: float) -> None:
        pass
