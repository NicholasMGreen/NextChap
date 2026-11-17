"""Per-run spend tracking and the --max-cost guard."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from nextchap.errors import BudgetExceeded
from nextchap.storage.atomic import append_line


@dataclass
class CallRecord:
    role: str
    model: str
    label: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0
    ts: float = field(default_factory=time.time)


@dataclass
class CallTotals:
    """Aggregated usage for one role/model bucket."""

    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cost: float = 0.0

    def add(self, rec: CallRecord) -> None:
        self.calls += 1
        self.prompt_tokens += rec.prompt_tokens
        self.completion_tokens += rec.completion_tokens
        self.cached_tokens += rec.cached_tokens
        self.cost += rec.cost


class CostMeter:
    def __init__(self, max_cost: float | None = None) -> None:
        self.max_cost = max_cost
        self.records: list[CallRecord] = []
        self.log_path: Path | None = None
        self.book_id: str | None = None
        self._listeners: list[Callable[[CostMeter], None]] = []

    def bind(self, log_path: Path, book_id: str) -> None:
        self.log_path = log_path
        self.book_id = book_id

    def subscribe(self, fn: Callable[[CostMeter], None]) -> None:
        self._listeners.append(fn)

    @property
    def total_cost(self) -> float:
        return sum(r.cost for r in self.records)

    @property
    def prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)

    @property
    def completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)

    @property
    def cached_tokens(self) -> int:
        return sum(r.cached_tokens for r in self.records)

    @property
    def cache_hit_rate(self) -> float:
        return self.cached_tokens / self.prompt_tokens if self.prompt_tokens else 0.0

    def by_role(self) -> dict[str, float]:
        out: dict[str, float] = defaultdict(float)
        for r in self.records:
            out[r.role] += r.cost
        return dict(out)

    def by_model(self) -> dict[str, CallTotals]:
        out: dict[str, CallTotals] = {}
        for r in self.records:
            out.setdefault(r.model, CallTotals()).add(r)
        return out

    def by_role_model(self) -> dict[tuple[str, str], CallTotals]:
        """Usage keyed by (role, model): what actually did the work in this run."""
        out: dict[tuple[str, str], CallTotals] = {}
        for r in self.records:
            out.setdefault((r.role, r.model), CallTotals()).add(r)
        return out

    def remaining(self) -> float | None:
        return None if self.max_cost is None else self.max_cost - self.total_cost

    def check_budget(self) -> None:
        if self.max_cost is not None and self.total_cost >= self.max_cost:
            raise BudgetExceeded(
                f"Spend ${self.total_cost:.4f} reached the --max-cost limit of ${self.max_cost:.2f}."
            )

    def record(self, rec: CallRecord) -> None:
        self.records.append(rec)
        if self.log_path is not None:
            payload = {"book": self.book_id, **asdict(rec)}
            append_line(self.log_path, json.dumps(payload, ensure_ascii=False))
        for fn in self._listeners:
            fn(self)


def read_usage_log(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def logged_spend(path: Path) -> float:
    total = 0.0
    for row in read_usage_log(path):
        try:
            total += float(row.get("cost") or 0)
        except (TypeError, ValueError):
            continue
    return total
