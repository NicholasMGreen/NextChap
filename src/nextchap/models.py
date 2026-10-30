"""Model registry backed by OpenRouter's /models endpoint (cached for a day)."""

from __future__ import annotations

import contextlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from nextchap.config import DEEPSEEK_FLASH, HAIKU, Settings
from nextchap.storage.atomic import atomic_write_json

CACHE_TTL_SECONDS = 24 * 3600


@dataclass
class ModelInfo:
    id: str
    name: str = ""
    context_length: int = 0
    prompt_price: float = 0.0  # USD per token
    completion_price: float = 0.0
    cache_read_price: float | None = None
    supported_parameters: set[str] = field(default_factory=set)

    @classmethod
    def from_api(cls, d: dict[str, Any]) -> ModelInfo:
        pricing = d.get("pricing") or {}

        def price(key: str) -> float | None:
            try:
                return float(pricing[key])
            except (KeyError, TypeError, ValueError):
                return None

        return cls(
            id=str(d["id"]),
            name=str(d.get("name", "")),
            context_length=int(d.get("context_length") or 0),
            prompt_price=price("prompt") or 0.0,
            completion_price=price("completion") or 0.0,
            cache_read_price=price("input_cache_read"),
            supported_parameters=set(d.get("supported_parameters") or []),
        )


# Used when the registry can't be fetched (offline, dry runs). Verified 2026-10-29.
KNOWN_MODELS: dict[str, ModelInfo] = {
    HAIKU: ModelInfo(HAIKU, "Claude Haiku 5.5", 1_000_000, 0.10e-6, 0.50e-6, 0.01e-6),
    DEEPSEEK_FLASH: ModelInfo(
        DEEPSEEK_FLASH, "DeepSeek V4.1 Flash 1017", 1_048_576, 0.15e-6, 0.60e-6, 0.003e-6
    ),
}


class ModelRegistry:
    def __init__(self, models: dict[str, ModelInfo], fetched_at: float | None = None) -> None:
        self.models = models
        self.fetched_at = fetched_at

    @property
    def live(self) -> bool:
        return self.fetched_at is not None

    def get(self, model_id: str) -> ModelInfo | None:
        return self.models.get(model_id) or KNOWN_MODELS.get(model_id)

    def supports(self, model_id: str, param: str) -> bool:
        info = self.models.get(model_id)
        if info is None or not info.supported_parameters:
            return True
        return param in info.supported_parameters

    def estimate_cost(
        self, model_id: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0
    ) -> float:
        info = self.get(model_id)
        if info is None:
            return 0.0
        uncached = max(0, prompt_tokens - cached_tokens)
        cache_price = (
            info.cache_read_price if info.cache_read_price is not None else info.prompt_price
        )
        return (
            uncached * info.prompt_price
            + cached_tokens * cache_price
            + completion_tokens * info.completion_price
        )

    @classmethod
    def offline(cls) -> ModelRegistry:
        return cls({})

    @classmethod
    def load(
        cls,
        settings: Settings,
        *,
        refresh: bool = False,
        transport: httpx.BaseTransport | None = None,
    ) -> ModelRegistry:
        cache = settings.cache_dir / "models.json"
        if not refresh and cache.exists():
            try:
                raw = json.loads(cache.read_text(encoding="utf-8"))
                if time.time() - float(raw["fetched_at"]) < CACHE_TTL_SECONDS:
                    return cls._from_payload(raw["data"], float(raw["fetched_at"]))
            except (OSError, ValueError, KeyError, TypeError):
                pass
        try:
            with httpx.Client(
                transport=transport, timeout=settings.network.connect_timeout * 2
            ) as c:
                resp = c.get(f"{settings.network.base_url}/models")
                resp.raise_for_status()
                data = resp.json()["data"]
        except (httpx.HTTPError, ValueError, KeyError):
            return cls.offline()
        now = time.time()
        _save_cache(cache, data, now)
        return cls._from_payload(data, now)

    @classmethod
    def _from_payload(cls, data: list[dict[str, Any]], fetched_at: float) -> ModelRegistry:
        models = {}
        for d in data:
            try:
                info = ModelInfo.from_api(d)
            except (KeyError, TypeError, ValueError):
                continue
            models[info.id] = info
        return cls(models, fetched_at)


def _save_cache(path: Path, data: list[dict[str, Any]], fetched_at: float) -> None:
    with contextlib.suppress(OSError):
        atomic_write_json(path, {"fetched_at": fetched_at, "data": data})
