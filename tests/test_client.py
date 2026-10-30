from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import BaseModel

from nextchap.config import NetworkConfig, Settings
from nextchap.errors import BudgetExceeded, ConfigError, FatalLLMError, LLMError
from nextchap.llm.client import EmptyCompletion, OpenRouterClient
from nextchap.llm.cost import CostMeter
from nextchap.models import ModelInfo, ModelRegistry


class Pick(BaseModel):
    name: str
    score: int


def sse(*chunks: dict[str, Any]) -> bytes:
    lines = [": OPENROUTER PROCESSING", ""]
    for c in chunks:
        lines += [f"data: {json.dumps(c)}", ""]
    lines += ["data: [DONE]", ""]
    return "\n".join(lines).encode()


def delta(
    text: str, finish: str | None = None, usage: dict[str, Any] | None = None
) -> dict[str, Any]:
    chunk: dict[str, Any] = {
        "model": "anthropic/claude-haiku-5.5",
        "choices": [{"delta": {"content": text}, "finish_reason": finish}],
    }
    if usage:
        chunk["usage"] = usage
    return chunk


def ok_json(content: str, finish: str = "stop", cost: float = 0.001) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": "deepseek/deepseek-v4.1-flash-1017",
            "choices": [{"message": {"content": content}, "finish_reason": finish}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "cost": cost},
        },
    )


def make(
    handler: Callable[[httpx.Request], httpx.Response],
    *,
    meter: CostMeter | None = None,
    registry: ModelRegistry | None = None,
) -> tuple[OpenRouterClient, list[httpx.Request], list[float]]:
    seen: list[httpx.Request] = []
    sleeps: list[float] = []

    def wrapped(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    s = Settings(network=NetworkConfig(max_attempts=3))
    client = OpenRouterClient(
        s,
        "sk-or-test",
        meter or CostMeter(),
        registry,
        transport=httpx.MockTransport(wrapped),
        sleep=sleeps.append,
    )
    return client, seen, sleeps


def body(req: httpx.Request) -> dict[str, Any]:
    return json.loads(req.content)


def test_requires_api_key() -> None:
    with pytest.raises(ConfigError):
        OpenRouterClient(Settings(), "", CostMeter())


def test_stream_text_with_usage_and_preview() -> None:
    previews: list[str] = []

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=sse(
                delta("Hello "),
                delta(
                    "world",
                    "stop",
                    {
                        "prompt_tokens": 100,
                        "completion_tokens": 2,
                        "cost": 0.0002,
                        "prompt_tokens_details": {"cached_tokens": 80},
                    },
                ),
            ),
        )

    meter = CostMeter()
    client, seen, _ = make(handler, meter=meter)
    comp = client.text("writer", "sys", "user", prefix="BIBLE", label="t", on_delta=previews.append)
    assert comp.text == "Hello world"
    assert previews[-1] == "Hello world"
    assert meter.total_cost == pytest.approx(0.0002)
    assert meter.cached_tokens == 80
    b = body(seen[0])
    assert b["stream"] is True
    assert b["usage"] == {"include": True}
    assert b["models"] == ["anthropic/claude-haiku-5.5", "deepseek/deepseek-v4.1-flash-1017"]
    assert b["reasoning"] == {"effort": "none", "exclude": True}
    # Anthropic models get an explicit cache breakpoint after the stable prefix
    assert b["messages"][1]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert seen[0].headers["Authorization"] == "Bearer sk-or-test"


def test_unsupported_params_are_not_sent() -> None:
    reg = ModelRegistry(
        {
            "anthropic/claude-haiku-5.5": ModelInfo(
                "anthropic/claude-haiku-5.5", supported_parameters={"max_tokens", "reasoning"}
            )
        },
        1.0,
    )
    client, seen, _ = make(
        lambda r: httpx.Response(200, content=sse(delta("x", "stop"))), registry=reg
    )
    client.text("writer", "s", "u")
    b = body(seen[0])
    assert "temperature" not in b and "top_p" not in b
    assert "reasoning" in b


def test_retries_rate_limit_honoring_retry_after() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(
                429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}
            )
        return httpx.Response(200, content=sse(delta("ok", "stop")))

    client, _, sleeps = make(handler)
    assert client.text("writer", "s", "u").text == "ok"
    assert sleeps == [7.0]


def test_auth_error_is_not_retried() -> None:
    client, seen, _ = make(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(FatalLLMError) as info:
        client.text("writer", "s", "u")
    assert len(seen) == 1
    assert "config set-key" in info.value.hint


def test_credit_error_has_hint() -> None:
    client, _, _ = make(lambda r: httpx.Response(402, json={"error": {"message": "no credits"}}))
    with pytest.raises(FatalLLMError) as info:
        client.text("writer", "s", "u")
    assert "credits" in info.value.hint


def test_empty_answer_retried_then_fails() -> None:
    client, seen, _ = make(lambda r: httpx.Response(200, content=sse(delta("", "length"))))
    with pytest.raises(EmptyCompletion, match="reasoning"):
        client.text("writer", "s", "u")
    assert len(seen) == 3


def test_5xx_then_success() -> None:
    responses = iter(
        [
            httpx.Response(503, text="overloaded"),
            httpx.Response(200, content=sse(delta("fine", "stop"))),
        ]
    )
    client, _, sleeps = make(lambda r: next(responses))
    assert client.text("writer", "s", "u").text == "fine"
    assert len(sleeps) == 1


def test_mid_stream_error_is_retried() -> None:
    responses = iter(
        [
            httpx.Response(
                200, content=sse(delta("par"), {"error": {"code": 502, "message": "upstream"}})
            ),
            httpx.Response(200, content=sse(delta("full answer", "stop"))),
        ]
    )
    client, _, _ = make(lambda r: next(responses))
    assert client.text("writer", "s", "u").text == "full answer"


def test_length_finish_triggers_continuation() -> None:
    responses = iter(
        [
            httpx.Response(200, content=sse(delta("First half", "length"))),
            httpx.Response(200, content=sse(delta(" second half.", "stop"))),
        ]
    )
    client, seen, _ = make(lambda r: next(responses))
    comp = client.text("writer", "s", "u")
    assert comp.text == "First half second half."
    assert comp.continued == 1
    follow = body(seen[1])["messages"]
    assert follow[-2] == {"role": "assistant", "content": "First half"}
    assert follow[-1]["role"] == "user"


def test_content_filter_falls_back_to_other_model() -> None:
    responses = iter(
        [
            httpx.Response(200, content=sse(delta("", "content_filter"))),
            httpx.Response(200, content=sse(delta("dark prose", "stop"))),
        ]
    )
    client, seen, _ = make(lambda r: next(responses))
    assert client.text("writer", "s", "u").text == "dark prose"
    assert body(seen[1])["model"] == "deepseek/deepseek-v4.1-flash-1017"


def test_structured_native_and_validated() -> None:
    client, seen, _ = make(lambda r: ok_json('{"name": "Lin", "score": 3}'))
    out = client.structured("analyst", "s", "u", Pick)
    assert out == Pick(name="Lin", score=3)
    b = body(seen[0])
    assert b["response_format"]["type"] == "json_schema"
    assert b["response_format"]["json_schema"]["schema"]["additionalProperties"] is False
    assert b["provider"] == {"require_parameters": True}
    assert "stream" not in b


def test_structured_repairs_invalid_json_once() -> None:
    responses = iter(
        [ok_json('```json\n{"name": "Lin"}\n```'), ok_json('{"name": "Lin", "score": 9}')]
    )
    client, seen, _ = make(lambda r: next(responses))
    assert client.structured("analyst", "s", "u", Pick).score == 9
    assert "did not validate" in body(seen[1])["messages"][-1]["content"]


def test_structured_gives_up_after_repair() -> None:
    client, _, _ = make(lambda r: ok_json("not json at all"))
    with pytest.raises(LLMError, match="invalid Pick JSON"):
        client.structured("analyst", "s", "u", Pick)


def test_structured_falls_back_to_prompt_mode_when_unsupported() -> None:
    responses = iter(
        [
            httpx.Response(
                404,
                json={
                    "error": {"message": "No endpoints found that support the requested parameters"}
                },
            ),
            ok_json('{"name": "A", "score": 1}'),
        ]
    )
    client, seen, _ = make(lambda r: next(responses))
    assert client.structured("analyst", "s", "u", Pick).name == "A"
    second = body(seen[1])
    assert "response_format" not in second
    assert "JSON Schema" in second["messages"][0]["content"]


def test_budget_checked_before_every_call() -> None:
    meter = CostMeter(max_cost=0.0015)
    client, seen, _ = make(lambda r: ok_json('{"name": "A", "score": 1}', cost=0.001), meter=meter)
    client.structured("analyst", "s", "u", Pick)
    client.structured("analyst", "s", "u", Pick)
    with pytest.raises(BudgetExceeded):
        client.structured("analyst", "s", "u", Pick)
    assert len(seen) == 2


def test_cost_estimated_from_registry_when_not_reported() -> None:
    reg = ModelRegistry(
        {
            "deepseek/deepseek-v4.1-flash-1017": ModelInfo(
                "deepseek/deepseek-v4.1-flash-1017", prompt_price=1e-6, completion_price=2e-6
            )
        },
        1.0,
    )
    meter = CostMeter()

    def handler(req: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "model": "deepseek/deepseek-v4.1-flash-1017",
                "choices": [
                    {"message": {"content": '{"name":"a","score":1}'}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    client, _, _ = make(handler, meter=meter, registry=reg)
    client.structured("analyst", "s", "u", Pick)
    assert meter.total_cost == pytest.approx(0.002)


def test_timeout_is_retryable() -> None:
    calls = {"n": 0}

    def handler(req: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            raise httpx.ReadTimeout("stalled", request=req)
        return httpx.Response(200, content=sse(delta("late", "stop")))

    client, _, sleeps = make(handler)
    assert client.text("writer", "s", "u").text == "late"
    assert len(sleeps) == 2


def test_embeddings() -> None:
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path.endswith("/embeddings")
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0.0, 1.0]},
                    {"index": 0, "embedding": [1.0, 0.0]},
                ],
                "usage": {"prompt_tokens": 4, "cost": 0.00001},
            },
        )

    client, _, _ = make(handler)
    assert client.embed("openai/text-embedding-3-small", ["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
