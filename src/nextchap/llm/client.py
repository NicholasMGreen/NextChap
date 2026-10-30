"""OpenRouter chat client.

Retries only transient failures (429/5xx/timeouts/empty answers), fails fast on auth,
credit and request errors, continues prose cut off by the token limit, enforces the
spend cap before every request, and records the cost OpenRouter reports for each call.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import ValidationError
from tenacity import (
    RetryCallState,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from nextchap.config import RoleModel, Settings
from nextchap.errors import ConfigError, FatalLLMError, LLMError, RetryableLLMError
from nextchap.llm.base import Completion, DeltaCallback, Role, T, Usage
from nextchap.llm.cost import CallRecord, CostMeter
from nextchap.llm.schema import extract_json, response_format, schema_prompt
from nextchap.models import ModelRegistry

log = logging.getLogger(__name__)

CONTINUE_PROMPT = (
    "Your previous reply was cut off by the length limit. Continue from the exact next "
    "character, without repeating anything and without any preamble. If the next word "
    "starts fresh, begin with a space."
)
RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504, 520, 522, 524, 529}


class EmptyCompletion(RetryableLLMError):
    pass


class StructuredUnsupported(FatalLLMError):
    pass


class _RateLimited(RetryableLLMError):
    def __init__(self, message: str, retry_after: float | None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def _error_message(body: str) -> str:
    try:
        data = json.loads(body)
    except ValueError:
        return body.strip()[:300]
    err = data.get("error") if isinstance(data, dict) else None
    if isinstance(err, dict):
        msg = str(err.get("message", ""))
        raw = (
            (err.get("metadata") or {}).get("raw")
            if isinstance(err.get("metadata"), dict)
            else None
        )
        return f"{msg} ({raw})" if raw and raw not in msg else msg
    return body.strip()[:300]


def status_error(status: int, message: str, retry_after: float | None = None) -> LLMError:
    low = message.lower()
    if status == 401:
        return FatalLLMError(
            f"OpenRouter rejected the API key: {message}",
            hint="Set OPENROUTER_API_KEY, or run `nextchap config set-key`.",
        )
    if status == 402:
        return FatalLLMError(
            f"Insufficient OpenRouter credits: {message}",
            hint="Top up at https://openrouter.ai/settings/credits",
        )
    if status == 403:
        return FatalLLMError(
            f"Request blocked by moderation: {message}",
            hint="Rephrase the plan/style guide, or point the role at another model in nextchap.toml.",
        )
    if status in (400, 404, 422) and any(
        k in low for k in ("response_format", "structured", "require_parameters", "no endpoints")
    ):
        return StructuredUnsupported(f"Structured output unsupported: {message}")
    if status == 429:
        return _RateLimited(f"Rate limited: {message}", retry_after)
    if status in RETRYABLE_STATUS or status >= 500:
        return _RateLimited(f"Provider error {status}: {message}", retry_after)
    return FatalLLMError(f"OpenRouter error {status}: {message}")


def _retry_after(headers: httpx.Headers) -> float | None:
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return min(120.0, max(0.0, float(value)))
    except ValueError:
        return None


_backoff = wait_exponential_jitter(multiplier=2, max=60)


def _wait(state: RetryCallState) -> float:
    exc = state.outcome.exception() if state.outcome else None
    if isinstance(exc, _RateLimited) and exc.retry_after is not None:
        return exc.retry_after
    return float(_backoff(state))


class OpenRouterClient:
    def __init__(
        self,
        settings: Settings,
        api_key: str,
        meter: CostMeter,
        registry: ModelRegistry | None = None,
        *,
        transport: httpx.BaseTransport | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ConfigError(
                "No OpenRouter API key configured.",
                hint="export OPENROUTER_API_KEY=sk-or-...  (or run `nextchap config set-key`)",
            )
        net = settings.network
        self.settings = settings
        self.meter = meter
        self.registry = registry or ModelRegistry.offline()
        self._sleep = sleep
        self._structured_ok: dict[str, bool] = {}
        self._http = httpx.Client(
            base_url=net.base_url,
            transport=transport,
            timeout=httpx.Timeout(net.read_timeout, connect=net.connect_timeout),
            headers={
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": net.app_url,
                "X-Title": net.app_title,
                "X-OpenRouter-Title": net.app_title,
            },
        )

    def close(self) -> None:
        self._http.close()

    # ── public API ────────────────────────────────────────────────────────────

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
    ) -> Completion:
        rc = self._role(role)
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": self._user_content(rc.model, prefix, user)},
        ]
        limit = max_tokens or rc.max_tokens
        comp = self._call(role, rc, messages, limit, label=label, on_delta=on_delta)

        if comp.finish_reason == "content_filter":
            if not rc.fallbacks:
                raise FatalLLMError(
                    f"{rc.model} refused the request (content filter).",
                    hint="Point this role at another model in nextchap.toml.",
                )
            log.warning("%s hit a content filter; retrying on %s", rc.model, rc.fallbacks[0])
            comp = self._call(
                role, rc, messages, limit, label=label, on_delta=on_delta, models=rc.fallbacks
            )

        text, usage, continued = comp.text, comp.usage, 0
        while (
            comp.finish_reason == "length"
            and continued < self.settings.generation.max_continuations
        ):
            continued += 1
            follow: list[dict[str, Any]] = [
                *messages,
                {"role": "assistant", "content": text},
                {"role": "user", "content": CONTINUE_PROMPT},
            ]
            base = text

            def relay(acc: str, base: str = base) -> None:
                if on_delta is not None:
                    on_delta(base + acc)

            comp = self._call(
                role, rc, follow, limit, label=f"{label}:cont{continued}", on_delta=relay
            )
            text = base + comp.text
            usage = _add_usage(usage, comp.usage)
        return Completion(text.strip(), comp.model, comp.finish_reason, usage, continued)

    def structured(
        self,
        role: Role,
        system: str,
        user: str,
        schema: type[T],
        *,
        prefix: str = "",
        label: str = "",
    ) -> T:
        rc = self._role(role)
        native = self._structured_ok.get(rc.model, True) and (
            self.registry.supports(rc.model, "structured_outputs")
            or self.registry.supports(rc.model, "response_format")
        )
        sys_text = system if native else f"{system}\n\n{schema_prompt(schema)}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": sys_text},
            {"role": "user", "content": self._user_content(rc.model, prefix, user)},
        ]
        limit = rc.max_tokens
        try:
            comp = self._call(
                role,
                rc,
                messages,
                limit,
                label=label,
                stream=False,
                fmt=response_format(schema) if native else None,
            )
            if comp.finish_reason == "length":
                comp = self._call(
                    role,
                    rc,
                    messages,
                    limit * 2,
                    label=f"{label}:bigger",
                    stream=False,
                    fmt=response_format(schema) if native else None,
                )
        except StructuredUnsupported:
            if not native:
                raise
            log.info("%s lacks native structured output; using schema-in-prompt", rc.model)
            self._structured_ok[rc.model] = False
            return self.structured(role, system, user, schema, prefix=prefix, label=label)

        try:
            return schema.model_validate_json(extract_json(comp.text))
        except (ValidationError, ValueError) as exc:
            error = str(exc)[:1500]
        repair: list[dict[str, Any]] = [
            *messages,
            {"role": "assistant", "content": comp.text},
            {
                "role": "user",
                "content": f"That JSON did not validate:\n{error}\n\nReturn only the corrected JSON object.\n"
                + schema_prompt(schema),
            },
        ]
        fixed = self._call(role, rc, repair, limit * 2, label=f"{label}:repair", stream=False)
        try:
            return schema.model_validate_json(extract_json(fixed.text))
        except (ValidationError, ValueError) as exc:
            raise LLMError(
                f"{rc.model} returned invalid {schema.__name__} JSON twice: {str(exc)[:300]}",
                hint="Re-run; if it persists, point the analyst role at another model.",
            ) from exc

    def embed(self, model: str, inputs: list[str]) -> list[list[float]]:
        def once() -> list[list[float]]:
            self.meter.check_budget()
            try:
                resp = self._http.post("/embeddings", json={"model": model, "input": inputs})
            except httpx.TimeoutException as exc:
                raise RetryableLLMError(f"Embedding request timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise RetryableLLMError(f"Network error: {exc}") from exc
            if resp.status_code >= 400:
                raise status_error(
                    resp.status_code, _error_message(resp.text), _retry_after(resp.headers)
                )
            data = resp.json()
            usage = data.get("usage") or {}
            tokens = int(usage.get("prompt_tokens") or 0)
            cost = usage.get("cost")
            self.meter.record(
                CallRecord(
                    role="embed",
                    model=model,
                    label="embed",
                    prompt_tokens=tokens,
                    cost=float(cost)
                    if cost is not None
                    else self.registry.estimate_cost(model, tokens, 0),
                )
            )
            rows = sorted(data["data"], key=lambda r: r.get("index", 0))
            return [list(map(float, r["embedding"])) for r in rows]

        vectors: list[list[float]] = self._retrying(once)
        return vectors

    # ── internals ─────────────────────────────────────────────────────────────

    def _role(self, role: Role) -> RoleModel:
        roles = self.settings.models.roles()
        if role not in roles:
            raise ConfigError(f"Unknown model role: {role}")
        return roles[role]

    @staticmethod
    def _user_content(model: str, prefix: str, user: str) -> str | list[dict[str, Any]]:
        if not prefix:
            return user
        if model.startswith("anthropic/"):
            return [
                {"type": "text", "text": prefix, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": user},
            ]
        return f"{prefix}\n\n{user}"

    def _retrying(self, fn: Callable[[], Any]) -> Any:
        retrying = Retrying(
            stop=stop_after_attempt(self.settings.network.max_attempts),
            wait=_wait,
            retry=retry_if_exception_type(RetryableLLMError),
            sleep=self._sleep,
            reraise=True,
            before_sleep=lambda s: log.warning(
                "retry %d after: %s", s.attempt_number, s.outcome.exception() if s.outcome else ""
            ),
        )
        return retrying(fn)

    def _body(
        self,
        rc: RoleModel,
        messages: list[dict[str, Any]],
        max_tokens: int,
        stream: bool,
        fmt: dict[str, Any] | None,
        models: list[str] | None,
    ) -> dict[str, Any]:
        primary = models[0] if models else rc.model
        body: dict[str, Any] = {
            "model": primary,
            "messages": messages,
            "max_tokens": max_tokens,
            "usage": {"include": True},
        }
        chain = models if models else [rc.model, *rc.fallbacks]
        if len(chain) > 1:
            body["models"] = chain
        if rc.temperature is not None and self.registry.supports(primary, "temperature"):
            body["temperature"] = rc.temperature
        if rc.top_p is not None and self.registry.supports(primary, "top_p"):
            body["top_p"] = rc.top_p
        if rc.reasoning is not None and self.registry.supports(primary, "reasoning"):
            body["reasoning"] = {"effort": rc.reasoning, "exclude": True}
        if fmt is not None:
            body["response_format"] = fmt
            body["provider"] = {"require_parameters": True}
        if stream:
            body["stream"] = True
        return body

    def _call(
        self,
        role: Role,
        rc: RoleModel,
        messages: list[dict[str, Any]],
        max_tokens: int,
        *,
        label: str,
        on_delta: DeltaCallback | None = None,
        stream: bool = True,
        fmt: dict[str, Any] | None = None,
        models: list[str] | None = None,
    ) -> Completion:
        body = self._body(rc, messages, max_tokens, stream, fmt, models)

        def once() -> Completion:
            self.meter.check_budget()
            try:
                comp = self._stream(body, on_delta) if stream else self._post(body)
            except httpx.TimeoutException as exc:
                raise RetryableLLMError(f"Request to {body['model']} timed out: {exc}") from exc
            except httpx.TransportError as exc:
                raise RetryableLLMError(f"Network error talking to OpenRouter: {exc}") from exc
            cost = comp.usage.cost
            if cost is None:
                cost = self.registry.estimate_cost(
                    comp.model,
                    comp.usage.prompt_tokens,
                    comp.usage.completion_tokens,
                    comp.usage.cached_tokens,
                )
            self.meter.record(
                CallRecord(
                    role=role,
                    model=comp.model,
                    label=label,
                    prompt_tokens=comp.usage.prompt_tokens,
                    completion_tokens=comp.usage.completion_tokens,
                    cached_tokens=comp.usage.cached_tokens,
                    cost=cost,
                )
            )
            if comp.finish_reason == "error":
                raise RetryableLLMError(f"{comp.model} reported an error mid-response.")
            if not comp.text.strip() and comp.finish_reason != "content_filter":
                why = (
                    " (output budget spent on hidden reasoning; lower the role's reasoning effort)"
                    if comp.finish_reason == "length"
                    else ""
                )
                raise EmptyCompletion(f"{comp.model} returned an empty answer{why}.")
            return comp

        result: Completion = self._retrying(once)
        return result

    def _post(self, body: dict[str, Any]) -> Completion:
        resp = self._http.post("/chat/completions", json=body)
        if resp.status_code >= 400:
            raise status_error(
                resp.status_code, _error_message(resp.text), _retry_after(resp.headers)
            )
        data = resp.json()
        if "error" in data and not data.get("choices"):
            err = data["error"] or {}
            raise status_error(int(err.get("code") or 502), str(err.get("message", "")))
        choice = (data.get("choices") or [{}])[0]
        message = choice.get("message") or {}
        return Completion(
            text=str(message.get("content") or ""),
            model=str(data.get("model") or body["model"]),
            finish_reason=str(choice.get("finish_reason") or "stop"),
            usage=_parse_usage(data.get("usage")),
        )

    def _stream(self, body: dict[str, Any], on_delta: DeltaCallback | None) -> Completion:
        parts: list[str] = []
        finish = "stop"
        model = str(body["model"])
        usage = Usage()
        with self._http.stream("POST", "/chat/completions", json=body) as resp:
            if resp.status_code >= 400:
                resp.read()
                raise status_error(
                    resp.status_code, _error_message(resp.text), _retry_after(resp.headers)
                )
            for line in resp.iter_lines():
                if not line or line.startswith(":") or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except ValueError:
                    continue
                if chunk.get("error"):
                    err = chunk["error"]
                    raise status_error(int(err.get("code") or 502), str(err.get("message", "")))
                model = str(chunk.get("model") or model)
                if chunk.get("usage"):
                    usage = _parse_usage(chunk["usage"])
                for choice in chunk.get("choices") or []:
                    delta = (choice.get("delta") or {}).get("content")
                    if delta:
                        parts.append(delta)
                        if on_delta is not None:
                            on_delta("".join(parts))
                    if choice.get("finish_reason"):
                        finish = str(choice["finish_reason"])
        return Completion("".join(parts), model, finish, usage)


def _parse_usage(raw: Any) -> Usage:
    if not isinstance(raw, dict):
        return Usage()
    details = raw.get("prompt_tokens_details") or {}
    cost = raw.get("cost")
    return Usage(
        prompt_tokens=int(raw.get("prompt_tokens") or 0),
        completion_tokens=int(raw.get("completion_tokens") or 0),
        cached_tokens=int(details.get("cached_tokens") or 0) if isinstance(details, dict) else 0,
        cost=float(cost) if cost is not None else None,
    )


def _add_usage(a: Usage, b: Usage) -> Usage:
    cost = None if a.cost is None and b.cost is None else (a.cost or 0.0) + (b.cost or 0.0)
    return Usage(
        a.prompt_tokens + b.prompt_tokens,
        a.completion_tokens + b.completion_tokens,
        a.cached_tokens + b.cached_tokens,
        cost,
    )
