"""Typed value prompts for the settings browser: parse, validate, retry.

Pure parsing and prompting. Never writes files, never touches settings, and
never contacts the network — model candidates are supplied by the caller.

Code against :class:`SettingSpecProtocol` instead of importing the descriptor
registry, so this module can be unit-tested standalone.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt

__all__ = ["SettingSpecProtocol", "ask_value", "parse_value", "render_value"]

_TRUE_WORDS = {"y", "yes", "true", "1"}
_FALSE_WORDS = {"n", "no", "false", "0"}
_MODEL_ERROR = "expected a model id like vendor/model-name"


class SettingSpecProtocol(Protocol):
    """Structural, read-only view of a setting descriptor (``config_schema.SettingSpec``).

    Declared as properties so a frozen dataclass satisfies it: protocol members are
    invariant, and ``SettingSpec.kind`` is a ``Literal``, which only widens to ``str``
    for a read-only attribute.
    """

    @property
    def key(self) -> str: ...

    # "str" | "int" | "float" | "bool" | "enum" | "list[float]" | "list[str]" | "model" | "model_list"
    @property
    def kind(self) -> str: ...

    @property
    def help(self) -> str: ...

    @property
    def choices(self) -> tuple[str, ...]: ...

    @property
    def minimum(self) -> float | None: ...

    @property
    def maximum(self) -> float | None: ...

    @property
    def optional(self) -> bool: ...


def render_value(kind: str, value: Any) -> str:
    """Current-value text that is easy to read and easy to edit again."""
    if value is None:
        return ""
    if kind == "bool":
        return "true" if value else "false"
    if kind in ("list[float]", "list[str]", "model_list"):
        if isinstance(value, (list, tuple)):
            return ", ".join(_format_item(item) for item in value)
        return str(value)
    return str(value)


def parse_value(spec: SettingSpecProtocol, raw: str) -> Any:
    """Parse user input for a setting, raising ``ValueError`` with a short hint."""
    if not raw.strip():
        if spec.optional:
            return None
        raise ValueError(_empty_message(spec.kind))
    parser = _PARSERS.get(spec.kind)
    if parser is None:
        raise ValueError(f"unsupported type: {spec.kind}")
    return parser(spec, raw)


def ask_value(
    console: Console,
    spec: SettingSpecProtocol,
    current: Any,
    *,
    models: Sequence[str] = (),
) -> tuple[bool, Any]:
    """Prompt until a value parses or the user backs out with ``q``.

    Returns ``(True, new_value)`` when committed, ``(False, current)`` on
    cancel. Invalid input prints the error inline and re-prompts.
    """
    if spec.kind in ("model", "model_list") and models:
        return _ask_model(console, spec, current, models)
    return _ask_scalar(console, spec, current)


# --- parsing helpers -------------------------------------------------------


def _empty_message(kind: str) -> str:
    if kind == "list[float]":
        return "expected at least one number"
    return "a value is required"


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


def _format_item(item: Any) -> str:
    if isinstance(item, bool):
        return "true" if item else "false"
    if isinstance(item, float) and item.is_integer():
        return str(int(item))
    return str(item)


def _check_range(value: float, spec: SettingSpecProtocol) -> float:
    if spec.minimum is not None and value < spec.minimum:
        raise ValueError(f"must be at least {_format_number(spec.minimum)}")
    if spec.maximum is not None and value > spec.maximum:
        raise ValueError(f"must be at most {_format_number(spec.maximum)}")
    return value


def _parse_str(spec: SettingSpecProtocol, raw: str) -> str:
    return raw.rstrip()


def _parse_int(spec: SettingSpecProtocol, raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError:
        raise ValueError("expected a whole number") from None
    return int(_check_range(value, spec))


def _parse_float(spec: SettingSpecProtocol, raw: str) -> float:
    try:
        value = float(raw.strip())
    except ValueError:
        raise ValueError("expected a number") from None
    return _check_range(value, spec)


def _parse_bool(spec: SettingSpecProtocol, raw: str) -> bool:
    lowered = raw.strip().lower()
    if lowered in _TRUE_WORDS:
        return True
    if lowered in _FALSE_WORDS:
        return False
    raise ValueError("expected y or n")


def _parse_enum(spec: SettingSpecProtocol, raw: str) -> str:
    text = raw.strip().lower()
    if not spec.choices:
        raise ValueError("no choices are defined for this setting")
    exact = [choice for choice in spec.choices if choice.lower() == text]
    if exact:
        return exact[0]
    prefixes = [choice for choice in spec.choices if choice.lower().startswith(text)]
    if len(prefixes) == 1:
        return prefixes[0]
    allowed = ", ".join(spec.choices)
    if len(prefixes) > 1:
        raise ValueError(f"ambiguous — choose one of: {allowed}")
    raise ValueError(f"expected one of: {allowed}")


def _parse_float_list(spec: SettingSpecProtocol, raw: str) -> list[float]:
    tokens = raw.replace(",", " ").split()
    items: list[float] = []
    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            raise ValueError("expected a number") from None
        items.append(_check_range(value, spec))
    return items


def _parse_str_list(spec: SettingSpecProtocol, raw: str) -> list[str]:
    items: list[str] = []
    for line in raw.splitlines():
        for part in line.split(","):
            item = part.strip()
            if item:
                items.append(item)
    return items


def _validate_model(raw: str) -> str:
    value = raw.strip()
    if not value or value.count("/") != 1 or any(char.isspace() for char in value):
        raise ValueError(_MODEL_ERROR)
    vendor, _, slug = value.partition("/")
    if not vendor or not slug:
        raise ValueError(_MODEL_ERROR)
    return value


def _parse_model(spec: SettingSpecProtocol, raw: str) -> str:
    return _validate_model(raw)


def _parse_model_list(spec: SettingSpecProtocol, raw: str) -> list[str]:
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    return [_validate_model(token) for token in tokens]


_PARSERS = {
    "str": _parse_str,
    "int": _parse_int,
    "float": _parse_float,
    "bool": _parse_bool,
    "enum": _parse_enum,
    "list[float]": _parse_float_list,
    "list[str]": _parse_str_list,
    "model": _parse_model,
    "model_list": _parse_model_list,
}


# --- prompting -------------------------------------------------------------


def _range_text(spec: SettingSpecProtocol) -> str:
    lo = _format_number(spec.minimum) if spec.minimum is not None else None
    hi = _format_number(spec.maximum) if spec.maximum is not None else None
    if lo is not None and hi is not None:
        return f"{lo}–{hi}"
    if lo is not None:
        return f"at least {lo}"
    return f"at most {hi}"


def _prompt_line(spec: SettingSpecProtocol, current: Any) -> str:
    parts = ["Value"]
    if spec.kind == "bool":
        parts.append(escape("[y/n]"))
    elif spec.kind in ("int", "float") and (spec.minimum is not None or spec.maximum is not None):
        parts.append(f"({_range_text(spec)})")
    if spec.optional:
        parts.append("(blank = unset)")
    parts.append(f"(current: {escape(render_value(spec.kind, current))})")
    return " ".join(parts)


def _print_header(console: Console, spec: SettingSpecProtocol) -> None:
    console.print(f"[bold]{escape(spec.key)}[/]")
    console.print(f"[dim]{escape(spec.help)}[/]")
    if spec.kind == "enum":
        console.print(f"[dim]one of: {escape(', '.join(spec.choices))}[/]")


def _ask_scalar(console: Console, spec: SettingSpecProtocol, current: Any) -> tuple[bool, Any]:
    _print_header(console, spec)
    prompt = _prompt_line(spec, current)
    while True:
        raw = Prompt.ask(prompt, default="", console=console, show_default=False)
        if raw.strip() == "q":
            return False, current
        try:
            return True, parse_value(spec, raw)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")


def _ask_model(
    console: Console, spec: SettingSpecProtocol, current: Any, models: Sequence[str]
) -> tuple[bool, Any]:
    ids = list(models)
    current_text = render_value(spec.kind, current)
    current_set = {item.strip() for item in current_text.split(",") if item.strip()}
    hint = (
        "number or id, comma-separated for lists; t = type an id manually; q = cancel"
        if spec.kind == "model_list"
        else "number or id; t = type an id manually; q = cancel"
    )
    while True:
        _print_header(console, spec)
        for index, model_id in enumerate(ids, 1):
            star = " [cyan]*[/]" if model_id in current_set else ""
            console.print(f"  [bold]{index}[/] {escape(model_id)}{star}")
        console.print(f"[dim]{escape(hint)}[/]")
        prompt = f"Model (current: {escape(current_text) if current_text else 'none'})"
        raw = Prompt.ask(prompt, default="", console=console, show_default=False).strip()
        if raw == "q":
            return False, current
        if raw == "t":
            manual = Prompt.ask("Model id", default="", console=console, show_default=False)
            try:
                return True, parse_value(spec, manual)
            except ValueError as exc:
                console.print(f"[red]{exc}[/]")
            continue
        try:
            return True, _resolve_model_selection(spec, raw, ids)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")


def _resolve_model_selection(spec: SettingSpecProtocol, raw: str, ids: Sequence[str]) -> Any:
    tokens = [t.strip() for t in raw.replace("\n", ",").split(",") if t.strip()]
    if not tokens:
        raise ValueError("a value is required")
    if spec.kind == "model" and len(tokens) > 1:
        raise ValueError("expected one model id")
    resolved: list[str] = []
    for token in tokens:
        if token.isdigit() and 1 <= int(token) <= len(ids):
            resolved.append(ids[int(token) - 1])
        else:
            resolved.append(_validate_model(token))
    if spec.kind == "model":
        return resolved[0]
    return resolved
