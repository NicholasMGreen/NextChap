"""Tests for cli/prompt_value.py: pure parsing + prompting, no network."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Any

import pytest
from rich.console import Console

from nextchap.cli import prompt_value
from nextchap.cli.prompt_value import ask_value, parse_value, render_value


@dataclass(frozen=True)
class FakeSpec:
    """Local stand-in for the real descriptor registry's SettingSpec."""

    key: str = "test.key"
    kind: str = "str"
    help: str = "help text"
    choices: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    optional: bool = False


def _spec(kind: str, **kwargs: Any) -> FakeSpec:
    return FakeSpec(kind=kind, **kwargs)


def _record_console() -> tuple[Console, io.StringIO]:
    file = io.StringIO()
    console = Console(file=file, width=200, legacy_windows=False, force_terminal=True)
    return console, file


# --- parse_value per kind ---------------------------------------------------


class TestParseStr:
    def test_happy_path(self) -> None:
        assert parse_value(_spec("str"), "hello") == "hello"

    def test_strips_trailing_whitespace(self) -> None:
        assert parse_value(_spec("str"), "hello  ") == "hello"

    def test_required_empty(self) -> None:
        with pytest.raises(ValueError, match="a value is required"):
            parse_value(_spec("str"), "   ")

    def test_optional_empty_is_none(self) -> None:
        assert parse_value(_spec("str", optional=True), "") is None


class TestParseInt:
    def test_happy_path(self) -> None:
        assert parse_value(_spec("int"), "42") == 42

    def test_error_message(self) -> None:
        with pytest.raises(ValueError, match="expected a whole number"):
            parse_value(_spec("int"), "abc")

    def test_minimum(self) -> None:
        spec = _spec("int", minimum=1)
        assert parse_value(spec, "1") == 1
        with pytest.raises(ValueError, match="must be at least 1"):
            parse_value(spec, "0")

    def test_maximum(self) -> None:
        spec = _spec("int", maximum=100)
        assert parse_value(spec, "100") == 100
        with pytest.raises(ValueError, match="must be at most 100"):
            parse_value(spec, "101")

    def test_optional_empty_is_none(self) -> None:
        assert parse_value(_spec("int", optional=True), "  ") is None


class TestParseFloat:
    def test_happy_path(self) -> None:
        assert parse_value(_spec("float"), "1.5") == 1.5

    def test_error_message(self) -> None:
        with pytest.raises(ValueError, match="expected a number"):
            parse_value(_spec("float"), "1.5x")

    def test_range(self) -> None:
        spec = _spec("float", minimum=0.1, maximum=2.0)
        assert parse_value(spec, "1.5") == 1.5
        with pytest.raises(ValueError, match="must be at most 2"):
            parse_value(spec, "2.5")


class TestParseBool:
    @pytest.mark.parametrize("raw", ["y", "Y", "yes", "true", "TRUE", "1"])
    def test_true(self, raw: str) -> None:
        assert parse_value(_spec("bool"), raw) is True

    @pytest.mark.parametrize("raw", ["n", "N", "no", "false", "FALSE", "0"])
    def test_false(self, raw: str) -> None:
        assert parse_value(_spec("bool"), raw) is False

    def test_error_message(self) -> None:
        with pytest.raises(ValueError, match="expected y or n"):
            parse_value(_spec("bool"), "maybe")


class TestParseEnum:
    def test_exact(self) -> None:
        spec = _spec("enum", choices=("local", "remote"))
        assert parse_value(spec, "local") == "local"

    def test_case_insensitive(self) -> None:
        spec = _spec("enum", choices=("local", "remote"))
        assert parse_value(spec, "REMOTE") == "remote"

    def test_unique_prefix(self) -> None:
        spec = _spec("enum", choices=("local", "remote"))
        assert parse_value(spec, "rem") == "remote"

    def test_ambiguous_prefix(self) -> None:
        spec = _spec("enum", choices=("hash", "hasher"))
        with pytest.raises(ValueError, match="ambiguous"):
            parse_value(spec, "ha")

    def test_miss_lists_choices(self) -> None:
        spec = _spec("enum", choices=("local", "remote"))
        with pytest.raises(ValueError, match="expected one of: local, remote"):
            parse_value(spec, "bogus")


class TestParseFloatList:
    def test_happy_path(self) -> None:
        assert parse_value(_spec("list[float]"), "1800, 2400") == [1800.0, 2400.0]

    def test_int_tokens_become_floats(self) -> None:
        assert parse_value(_spec("list[float]"), "1,2,3") == [1.0, 2.0, 3.0]

    def test_empty_items_dropped(self) -> None:
        assert parse_value(_spec("list[float]"), " 1 , , 2 ") == [1.0, 2.0]

    def test_required_empty(self) -> None:
        with pytest.raises(ValueError, match="expected at least one number"):
            parse_value(_spec("list[float]"), "")

    def test_optional_empty_is_none(self) -> None:
        assert parse_value(_spec("list[float]", optional=True), "") is None

    def test_bad_item(self) -> None:
        with pytest.raises(ValueError, match="expected a number"):
            parse_value(_spec("list[float]"), "1, oops")

    def test_range_applies_per_item(self) -> None:
        spec = _spec("list[float]", minimum=1, maximum=100)
        assert parse_value(spec, "1, 100") == [1.0, 100.0]
        with pytest.raises(ValueError, match="must be at most 100"):
            parse_value(spec, "1, 200")
        with pytest.raises(ValueError, match="must be at least 1"):
            parse_value(spec, "0, 2")


class TestParseStrList:
    def test_comma_separated(self) -> None:
        assert parse_value(_spec("list[str]"), "a, b") == ["a", "b"]

    def test_multiline_paste(self) -> None:
        assert parse_value(_spec("list[str]"), "a\nb\n c") == ["a", "b", "c"]

    def test_empty_items_dropped(self) -> None:
        assert parse_value(_spec("list[str]"), "a,, b") == ["a", "b"]

    def test_required_empty(self) -> None:
        with pytest.raises(ValueError, match="a value is required"):
            parse_value(_spec("list[str]"), "   ")


class TestParseModel:
    def test_happy_path(self) -> None:
        assert parse_value(_spec("model"), "anthropic/claude-haiku") == "anthropic/claude-haiku"

    @pytest.mark.parametrize("raw", ["noid", "a/b/c", "a b/c", "/slug", "vendor/"])
    def test_bad_ids(self, raw: str) -> None:
        with pytest.raises(ValueError, match="expected a model id like vendor/model-name"):
            parse_value(_spec("model"), raw)

    def test_required_empty(self) -> None:
        with pytest.raises(ValueError, match="a value is required"):
            parse_value(_spec("model"), "")


class TestParseModelList:
    def test_happy_path(self) -> None:
        spec = _spec("model_list")
        assert parse_value(spec, "a/b, c/d") == ["a/b", "c/d"]

    def test_newline_separated(self) -> None:
        spec = _spec("model_list")
        assert parse_value(spec, "a/b\nc/d") == ["a/b", "c/d"]

    def test_bad_item(self) -> None:
        with pytest.raises(ValueError, match="expected a model id"):
            parse_value(_spec("model_list"), "a/b, noid")


class TestOptionalEmpty:
    def test_every_kind_returns_none(self) -> None:
        kinds = [
            "str",
            "int",
            "float",
            "bool",
            "enum",
            "list[float]",
            "list[str]",
            "model",
            "model_list",
        ]
        for kind in kinds:
            assert parse_value(_spec(kind, optional=True), "  ") is None, kind


# --- render_value round-trips ------------------------------------------------


class TestRenderValue:
    def test_bool(self) -> None:
        assert render_value("bool", True) == "true"
        assert render_value("bool", False) == "false"

    def test_none(self) -> None:
        assert render_value("str", None) == ""
        assert render_value("bool", None) == ""

    def test_list_float(self) -> None:
        assert render_value("list[float]", [1800.0, 2400.0]) == "1800, 2400"

    def test_list_str(self) -> None:
        assert render_value("list[str]", ["a", "b"]) == "a, b"

    def test_model_list(self) -> None:
        assert render_value("model_list", ["a/b", "c/d"]) == "a/b, c/d"

    def test_scalars(self) -> None:
        assert render_value("str", "hello") == "hello"
        assert render_value("int", 42) == "42"
        assert render_value("float", 1.5) == "1.5"
        assert render_value("enum", "remote") == "remote"
        assert render_value("model", "a/b") == "a/b"

    @pytest.mark.parametrize(
        "kind, value",
        [
            ("str", "hello"),
            ("int", 42),
            ("float", 1.5),
            ("bool", True),
            ("enum", "remote"),
            ("list[float]", [1800.0, 2400.0]),
            ("list[str]", ["a", "b"]),
            ("model", "a/b"),
            ("model_list", ["a/b", "c/d"]),
        ],
    )
    def test_round_trip(self, kind: str, value: Any) -> None:
        rendered = render_value(kind, value)
        spec = _spec(kind, choices=("local", "remote") if kind == "enum" else ())
        assert parse_value(spec, rendered) == value


# --- ask_value ---------------------------------------------------------------


class TestAskValue:
    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, inputs: list[str], file: io.StringIO) -> None:
        queue = list(inputs)

        def fake_ask(*args: Any, **kwargs: Any) -> str:
            if not queue:
                raise AssertionError("ask_value consumed more input than expected")
            file.write(f"{args[0]}\n")  # the prompt text ask_value would have shown
            return queue.pop(0)

        monkeypatch.setattr(prompt_value.Prompt, "ask", fake_ask)

    def test_retry_then_commit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["abc", "42"], file)
        committed, value = ask_value(console, _spec("int"), 0)
        assert committed is True
        assert value == 42
        assert re.search(r"expected a whole number", file.getvalue())
        assert re.search(r"\x1b\[31m", file.getvalue())  # red error

    def test_q_cancels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["q"], file)
        committed, value = ask_value(console, _spec("int"), 7)
        assert committed is False
        assert value == 7

    def test_optional_blank_commits_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, [""], file)
        committed, value = ask_value(console, _spec("str", optional=True), "old")
        assert committed is True
        assert value is None

    def test_enum_prompt_shows_choices(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["rem"], file)
        committed, value = ask_value(console, _spec("enum", choices=("local", "remote")), "local")
        assert committed is True
        assert value == "remote"
        assert "one of: local, remote" in file.getvalue()

    def test_range_prompt_shows_min_max(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["50"], file)
        committed, _ = ask_value(console, _spec("int", minimum=1, maximum=100), 10)
        assert committed is True
        assert "1–100" in file.getvalue()

    def test_optional_shows_blank_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["x"], file)
        ask_value(console, _spec("str", optional=True), "old")
        assert "blank = unset" in file.getvalue()


class TestAskModelPicker:
    @staticmethod
    def _patch(monkeypatch: pytest.MonkeyPatch, inputs: list[str], file: io.StringIO) -> None:
        queue = list(inputs)

        def fake_ask(*args: Any, **kwargs: Any) -> str:
            if not queue:
                raise AssertionError("ask_value consumed more input than expected")
            file.write(f"{args[0]}\n")  # the prompt text ask_value would have shown
            return queue.pop(0)

        monkeypatch.setattr(prompt_value.Prompt, "ask", fake_ask)

    def test_select_by_number(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["2"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b", "c/d"))
        assert committed is True
        assert value == "c/d"

    def test_select_by_typed_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["c/d"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b", "c/d"))
        assert committed is True
        assert value == "c/d"

    def test_t_prompts_for_manual_id(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["t", "vendor/new-model"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b",))
        assert committed is True
        assert value == "vendor/new-model"

    def test_t_manual_id_validated(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["t", "noid", "t", "a/b"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b",))
        assert committed is True
        assert value == "a/b"
        assert "expected a model id" in file.getvalue()

    def test_model_list_by_numbers(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["1, 2"], file)
        committed, value = ask_value(console, _spec("model_list"), ["a/b"], models=("a/b", "c/d"))
        assert committed is True
        assert value == ["a/b", "c/d"]

    def test_model_list_by_typed_ids(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["x/y\nc/d"], file)
        committed, value = ask_value(console, _spec("model_list"), ["a/b"], models=("a/b",))
        assert committed is True
        assert value == ["x/y", "c/d"]

    def test_q_cancels(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["q"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b",))
        assert committed is False
        assert value == "a/b"

    def test_marks_current_with_star(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["1"], file)
        ask_value(console, _spec("model"), "a/b", models=("a/b", "c/d"))
        assert "*" in file.getvalue()

    def test_falls_back_to_plain_prompt_without_models(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["a/b"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=())
        assert committed is True
        assert value == "a/b"
        assert "  1 " not in file.getvalue()  # no numbered picker

    def test_out_of_range_number_is_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        console, file = _record_console()
        self._patch(monkeypatch, ["9", "1"], file)
        committed, value = ask_value(console, _spec("model"), "a/b", models=("a/b",))
        assert committed is True
        assert value == "a/b"
        assert "expected a model id" in file.getvalue()
