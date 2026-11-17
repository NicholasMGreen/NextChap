"""Footer rendering: cost meter aggregation and the status line itself."""

from __future__ import annotations

from io import StringIO

from rich.console import Console
from rich.text import Text

from nextchap.cli.ui import FOOTER_ICON, RichReporter, fmt_tokens
from nextchap.config import Settings
from nextchap.llm.cost import CallRecord, CostMeter


def _settings(**overrides: object) -> Settings:
    return Settings(**overrides)  # type: ignore[arg-type]


def _meter(*records: CallRecord, max_cost: float | None = None) -> CostMeter:
    meter = CostMeter(max_cost)
    for rec in records:
        meter.record(rec)
    return meter


def _footer(meter: CostMeter, settings: Settings | None = None) -> Text:
    console = Console(file=StringIO(), width=200)
    return RichReporter(console, meter, settings or _settings()).footer()


def _styles_of(text: Text, needle: str) -> set[str]:
    start = text.plain.index(needle)
    end = start + len(needle)
    return {str(s.style) for s in text.spans if s.start < end and s.end > start}


def _analyst(
    prompt: int = 100, completion: int = 50, cached: int = 0, cost: float = 0.01
) -> CallRecord:
    return CallRecord(
        role="analyst",
        model="deepseek/deepseek-v4.1-flash-1017",
        label="summary",
        prompt_tokens=prompt,
        completion_tokens=completion,
        cached_tokens=cached,
        cost=cost,
    )


def _writer(cost: float = 0.05) -> CallRecord:
    return CallRecord(
        role="writer",
        model="anthropic/claude-haiku-5.5",
        label="scene",
        prompt_tokens=1000,
        completion_tokens=500,
        cost=cost,
    )


def test_fmt_tokens() -> None:
    assert fmt_tokens(999) == "999"
    assert fmt_tokens(41_200) == "41.2k"
    assert fmt_tokens(1_300_000) == "1.3M"


def test_by_model_totals_match_records() -> None:
    meter = _meter(_analyst(prompt=100, completion=10, cached=40), _analyst(prompt=50, cost=0.02))
    totals = meter.by_model()["deepseek/deepseek-v4.1-flash-1017"]
    assert totals.calls == 2
    assert totals.prompt_tokens == 150
    assert totals.completion_tokens == 60
    assert totals.cached_tokens == 40
    assert totals.cost == 0.03
    assert [t.cost for t in meter.by_model().values()] == [0.03]
    assert meter.by_role() == {"analyst": 0.03}


def test_footer_shows_roles_that_actually_worked() -> None:
    meter = _meter(_analyst(), _analyst(), _writer())
    plain = _footer(meter).plain
    assert "analyst deepseek/deepseek-v4.1-flash-1017 (2)" in plain
    assert "writer anthropic/claude-haiku-5.5 (1)" in plain
    assert "planner" not in plain and "critic" not in plain


def test_footer_hides_writer_when_only_analyst_ran() -> None:
    plain = _footer(_meter(_analyst())).plain
    assert "analyst deepseek/deepseek-v4.1-flash-1017 (1)" in plain
    assert "writer " not in plain


def test_footer_static_fallback_without_records() -> None:
    settings = _settings()
    plain = _footer(_meter(), settings).plain
    assert f"writer {settings.models.writer.model}" in plain
    assert "analyst" not in plain


def test_footer_icon_can_be_disabled() -> None:
    settings = _settings(ui={"footer_icon": False})
    text = _footer(_meter(_analyst(prompt=41_200, completion=15_000)), settings)
    assert FOOTER_ICON not in text.plain
    assert "in 41.2k" in text.plain and "out 15.0k" in text.plain
    assert "analyst deepseek/deepseek-v4.1-flash-1017 (1)" in text.plain


def test_footer_icon_renders_arrow_counter() -> None:
    plain = _footer(_meter(_analyst(prompt=41_200, completion=15_000))).plain
    assert f"{FOOTER_ICON} ↑41.2k ↓15.0k" in plain
    assert "tokens" not in plain


def test_footer_spend_colours_by_budget_fraction() -> None:
    cheap = _footer(_meter(_analyst(cost=0.01), max_cost=1.0))
    assert "bold green" in _styles_of(cheap, "$0.0100")

    half = _footer(_meter(_analyst(cost=0.6), max_cost=1.0))
    assert "bold yellow" in _styles_of(half, "$0.6000")

    nearly = _footer(_meter(_analyst(cost=0.9), max_cost=1.0))
    assert "bold red" in _styles_of(nearly, "$0.9000")


def test_footer_cache_rate_colours() -> None:
    cold = _footer(_meter(_analyst(prompt=1000, cached=0)))
    assert "dim" in _styles_of(cold, "0%")

    warm = _footer(_meter(_analyst(prompt=1000, cached=600)))
    assert "green" in _styles_of(warm, "60%")


def test_footer_without_budget_shows_bold_spend() -> None:
    text = _footer(_meter(_analyst(cost=0.01)))
    assert "bold yellow" in _styles_of(text, "$0.0100")
