from __future__ import annotations

import pytest

from nextchap.pipeline.book import Book
from nextchap.pipeline.engine import Engine
from nextchap.pipeline.humanize import (
    flat_rhythm,
    humanize_findings,
    sentence_lengths,
    tell_hits,
)
from nextchap.prompts import PROMPT_VERSION, raw, render
from tests.fakes import FakeLLM

VARIED = (
    "Rain. The courtyard smelled of wet stone and iron, and Lin Feng counted the guards twice "
    "before he moved. Six. He kept to the wall, counting breaths, while the far gate opened on "
    "oiled hinges that had no business being oiled tonight of all nights. Wait. He froze. "
    "Nothing moved but the lantern shadows that had been moving all evening without ever "
    "bothering anybody, and he let the breath out slowly through his teeth."
)


def test_sentence_lengths_ignores_blank_lines_and_separators() -> None:
    assert sentence_lengths("One two three. Four!\n\n* * *\n\nFive six seven?") == [3, 1, 3]


FLAT = [
    "He walked into the hall and looked around at the guards by the door.",
    "She sat down at the table and opened the heavy book on the table.",
    "They talked about the plan for a while before the evening food came.",
    "The meeting ended with nobody saying the thing that really mattered.",
    "He walked into the hall and looked around at the guards by the door.",
    "She read the page and turned it over without saying a word to him.",
    "They left the building together and stood in the rain for a moment.",
    "The lanterns went out one by one along the wall of the courtyard.",
]


def test_flat_rhythm_flags_a_uniform_paragraph() -> None:
    finding = flat_rhythm(" ".join(FLAT))
    assert finding is not None and "Flat sentence rhythm" in finding


def test_flat_rhythm_accepts_varied_prose() -> None:
    assert flat_rhythm(VARIED) is None


def test_flat_rhythm_ignores_short_text() -> None:
    assert flat_rhythm("One two three. Four five six. Seven eight nine.") is None


def test_tell_hits_and_pivot() -> None:
    assert tell_hits("A testament to his will. It turns out nobody came.") == [
        "a testament to",
        "turns out",
    ]
    assert tell_hits("It was not just anger. Not only fear.") == ["not just / not only"]
    assert tell_hits("The rain fell. He waited.") == []


def test_humanize_findings_combines_tells_and_rhythm() -> None:
    prose = "It was a testament to his will. " + VARIED
    findings = humanize_findings(prose)
    assert any("a testament to" in f for f in findings)
    assert not any("rhythm" in f for f in findings)


def test_humanize_findings_clean_prose_is_clean() -> None:
    assert humanize_findings(VARIED) == []


def test_writer_system_carries_the_humanize_rules() -> None:
    rules = raw("humanize")
    system = render("writer_system", pov="third", tone="tense", humanize=rules)
    assert "VOICE AND HUMANITY RULES" in system
    assert "no typos" not in system.lower()
    assert "Never typos" in system


def test_humanize_prompt_has_no_placeholders() -> None:
    assert PROMPT_VERSION == "2.1.0"
    assert "$" not in raw("humanize")


def test_generate_sends_humanize_rules_to_the_writer(
    imported: tuple[Engine, FakeLLM, Book],
) -> None:
    engine, fake, _ = imported
    engine.generate(1)
    systems = [s for role, s, _ in fake.prompts if role == "writer"]
    assert systems and all("VOICE AND HUMANITY RULES" in s for s in systems)


def test_humanize_off_drops_the_rules(
    imported: tuple[Engine, FakeLLM, Book], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, fake, _ = imported
    engine.cfg.humanize = False
    engine.generate(1)
    assert not any("VOICE AND HUMANITY RULES" in s for _, s, _ in fake.prompts)


def test_humanize_findings_reach_the_critic(
    imported: tuple[Engine, FakeLLM, Book], monkeypatch: pytest.MonkeyPatch
) -> None:
    engine, fake, _ = imported
    finding = 'Machine-sounding phrasing: "turns out".'
    monkeypatch.setattr("nextchap.pipeline.engine.humanize_findings", lambda prose: [finding])
    engine.generate(1)
    critic_prompts = [u for role, _, u in fake.prompts if role == "critic"]
    assert critic_prompts and all(finding in u for u in critic_prompts)
