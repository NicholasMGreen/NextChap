from __future__ import annotations

import json

import pytest

from nextchap.errors import BudgetExceeded, GenerationAborted
from nextchap.ingest import master_book
from nextchap.pipeline.book import Book
from nextchap.pipeline.engine import Engine
from tests.conftest import make_engine
from tests.fakes import FakeLLM


def test_import_builds_all_memory(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, book = imported
    p = book.paths
    assert len(book.master().chapters) == 6
    assert len(book.summaries.load()) == 6
    assert book.world.load().chapter_count == 6
    assert book.world.load().characters[0].id == "char_lin_feng"
    assert book.bible().startswith("# Story bible")
    assert len(book.arcs.load()) == 1
    assert p.style.exists()
    assert set(engine.store.indexed()) == set(range(1, 7))
    assert len(fake.labels("summary:")) == 6


def test_sync_is_idempotent(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, _ = imported
    before = len(fake.calls)
    engine.sync()
    assert len(fake.calls) == before


def test_generate_two_chapters(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, book = imported
    done = engine.generate(2)
    assert [c.number for c in done] == [7, 8]
    mb = book.master()
    assert [c.number for c in mb.chapters][-2:] == [7, 8]
    assert mb.by_number()[7].title == "Generated 7"
    assert "* * *" in mb.by_number()[7].content
    state = book.world.load()
    assert state.chapter_count == 8
    assert state.last_chapter_hook == "Hook after chapter 8"
    assert {t.description for t in state.active_threads} >= {"Mystery 7", "Mystery 8"}
    assert book.meta().generated == [7, 8]
    assert not book.paths.journal.exists()
    assert not book.paths.staging.exists() or not any(book.paths.staging.iterdir())
    assert (book.paths.generated / "chapter_0007.md").exists()
    assert (book.paths.generated / "chapter_0007_plan.md").exists()
    assert book.paths.combined.read_text(encoding="utf-8").count("# Chapter") == 2
    assert book.paths.export_epub("novel").exists()
    assert book.world.snapshots()[-2:] == [7, 8]
    assert len(fake.labels("write:ch7")) == 3


def test_crash_during_commit_recovers_without_double_apply(
    imported: tuple[Engine, FakeLLM, Book], settings: object
) -> None:
    engine, _, book = imported
    original = engine.store.upsert_chapters
    calls = {"n": 0}

    def boom(chapters: list[master_book.Chapter]) -> int:
        calls["n"] += 1
        raise RuntimeError("power cut")

    engine.store.upsert_chapters = boom  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        engine.generate(1)
    assert json.loads(book.paths.journal.read_text())["step"] == "committing"
    assert book.master().last_number == 7  # master book written before the crash

    engine2, fake2, _ = make_engine(settings)  # type: ignore[arg-type]
    engine2.store.upsert_chapters = original  # type: ignore[method-assign]
    engine2._store = None
    engine2.generate(1)
    mb = book.master()
    assert [c.number for c in mb.chapters][-2:] == [7, 8]
    assert sum(1 for c in mb.chapters if c.number == 7) == 1
    assert book.meta().generated == [7, 8]
    assert book.world.load().chapter_count == 8
    # chapter 7 was not re-planned or re-written after the crash
    assert not fake2.labels("plan:ch7") and not fake2.labels("write:ch7")
    assert set(engine2.store.indexed()) == set(range(1, 9))
    assert sorted(book.summaries.load()) == list(range(1, 9))


def test_crash_mid_writing_resumes_from_last_scene(
    imported: tuple[Engine, FakeLLM, Book], settings: object
) -> None:
    engine, fake, book = imported

    def die_on_scene_three(role: str, label: str) -> None:
        if label == "write:ch7:s3":
            raise KeyboardInterrupt

    fake.before_call = die_on_scene_three
    with pytest.raises(KeyboardInterrupt):
        engine.generate(1)
    engine2, fake2, _ = make_engine(settings)  # type: ignore[arg-type]
    engine2.generate(1)
    assert fake2.labels("write:ch7") == ["write:ch7:s3"]
    assert not fake2.labels("plan:ch7")
    assert book.master().last_number == 7


def test_budget_stops_between_chapters(settings: object, novel_txt: object) -> None:
    engine, fake, _book = make_engine(settings, max_cost=0.5)  # type: ignore[arg-type]
    from nextchap.ingest.text import parse_text
    from nextchap.pipeline.book import BookMeta

    engine.import_chapters(
        parse_text(novel_txt.read_text()).chapters,  # type: ignore[attr-defined]
        BookMeta(book_id="test"),
        None,
    )
    fake.cost_per_call = 0.04
    engine.meter.max_cost = engine.meter.total_cost + 0.5
    done = engine.generate(5)
    assert 1 <= len(done) < 5
    assert engine.meter.total_cost <= engine.meter.max_cost + 0.5


def test_budget_exceeded_mid_chapter_keeps_progress(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, _fake, book = imported
    engine.meter.max_cost = engine.meter.total_cost + 0.0025  # allows ~2 calls
    with pytest.raises(BudgetExceeded):
        engine.generate(1)
    assert book.paths.journal.exists()


def test_major_critic_issue_triggers_scene_rewrite(settings: object, novel_txt: object) -> None:
    engine, fake, book = make_engine(settings)  # type: ignore[arg-type]
    from nextchap.ingest.text import parse_text
    from nextchap.pipeline.book import BookMeta

    engine.import_chapters(
        parse_text(novel_txt.read_text()).chapters,  # type: ignore[attr-defined]
        BookMeta(book_id="test"),
        None,
    )
    fake.critic_verdict = "major"
    engine.generate(1)
    assert fake.labels("rewrite:ch7") == ["rewrite:ch7:s2"]
    critic = json.loads((book.paths.generated / "chapter_0007_critic.json").read_text())
    assert critic["report"]["verdict"] == "major"


def test_stop_request_aborts_safely(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, book = imported

    def stop_after_plan(role: str, label: str) -> None:
        if label.startswith("write:ch7:s1"):
            engine.stop_requested = True

    fake.before_call = stop_after_plan
    with pytest.raises(GenerationAborted):
        engine.generate(1)
    assert json.loads(book.paths.journal.read_text())["step"] == "planned"


class ScriptedReview:
    def __init__(self) -> None:
        self.plan_calls = 0

    def review_plan(self, chapter: int, md: str) -> tuple[str, str]:
        self.plan_calls += 1
        if self.plan_calls == 1:
            return "regenerate", "More action please"
        return "accept", md.replace("# Generated", "# Edited")

    def review_chapter(self, chapter: int, prose: str) -> tuple[str, str]:
        return "accept", prose


def test_review_hooks_regenerate_and_edit(settings: object, novel_txt: object) -> None:
    review = ScriptedReview()
    engine, fake, book = make_engine(settings, review=review)  # type: ignore[arg-type]
    from nextchap.ingest.text import parse_text
    from nextchap.pipeline.book import BookMeta

    engine.import_chapters(
        parse_text(novel_txt.read_text()).chapters,  # type: ignore[attr-defined]
        BookMeta(book_id="test"),
        None,
    )
    engine.generate(1)
    assert review.plan_calls == 2
    assert len(fake.labels("plan:ch7")) == 2
    assert book.master().by_number()[7].title.startswith("Edited")


def test_user_appended_chapter_is_ingested(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, book = imported
    master_book.upsert_chapter(
        book.paths.master_book,
        master_book.Chapter(7, "Hand written", "Lin Feng wrote this himself. " * 50),
    )
    engine.sync()
    assert book.world.load().chapter_count == 7
    assert 7 in book.summaries.load()
    assert 7 in engine.store.indexed()
    assert fake.labels("world:ch7")


def test_edited_chapter_is_resummarized(imported: tuple[Engine, FakeLLM, Book]) -> None:
    engine, fake, book = imported
    mb = book.master()
    mb.chapters[2].content += "\n\nA new paragraph about the Azure Dragon Fist."
    master_book.save(book.paths.master_book, mb)
    engine.sync()
    assert fake.labels("summary:ch3") == ["summary:ch3", "summary:ch3"]
