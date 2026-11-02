"""Fuzzy book resolution ladder: omitted, exact, prefix, typo, ambiguous, missing."""

from __future__ import annotations

from pathlib import Path

import pytest
from rich.console import Console

from nextchap.cli import resolve
from nextchap.cli.resolve import AmbiguousBook
from nextchap.config import Settings
from nextchap.errors import BookNotFound, InvalidBookId
from nextchap.storage.paths import books_dir

BOOKS = [
    "deadpool_twilight_v2",
    "chaos_derry",
    "lord_unknown",
]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    s = Settings(data_dir=tmp_path / "data")
    for book_id in BOOKS:
        (books_dir(s.data_root) / book_id).mkdir(parents=True)
    return s


def test_match_priority(settings: Settings) -> None:
    assert resolve.match_book_ids(BOOKS, "chaos_derry") == ["chaos_derry"]
    assert resolve.match_book_ids(BOOKS, "dead") == ["deadpool_twilight_v2"]
    assert set(resolve.match_book_ids(BOOKS, "c")) == {"chaos_derry"}
    assert resolve.match_book_ids(BOOKS, "chaos_deri") == ["chaos_derry"]
    assert resolve.match_book_ids(BOOKS, "zzzzz") == []


def test_omitted_single_book_auto_selects(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = Settings(data_dir=tmp_path / "data")
    (books_dir(s.data_root) / "only_book").mkdir(parents=True)
    paths = resolve.resolve_book(s, None, console=Console(stderr=True, force_terminal=False))
    assert paths.book_id == "only_book"
    assert "using book" in capsys.readouterr().err


def test_omitted_many_books_needs_a_choice(settings: Settings) -> None:
    with pytest.raises(AmbiguousBook, match="Which book"):
        resolve.resolve_book(settings, None, console=Console(), interactive=False)


def test_omitted_many_books_picker(settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resolve.Prompt, "ask", lambda *a, **k: "1")
    paths = resolve.resolve_book(settings, None, console=Console(), interactive=True)
    assert paths.book_id == "chaos_derry"


def test_exact_prefix_and_typo(settings: Settings) -> None:
    console = Console()
    assert resolve.resolve_book(settings, "lord_unknown", console=console).book_id == "lord_unknown"
    assert (
        resolve.resolve_book(settings, "deadp", console=console).book_id == "deadpool_twilight_v2"
    )
    assert resolve.resolve_book(settings, "chaos_deri", console=console).book_id == "chaos_derry"


def test_uppercase_and_spaces_are_normalized(settings: Settings) -> None:
    console = Console()
    assert (
        resolve.resolve_book(settings, "DEADPOOL", console=console).book_id
        == "deadpool_twilight_v2"
    )
    assert resolve.resolve_book(settings, "Lord Unknown", console=console).book_id == "lord_unknown"


def test_require_exact_rejects_prefix_and_typo(settings: Settings) -> None:
    with pytest.raises(BookNotFound, match="exact ID"):
        resolve.resolve_book(
            settings, "deadp", console=Console(), interactive=False, require_exact=True
        )
    assert (
        resolve.resolve_book(
            settings, "deadpool_twilight_v2", console=Console(), require_exact=True
        ).book_id
        == "deadpool_twilight_v2"
    )


def test_ambiguous_prefix_errors_non_interactively(settings: Settings) -> None:
    (books_dir(settings.data_root) / "dead_end").mkdir()
    with pytest.raises(AmbiguousBook, match="dead"):
        resolve.resolve_book(settings, "dead", console=Console(), interactive=False)


def test_ambiguous_prefix_picker_filters(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    (books_dir(settings.data_root) / "dead_end").mkdir()
    monkeypatch.setattr(resolve.Prompt, "ask", lambda *a, **k: "1")
    paths = resolve.resolve_book(settings, "dead", console=Console(), interactive=True)
    assert paths.book_id == "dead_end"


def test_no_books_and_no_match(settings: Settings, tmp_path: Path) -> None:
    with pytest.raises(BookNotFound, match="not found"):
        resolve.resolve_book(settings, "nope", console=Console(), interactive=False)
    empty = Settings(data_dir=tmp_path / "empty")
    with pytest.raises(BookNotFound, match="No books found"):
        resolve.resolve_book(empty, None, console=Console(), interactive=False)


def test_invalid_ids_are_still_rejected(settings: Settings) -> None:
    with pytest.raises(InvalidBookId):
        resolve.resolve_book(settings, "../../etc", console=Console(), interactive=False)
