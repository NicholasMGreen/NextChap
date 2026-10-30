"""End-to-end CLI runs with the fake model client and hash embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextchap.cli import app as app_mod
from nextchap.cli.context import AppContext
from nextchap.cli.legacy import book_manager, novel_generator
from nextchap.llm.cost import CostMeter
from tests.fakes import FakeLLM


@pytest.fixture
def cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("NEXTCHAP_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("NEXTCHAP_RETRIEVAL__EMBED_BACKEND", "hash")
    monkeypatch.setattr(AppContext, "client", lambda self, meter: FakeLLM(meter))

    def run(*args: str) -> tuple[int, str, str]:
        code = app_mod.main(["--plain", *args])
        out, err = capsys.readouterr()
        return code, out, err

    return run


def test_full_workflow(cli, novel_epub: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("import", str(novel_epub), "--dry-run")
    assert code == 0 and "Detected chapters" in err and "Estimated import cost" in err
    assert not (tmp_path / "data" / "books" / "novel").exists()

    code, _, err = cli("import", str(novel_epub), "--yes")
    assert code == 0, err
    assert "Imported" in err

    code, _, err = cli("import", str(novel_epub), "--yes")
    assert code == 1 and "already exists" in err

    code, _, err = cli("generate", "novel", "--dry-run", "-n", "3")
    assert code == 0 and "Estimated cost for 3 chapter(s)" in err

    code, _, err = cli("generate", "novel", "-n", "2")
    assert code == 0, err
    assert "2 chapter(s) added" in err and "done: chapter 5" in err

    code, out, _ = cli("books", "list", "--json")
    rows = json.loads(out)
    assert rows[0]["book_id"] == "novel" and rows[0]["chapters"] == 6 and rows[0]["generated"] == 2

    code, out, _ = cli("books", "info", "novel", "--json")
    info = json.loads(out)
    assert info["world_state"]["chapter_count"] == 6 and info["total_spend"] > 0

    code, out, _ = cli("state", "show", "novel")
    assert code == 0 and "Lin Feng" in out
    code, out, _ = cli("state", "diff", "novel")
    assert code == 0 and "Mystery 6" in out

    out_epub = tmp_path / "x.epub"
    code, _, err = cli("export", "epub", "novel", "-o", str(out_epub))
    assert code == 0 and out_epub.exists()

    code, _, err = cli("index", "rebuild", "novel")
    assert code == 0 and "Indexed" in err

    code, _, err = cli("state", "rollback", "novel", "--to", "5", "--truncate", "--yes")
    assert code == 0 and "removed chapters 6-6" in err
    code, out, _ = cli("books", "list", "--json")
    assert json.loads(out)[0]["chapters"] == 5

    code, _, err = cli("clean", "novel")
    assert code == 0

    code, _, err = cli("books", "delete", "novel", "--yes")
    assert code == 0
    code, out, _ = cli("books", "list", "--json")
    assert json.loads(out) == []


def test_path_traversal_rejected(cli) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("books", "delete", "../../etc", "--yes")
    assert code == 1 and "Invalid book ID" in err


def test_unknown_book(cli) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("generate", "nope")
    assert code == 1 and "not found" in err


def test_config_set_and_show(cli, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("config", "set", "generation.min_words", "1200", "--project")
    assert code == 0
    assert "min_words = 1200" in (tmp_path / "nextchap.toml").read_text()
    code, out, _ = cli("config", "show", "--json")
    assert json.loads(out)["generation"]["min_words"] == 1200
    code, _, err = cli("config", "set", "generation.min_words", '"lots"', "--project")
    assert code == 1 and "Invalid value" in err
    code, _, err = cli("config", "set", "api_key", "sk-or-x")
    assert code == 1


def test_models_offline_json(cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from nextchap.models import ModelRegistry

    monkeypatch.setattr(AppContext, "registry", lambda self, refresh=False: ModelRegistry.offline())
    code, out, _ = cli("models", "--json")
    data = json.loads(out)
    assert code == 0 and data["live"] is False
    assert {r["role"] for r in data["roles"]} == {"writer", "planner", "analyst", "critic"}


def test_budget_flag_stops_generation(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt), "--yes")[0] == 0
    code, _, err = cli("generate", "novel", "-n", "1", "--max-cost", "0.002")
    assert code == 1 and "max-cost" in err
    # progress is kept; a later run with budget resumes the same chapter
    code, _, err = cli("generate", "novel", "-n", "1")
    assert code == 0 and "Resuming chapter 7" in err


def test_legacy_shims(cli, novel_txt: Path, capsys: pytest.CaptureFixture[str]) -> None:  # type: ignore[no-untyped-def]
    assert novel_generator([str(novel_txt), "1", "--book-id", "Legacy Book"]) == 0
    err = capsys.readouterr().err
    assert "deprecated" in err
    assert book_manager(["list"]) == 0
    assert "legacy_book" in capsys.readouterr().out
    assert book_manager(["rebuild-state", "Legacy Book", "--dry-run"]) == 0
    assert book_manager(["rebuild-chroma", "legacy_book"]) == 0


def test_legacy_shim_requires_migration(cli, tmp_path: Path, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "novel_books" / "old").mkdir(parents=True)
    (tmp_path / "novel_books" / "old" / "master_book.md").write_text("## Chapter 1: A\n\nx\n")
    assert novel_generator([str(novel_txt), "1", "--book-id", "old"]) == 1
    code, _, err = cli("migrate", "--yes")
    assert code == 0 and "migrated" in err


def test_doctor_fails_without_key(cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from nextchap.models import ModelRegistry

    monkeypatch.setattr(AppContext, "registry", lambda self, refresh=False: ModelRegistry.offline())
    code, out, _ = cli("doctor", "--json")
    checks = {c["check"]: c["status"] for c in json.loads(out)}
    assert code == 1 and checks["API key"] == "fail" and checks["Python"] == "ok"


def test_meter_starts_clean() -> None:
    assert CostMeter().total_cost == 0
