"""End-to-end CLI runs with the fake model client and hash embeddings."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextchap.cli import app as app_mod
from nextchap.cli.context import AppContext
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
    assert not (tmp_path / "data" / "novel_books" / "novel").exists()

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
    code, out, _ = cli("config", "show")
    assert code == 0 and "[defaults]" in out and "None" not in out
    code, _, err = cli("config", "set", "generation.min_words", '"lots"', "--project")
    assert code == 1 and "Invalid value" in err
    code, _, err = cli("config", "set", "api_key", "sk-or-x")
    assert code == 1
    code, _, err = cli("config", "set", "defaults.chapters", "0", "--project")
    assert code == 1 and "Invalid value" in err


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


def test_doctor_fails_without_key(cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from nextchap.models import ModelRegistry

    monkeypatch.setattr(AppContext, "registry", lambda self, refresh=False: ModelRegistry.offline())
    code, out, _ = cli("doctor", "--json")
    checks = {c["check"]: c["status"] for c in json.loads(out)}
    assert code == 1 and checks["API key"] == "fail" and checks["Python"] == "ok"
    assert "Shell completion" in checks


def test_non_tty_import_defaults_to_yes(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("import", str(novel_txt))
    assert code == 0 and "Imported" in err


def test_defaults_precedence(cli, novel_txt: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    code, _, err = cli("generate", "novel", "--dry-run")
    assert code == 0 and "Estimated cost for 1 chapter(s)" in err

    (tmp_path / "nextchap.toml").write_text("[defaults]\nchapters = 3\n", encoding="utf-8")
    code, _, err = cli("generate", "novel", "--dry-run")
    assert code == 0 and "Estimated cost for 3 chapter(s)" in err

    code, _, err = cli("generate", "novel", "--dry-run", "-n", "2")
    assert code == 0 and "Estimated cost for 2 chapter(s)" in err


def test_status_fuzzy_and_fix(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0

    code, out, err = cli("status", "nov")
    assert code == 0 and "Total spend" in out and "Journal" in out and "novel" in out

    code, _, err = cli("generate", "novl", "--dry-run")
    assert code == 0 and "using book" in err and "novel" in err

    code, _, err = cli("fix", "novel")
    assert code == 0 and ("no problems found" in err or "Diagnosis" in err)

    code, _, err = cli("fix", "novel", "--rebuild-index")
    assert code == 0 and "Indexed" in err


def test_status_lists_all_books(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt), "--book-id", "one")[0] == 0
    assert cli("import", str(novel_txt), "--book-id", "two")[0] == 0
    code, out, _ = cli("status", "--json")
    rows = json.loads(out)
    assert code == 0 and {r["book_id"] for r in rows} == {"one", "two"}


def test_export_new_and_legacy_forms(cli, novel_txt: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    new_path = tmp_path / "new.epub"
    code, _, err = cli("export", "novel", "-o", str(new_path))
    assert code == 0 and new_path.exists() and "Wrote" in err

    old_path = tmp_path / "old.epub"
    code, _, err = cli("export", "epub", "novel", "-o", str(old_path))
    assert code == 0 and old_path.exists() and "deprecated" in err


def test_unknown_command_suggests(cli) -> None:  # type: ignore[no-untyped-def]
    code, _, err = cli("statuss")
    assert code == 2 and "Did you mean" in err and "status" in err


def test_interactive_menu_via_bare_command(
    cli, novel_txt: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.cli import menu

    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))
    monkeypatch.setattr(menu.Prompt, "ask", lambda *a, **k: "2")
    code = app_mod.main([])
    out, _ = capsys.readouterr()
    assert code == 0 and "Total spend" in out


def test_meter_starts_clean() -> None:
    assert CostMeter().total_cost == 0


def test_preview_updates_only_on_new_lines() -> None:
    """Per-token preview updates must not touch the panel (they strobe the Live UI)."""
    from rich.console import Console

    from nextchap.cli.ui import RichReporter
    from nextchap.config import Settings

    r = RichReporter(Console(), CostMeter(), Settings())
    r.preview("first words with no line break yet")
    assert r.preview_text == ""  # no newline yet: panel stays untouched
    r.preview("first words\nwith a line break")
    assert r.preview_text.endswith("line break")
    r.preview("first words\nwith a line break plus more")  # grew, but no new line
    assert r.preview_text.endswith("line break")
    r.preview("first words\nwith a line break\nand another")
    assert r.preview_text.endswith("another")
