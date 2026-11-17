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
    assert "Step" in err
    assert not (tmp_path / "data" / "novel_books" / "novel").exists()

    code, _, err = cli("import", str(novel_epub), "--yes")
    assert code == 0, err
    assert "Imported" in err

    code, _, err = cli("import", str(novel_epub), "--yes")
    assert code == 1 and "already exists" in err

    code, _, err = cli("generate", "novel", "--dry-run", "-n", "3")
    assert code == 0 and "Estimated cost for 3 chapter(s)" in err and "Step" in err

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

    code, out, _ = cli("status", "novel", "--json")
    data = json.loads(out)
    assert code == 0 and data["index"]["ok"] is True and data["index"]["chunks"] > 0
    assert data["journal"] is None and "[yellow]" not in out

    code, _, err = cli("generate", "novl", "--dry-run")
    assert code == 0 and "using book" in err and "novel" in err

    code, _, err = cli("fix", "novel")
    assert code == 0 and ("no problems found" in err or "Diagnosis" in err)

    code, _, err = cli("fix", "novel", "--rebuild-index")
    assert code == 0 and "Indexed" in err


def test_fix_guards_and_exact_delete(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0

    code, _, err = cli("fix", "novel", "--rebuild-index", "--clean")
    assert code == 1 and "one repair at a time" in err
    code, _, err = cli("fix", "novel", "--clean", "--dry-run")
    assert code == 1 and "--dry-run only applies" in err

    code, _, err = cli("books", "delete", "novl", "--yes")
    assert code == 1 and "exact ID" in err
    code, out, _ = cli("status", "novel", "--json")
    assert json.loads(out)["book_id"] == "novel"

    code, _, err = cli("books", "delete", "novel", "--yes")
    assert code == 0 and "Deleted" in err


def test_fix_interactive_repairs(
    cli, novel_txt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))

    monkeypatch.setattr(app_mod.Prompt, "ask", lambda *a, **k: "1")
    code, _, err = cli("fix", "novel")
    assert code == 0 and "Indexed" in err

    monkeypatch.setattr(app_mod.Prompt, "ask", lambda *a, **k: "2")
    code, _, err = cli("fix", "novel")
    assert code == 0 and "Renumbered" in err

    master = tmp_path / "data" / "novel_books" / "novel" / "master_book.md"
    master.write_text(
        master.read_text(encoding="utf-8") + "\nRead 10 chapters in advance on Patreon!\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(app_mod.Prompt, "ask", lambda *a, **k: "3")
    code, _, err = cli("fix", "novel")
    assert code == 0 and "Removed" in err


def test_fix_interactive_rollback(
    cli, novel_txt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.config import Settings
    from nextchap.pipeline.book import Book

    book = Book.open(Settings(data_dir=tmp_path / "data"), "novel")
    book.world.save(book.world.load().model_copy(update={"chapter_count": 5}))

    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))
    answers = iter(["4", "5"])
    monkeypatch.setattr(app_mod.Prompt, "ask", lambda *a, **k: next(answers))
    monkeypatch.setattr(app_mod.Confirm, "ask", lambda *a, **k: True)
    code, _, err = cli("fix", "novel")
    assert code == 0 and "restored to chapter 5" in err and "removed chapters 6-6" in err


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


def test_query_returns_matching_chapter(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    code, out, err = cli("query", "novel", "src3", "-k", "1")
    assert code == 0, err
    assert "Query: src3" in out and "Book: novel" in out and "Ch 03" in out


def test_query_json_shape_and_single_positional(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    code, out, err = cli("query", "src3", "--json")
    assert code == 0, err
    rows = json.loads(out)
    assert rows and set(rows[0]) == {"chapter", "score", "excerpt"}
    assert rows[0]["chapter"] == 3


def test_query_fuzzy_book_and_explicit_book(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    for partial in ("nov", "novl"):
        code, out, err = cli("query", partial, "src3", "--json")
        assert code == 0, err
        assert json.loads(out)[0]["chapter"] == 3


def test_query_no_question_non_tty(cli, novel_txt: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    code, _, err = cli("query", "novel")
    assert code == 1 and "No question given" in err


def test_query_missing_index_errors_with_fix_hint(cli, novel_txt: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.config import Settings
    from nextchap.retrieval.store import VectorStore

    book_dir = tmp_path / "data" / "novel_books" / "novel"
    VectorStore.reset(book_dir / "index")
    code, _, err = cli("query", "novel", "src3")
    assert code == 1 and "nextchap fix" in err and "rebuild-index" in err
    assert Settings(data_dir=tmp_path / "data").data_root.exists()


def test_query_embedding_mismatch_does_not_rebuild(cli, novel_txt: Path, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.config import RetrievalConfig, Settings
    from nextchap.ingest.master_book import Chapter
    from nextchap.retrieval.embed import HashEmbedder
    from nextchap.retrieval.store import VectorStore

    index = tmp_path / "data" / "novel_books" / "novel" / "index"
    VectorStore.reset(index)
    VectorStore(index, HashEmbedder(dim=64), RetrievalConfig()).upsert_chapters(
        [Chapter(1, "T", "src3 words here")]
    )
    code, _, err = cli("query", "novel", "src3")
    assert code == 1 and "embed_model" in err and "nextchap fix" in err
    info = VectorStore.inspect(index)
    assert info is not None and info.chunks == 1  # no rebuild happened
    assert Settings(data_dir=tmp_path / "data").retrieval.embed_backend == "hash"


def test_config_keys_lists_schema(cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = cli("config", "keys", "--json")
    rows = json.loads(out)
    assert code == 0
    by_key = {r["key"]: r for r in rows}
    model = by_key["models.writer.model"]
    assert model["kind"] == "model" and model["help"]
    assert by_key["api_key"]["secret"] is True and by_key["api_key"]["editable"] is False
    monkeypatch.setenv("COLUMNS", "250")  # keep rich from wrapping the keys table
    code, out, _ = cli("config", "keys")
    assert code == 0 and "models.writer.model" in out and "api_key" in out


def test_config_get_sources_and_unknown(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    from nextchap import config_file

    monkeypatch.setattr(config_file, "user_config_path", lambda: tmp_path / "no-user-config.toml")
    code, out, _ = cli("config", "get", "generation.min_words")
    assert code == 0 and "generation.min_words = 900 (default, default 900)" in out

    code, _, err = cli("config", "set", "generation.min_words", "1200", "--project")
    assert code == 0, err
    code, out, _ = cli("config", "get", "generation.min_words")
    assert code == 0 and "generation.min_words = 1200 (project, default 900)" in out

    code, out, _ = cli("config", "get", "generation.min_words", "--json")
    record = json.loads(out)
    assert record["source"] == "project" and record["value"] == 1200
    assert record["default"] == 900 and record["changed"] is True and record["help"]

    code, _, err = cli("config", "get", "nope.missing")
    assert code == 1 and "unknown setting: nope.missing" in err and "config keys" in err


def test_config_list_changed_and_all(cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from nextchap import config_file

    monkeypatch.setattr(config_file, "user_config_path", lambda: tmp_path / "no-user-config.toml")
    monkeypatch.delenv("NEXTCHAP_RETRIEVAL__EMBED_BACKEND", raising=False)

    code, out, _ = cli("config", "list")
    assert code == 0 and "Nothing changed from the defaults." in out and "--all" in out

    code, _, err = cli("config", "set", "generation.min_words", "1200", "--project")
    assert code == 0, err
    code, out, _ = cli("config", "list")
    assert code == 0 and "generation.min_words" in out and "project" in out
    code, out, _ = cli("config", "list", "--all")
    assert code == 0 and "generation.min_words" in out and "models.writer.model" in out
    code, out, _ = cli("config", "list", "--json")
    rows = json.loads(out)
    assert len(rows) == 1 and rows[0]["key"] == "generation.min_words"
    code, out, _ = cli("config", "list", "--all", "--json")
    assert len(json.loads(out)) > 20

    code, _, err = cli("config", "list", "--section", "bogus")
    assert code == 1 and "unknown section" in err


def test_config_set_atomic_with_backup_and_restore(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    from nextchap import config_file

    monkeypatch.setattr(config_file, "user_config_path", lambda: tmp_path / "no-user-config.toml")

    def boom(self: Path, text: str, encoding: str = "utf-8") -> int:
        raise AssertionError("config set must write through storage.atomic, not Path.write_text")

    monkeypatch.setattr(Path, "write_text", boom)
    code, _, err = cli("config", "set", "generation.min_words", "1200", "--project")
    assert code == 0, err
    toml = tmp_path / "nextchap.toml"
    assert not toml.with_name("nextchap.toml.bak").exists()  # first write has nothing to back up

    monkeypatch.setattr(Path, "write_text", boom)  # re-arm in case a prior write replaced it
    code, _, err = cli("config", "set", "generation.min_words", "1300", "--project")
    assert code == 0, err
    assert toml.with_name("nextchap.toml.bak").exists()
    assert "min_words = 1200" in toml.with_name("nextchap.toml.bak").read_text()

    code, _, err = cli("config", "unset", "generation.min_words", "--project", "--restore")
    assert code == 0, err
    assert "min_words = 1200" in toml.read_text()


def test_config_unset_absent_key_is_clean(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    from nextchap import config_file

    monkeypatch.setattr(config_file, "user_config_path", lambda: tmp_path / "no-user-config.toml")
    code, out, _ = cli("config", "unset", "generation.min_words", "--project")
    assert code == 0 and "not set" in out
    code, _, err = cli("config", "unset", "generation.min_words", "--project", "--restore")
    assert code == 1 and "No backup" in err


def test_config_edit_validates_before_saving(
    cli, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:  # type: ignore[no-untyped-def]
    import click

    from nextchap import config_file
    from nextchap.cli import commands_extra

    monkeypatch.setattr(config_file, "user_config_path", lambda: tmp_path / "no-user-config.toml")
    toml = tmp_path / "nextchap.toml"

    monkeypatch.setattr(click, "edit", lambda *a, **k: "[defaults]\nchapters = 2\n")
    code, _, err = cli("config", "edit", "--project")
    assert code == 0, err
    assert "chapters = 2" in toml.read_text()
    code, out, _ = cli("config", "get", "defaults.chapters")
    assert "defaults.chapters = 2 (project" in out

    good = toml.read_text()
    monkeypatch.setattr(click, "edit", lambda *a, **k: "[defaults]\nchapters = 0\n")
    code, _, err = cli("config", "edit", "--project")
    assert code == 1 and "Invalid value" in err
    assert toml.read_text() == good  # the invalid edit left the file untouched

    monkeypatch.setattr(click, "edit", lambda *a, **k: 'api_key = "sk-or-x"\n')
    code, _, err = cli("config", "edit", "--project")
    assert code == 1 and "Refusing to store the API key" in err
    assert toml.read_text() == good

    monkeypatch.setattr(commands_extra.click, "edit", lambda *a, **k: None)
    code, _, err = cli("config", "edit", "--project")
    assert code == 0 and "No changes" in err


def test_config_bare_command_non_tty_prints_summary(cli) -> None:  # type: ignore[no-untyped-def]
    code, out, err = cli("config")
    assert code == 0 and "NextChap settings" in out
    assert "Non-interactive" in err


def test_doctor_settings_row_warns_on_env_overrides(cli, monkeypatch: pytest.MonkeyPatch) -> None:  # type: ignore[no-untyped-def]
    from nextchap.models import ModelRegistry

    monkeypatch.setattr(AppContext, "registry", lambda self, refresh=False: ModelRegistry.offline())
    _code, out, _ = cli("doctor", "--json")  # no API key under test, so doctor exits 1
    checks = {c["check"]: c for c in json.loads(out)}
    assert checks["settings"]["status"] == "warn"
    assert "NEXTCHAP_RETRIEVAL__EMBED_BACKEND" in checks["settings"]["detail"]

    monkeypatch.delenv("NEXTCHAP_RETRIEVAL__EMBED_BACKEND", raising=False)
    _code, out, _ = cli("doctor", "--json")
    checks = {c["check"]: c for c in json.loads(out)}
    assert checks["settings"]["status"] == "ok"


def test_query_action_in_menu(
    cli,
    novel_txt: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.cli import menu

    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))
    answers = iter(["4", "src3", "8"])
    monkeypatch.setattr(menu.Prompt, "ask", lambda *a, **k: next(answers))
    code = app_mod.main([])
    out, err = capsys.readouterr()
    assert code == 0
    assert "query passages (no API cost)" in err
    assert "Ch 03" in out


def test_menu_dispatches_to_settings(
    cli, novel_txt: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    assert cli("import", str(novel_txt))[0] == 0
    from nextchap.cli import menu

    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))
    # menu.Prompt and settings_browser.Prompt are the same rich.prompt.Prompt class
    answers = iter(["7", "q"])  # 7 = settings action, then quit the browser
    monkeypatch.setattr(menu.Prompt, "ask", lambda *a, **k: next(answers))
    code = app_mod.main([])
    _, err = capsys.readouterr()
    assert code == 0
    assert "settings (models, defaults, generation…)" in err
    assert "NextChap settings" in err  # the browser opened and quit cleanly


def test_menu_without_books_offers_settings(
    cli, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:  # type: ignore[no-untyped-def]
    from nextchap.cli import menu

    monkeypatch.setattr(AppContext, "interactive", property(lambda self: True))
    # menu.Prompt and settings_browser.Prompt are the same rich.prompt.Prompt class
    answers = iter(["1", "q"])  # 1 = settings (the only action offered), then quit
    monkeypatch.setattr(menu.Prompt, "ask", lambda *a, **k: next(answers))
    code = app_mod.main([])
    _, err = capsys.readouterr()
    assert code == 0
    assert "No books yet" in err and "nextchap import" in err
    assert "NextChap settings" in err  # choosing 1 opened the settings browser
