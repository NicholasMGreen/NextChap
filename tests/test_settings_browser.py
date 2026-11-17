"""Tests for cli/settings_browser.py: the interactive picker, no network, no real user config."""

from __future__ import annotations

import json
import tomllib
from io import StringIO
from pathlib import Path
from typing import Any

import httpx
import pytest
from rich.console import Console

from nextchap import config as config_mod
from nextchap.cli import settings_browser
from nextchap.cli.context import AppContext
from nextchap.cli.settings_browser import model_ids, run_settings_browser
from nextchap.config import DEEPSEEK_FLASH, HAIKU, Settings
from nextchap.config_schema import DESCRIPTORS

_VISIBLE = [s for s in DESCRIPTORS if s.section != "security"]


@pytest.fixture(autouse=True)
def _hash_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEXTCHAP_RETRIEVAL__EMBED_BACKEND", "hash")


@pytest.fixture(autouse=True)
def _layers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point both config layers at tmp_path so no real user config is touched."""
    user = tmp_path / "user-config.toml"
    monkeypatch.setattr(config_mod, "user_config_path", lambda: user)
    monkeypatch.setattr(settings_browser.config_file, "user_config_path", lambda: user)
    monkeypatch.setattr(settings_browser.config_file, "PROJECT_CONFIG", tmp_path / "nextchap.toml")
    return user


@pytest.fixture(autouse=True)
def _cache_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    cache = tmp_path / "cache"
    cache.mkdir()
    monkeypatch.setattr(config_mod.platformdirs, "user_cache_dir", lambda _name: str(cache))
    return cache


def _row(key: str) -> int:
    """The 1-based row number the browser prints for ``key``."""
    return [s.key for s in _VISIBLE].index(key) + 1


def _ctx(tmp_path: Path, *, plain: bool = False) -> tuple[AppContext, Console, Console]:
    out = Console(file=StringIO(), width=200, record=True, legacy_windows=False)
    err = Console(
        file=StringIO(), width=200, force_terminal=True, record=True, legacy_windows=False
    )
    ctx = AppContext(data_dir=tmp_path / "data", plain=plain, out=out, err=err)
    return ctx, out, err


def _text(console: Console) -> str:
    return console.export_text()


def _patch_ask(monkeypatch: pytest.MonkeyPatch, answers: list[str]) -> list[str]:
    """Feed the picker from ``answers``; leftovers prove nothing extra was prompted."""
    queue = list(answers)

    def fake_ask(*args: Any, **kwargs: Any) -> str:
        if not queue:
            raise AssertionError("settings browser consumed more input than expected")
        return queue.pop(0)

    monkeypatch.setattr(settings_browser.Prompt, "ask", fake_ask)
    return queue


def _load(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _line(text: str, key: str) -> str:
    return next(line for line in text.splitlines() if key in line)


# --- the table --------------------------------------------------------------


class TestTable:
    def test_lists_keys_with_source_and_changed_marker(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        _layers.write_text("[defaults]\nreview = true\n", encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, ["q"])

        run_settings_browser(ctx)

        text = _text(err)
        assert f"NextChap settings · {len(_VISIBLE)} keys ·" in text
        assert "── Data location" in text and "data_dir" in text
        # A value supplied by the user layer: user source, changed from default.
        review_line = _line(text, "defaults.review")
        assert "user" in review_line and "*" in review_line
        # An env var wins: env source, changed from default.
        embed_line = _line(text, "retrieval.embed_backend")
        assert "env" in embed_line and "*" in embed_line
        # An untouched key: default source, no change marker.
        model_line = _line(text, "models.writer.model")
        assert "default" in model_line and "*" not in model_line

    def test_filter_narrows_and_clears(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, ["/review", "/", "q"])

        run_settings_browser(ctx)

        chunks = _text(err).split("NextChap settings ·")
        assert len(chunks) == 4  # initial render + /review + cleared
        filtered = chunks[2]
        assert "defaults.review" in filtered
        assert "models.writer.model" not in filtered
        cleared = chunks[3]
        assert "models.writer.model" in cleared
        assert f"{len(_VISIBLE)} keys" in cleared

    def test_section_opens_filtered(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        models_rows = [s for s in _VISIBLE if s.section == "models"]
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, ["q"])

        run_settings_browser(ctx, section="models")

        text = _text(err)
        assert f"· {len(models_rows)} keys" in text
        assert "models.writer.model" in text
        assert "generation.min_words" not in text

    def test_q_exits_cleanly(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, _, err = _ctx(tmp_path)
        queue = _patch_ask(monkeypatch, ["q"])

        run_settings_browser(ctx)

        assert queue == []
        assert "NextChap settings" in _text(err)

    def test_unknown_row_is_reported(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, ["999", "q"])

        run_settings_browser(ctx)

        assert "no such row: 999" in _text(err)


# --- editing one row --------------------------------------------------------


class TestEdit:
    def test_write_to_user_layer_and_backup(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        original = "[defaults]\nreview = true\n"
        _layers.write_text(original, encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, [str(_row("generation.min_words")), "1200", "u", "q"])

        run_settings_browser(ctx)

        assert _load(_layers) == {"defaults": {"review": True}, "generation": {"min_words": 1200}}
        bak = _layers.with_name(_layers.name + ".bak")
        assert bak.read_text(encoding="utf-8") == original
        assert "✓ generation.min_words = 1200" in _text(err)

    def test_invalid_value_reprompts_and_file_untouched(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        original = "[defaults]\nreview = true\n"
        _layers.write_text(original, encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, [str(_row("generation.min_words")), "abc", "q", "q"])

        run_settings_browser(ctx)

        assert _layers.read_text(encoding="utf-8") == original
        assert not _layers.with_name(_layers.name + ".bak").exists()
        assert "expected a whole number" in _text(err)

    def test_blank_on_optional_key_removes_it(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        original = "[defaults]\nmax_cost = 2.5\nreview = true\n"
        _layers.write_text(original, encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, [str(_row("defaults.max_cost")), "", "u", "q"])

        run_settings_browser(ctx)

        data = _load(_layers)
        assert "max_cost" not in data["defaults"]
        assert data["defaults"]["review"] is True
        assert "✓ defaults.max_cost = (unset)" in _text(err)

    def test_write_to_project_layer(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        _layers.write_text("[defaults]\nreview = true\n", encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, [str(_row("defaults.review")), "n", "p", "q"])

        run_settings_browser(ctx)

        project = tmp_path / "nextchap.toml"
        assert _load(project) == {"defaults": {"review": False}}
        assert _load(_layers) == {"defaults": {"review": True}}  # user layer untouched
        assert "✓ defaults.review = false" in _text(err)

    def test_env_shadow_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, _layers: Path
    ) -> None:
        monkeypatch.setenv("NEXTCHAP_GENERATION__MIN_WORDS", "2000")
        _layers.write_text("[defaults]\nreview = true\n", encoding="utf-8")
        ctx, _, err = _ctx(tmp_path)
        _patch_ask(monkeypatch, [str(_row("generation.min_words")), "1200", "u", "q"])

        run_settings_browser(ctx)

        text = _text(err)
        assert "✓ generation.min_words" in text
        assert "! written, but NEXTCHAP_GENERATION__MIN_WORDS still wins" in text


# --- non-interactive --------------------------------------------------------


class TestNonInteractive:
    def test_prints_changed_table_and_prompts_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("NEXTCHAP_GENERATION__MIN_WORDS", "1200")
        ctx, out, err = _ctx(tmp_path, plain=True)
        queue = _patch_ask(monkeypatch, ["SHOULD-NOT-BE-CONSUMED"])

        run_settings_browser(ctx)

        assert queue == ["SHOULD-NOT-BE-CONSUMED"]
        out_text = _text(out)
        assert "generation.min_words" in out_text
        assert "1200" in out_text
        assert "Prompt" not in out_text
        err_text = _text(err)
        assert "Non-interactive: use nextchap config list | get | set." in err_text
        assert "NextChap settings" not in err_text


# --- model_ids --------------------------------------------------------------


class TestModelIds:
    def test_offline_table_only(self, tmp_path: Path, _cache_dir: Path) -> None:
        settings = Settings(data_dir=tmp_path / "data")
        ids = model_ids(settings)
        # The four configured role models come first (writer, planner, analyst, critic).
        assert ids[0] == DEEPSEEK_FLASH and ids[1] == HAIKU
        assert set(ids) == {DEEPSEEK_FLASH, HAIKU}

    def test_never_touches_the_network(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(*args: Any, **kwargs: Any) -> Any:
            raise AssertionError("model_ids must not touch the network")

        monkeypatch.setattr(httpx, "Client", _boom)
        settings = Settings(data_dir=tmp_path / "data")
        assert set(model_ids(settings)) == {DEEPSEEK_FLASH, HAIKU}

    def test_cache_union_and_cap(self, tmp_path: Path, _cache_dir: Path) -> None:
        payload = {
            "fetched_at": 1.0,
            "data": [{"id": f"vendor/model-{i:02d}", "name": f"m{i}"} for i in range(20)],
        }
        (_cache_dir / "models.json").write_text(json.dumps(payload), encoding="utf-8")
        settings = Settings(data_dir=tmp_path / "data")

        ids = model_ids(settings)

        assert len(ids) == 15
        assert ids[0] == DEEPSEEK_FLASH and ids[1] == HAIKU
        assert "vendor/model-00" in ids and "vendor/model-12" in ids
        assert "vendor/model-13" not in ids

    def test_corrupt_cache_is_tolerated(self, tmp_path: Path, _cache_dir: Path) -> None:
        (_cache_dir / "models.json").write_text("not json{{", encoding="utf-8")
        settings = Settings(data_dir=tmp_path / "data")
        assert set(model_ids(settings)) == {DEEPSEEK_FLASH, HAIKU}
