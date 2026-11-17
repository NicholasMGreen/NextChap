"""Tests for the safe config-file layer (src/nextchap/config_file.py)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import tomli_w

from nextchap import config_file
from nextchap.config_file import (
    backup_path,
    drop_value,
    env_shadow,
    env_var_name,
    layer_path,
    load_layer,
    merge_value,
    read_layer_value,
    restore_layer,
    source_of,
    validate_layer,
    write_layer,
)
from nextchap.errors import ConfigError

# --- helpers -----------------------------------------------------------------


def make_layer(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    config_file.atomic_write_text(path, tomli_w.dumps(data))
    return path


# --- round trip --------------------------------------------------------------


def test_merge_validate_write_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "nextchap.toml"
    assert load_layer(path) == {}

    merged = merge_value({}, "generation.min_words", 1200)
    assert merged == {"generation": {"min_words": 1200}}
    validate_layer(merged, "generation.min_words")
    assert write_layer(path, merged) is None

    data = load_layer(path)
    found, value = read_layer_value(data, "generation.min_words")
    assert found and value == 1200
    assert read_layer_value(data, "generation.nope") == (False, None)
    assert read_layer_value(data, "generation.min_words.deeper") == (False, None)


# --- merge errors ------------------------------------------------------------


def test_merge_non_table_conflict_raises() -> None:
    data = {"generation": 5}
    with pytest.raises(ConfigError) as excinfo:
        merge_value(data, "generation.min_words", 1200)
    assert "conflicts with an existing non-table value" in str(excinfo.value)
    # Pure: the input is untouched.
    assert data == {"generation": 5}


def test_merge_api_key_refused() -> None:
    for key in ("api_key", "api_key.something"):
        with pytest.raises(ConfigError) as excinfo:
            merge_value({}, key, "sk-or-x")
    assert "Refusing to store the API key in a config file." in str(excinfo.value)
    assert excinfo.value.hint == (
        "Use `nextchap config set-key` or the OPENROUTER_API_KEY env var."
    )


def test_merge_rejects_empty_segments_and_trailing_dot() -> None:
    for bad in ("", "a..b", ".a", "a."):
        with pytest.raises(ConfigError):
            merge_value({}, bad, 1)


# --- drop_value --------------------------------------------------------------


def test_drop_value_prunes_empty_tables() -> None:
    data = {"generation": {"min_words": 900}, "models": {"writer": {"top_p": 0.95}}}
    result, found = drop_value(data, "models.writer.top_p")
    assert found
    assert result == {"generation": {"min_words": 900}}

    result, found = drop_value(data, "models.writer.nope")
    assert not found
    assert result == data  # pure, unchanged on miss

    result, found = drop_value(data, "no.such.path")
    assert not found
    assert result == data


# --- atomic write ------------------------------------------------------------


def test_write_layer_uses_atomic_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "nextchap.toml"
    calls: list[Path] = []

    def fake_atomic(p: Path, text: str) -> None:
        calls.append(p)
        with p.open("w", encoding="utf-8") as fh:  # only the fake does the real work
            fh.write(text)

    monkeypatch.setattr(config_file, "atomic_write_text", fake_atomic)

    def no_plain_write(self: Path, text: str, **kwargs: Any) -> None:
        raise AssertionError("plain Path.write_text must not be used")

    monkeypatch.setattr(Path, "write_text", no_plain_write)

    assert write_layer(path, {"defaults": {"chapters": 3}}) is None
    assert calls == [path]
    assert load_layer(path) == {"defaults": {"chapters": 3}}


# --- invalid values leave the file untouched ---------------------------------


def test_invalid_value_leaves_file_untouched(tmp_path: Path) -> None:
    path = tmp_path / "nextchap.toml"
    assert write_layer(path, {"defaults": {"chapters": 3}}) is None
    before = path.read_bytes()

    data = load_layer(path)
    merged = merge_value(data, "defaults.chapters", "not-a-number")
    with pytest.raises(ConfigError) as excinfo:
        validate_layer(merged, "defaults.chapters")
    assert "Invalid value for defaults.chapters" in str(excinfo.value)
    assert "defaults.chapters" in excinfo.value.hint
    assert "nextchap config keys" in excinfo.value.hint
    assert path.read_bytes() == before


# --- backup / restore --------------------------------------------------------


def test_backup_and_restore(tmp_path: Path) -> None:
    path = tmp_path / "nextchap.toml"
    assert write_layer(path, {"defaults": {"chapters": 3}}) is None
    original = path.read_bytes()

    backup = write_layer(path, {"defaults": {"chapters": 5}})
    assert backup == backup_path(path)
    assert backup is not None and backup.exists()
    assert backup.read_bytes() == original
    assert path.read_bytes() != original

    assert restore_layer(path)
    assert path.read_bytes() == original


def test_restore_without_backup_is_false(tmp_path: Path) -> None:
    assert not restore_layer(tmp_path / "nextchap.toml")


# --- env names and provenance ------------------------------------------------


def test_env_var_name_nested() -> None:
    assert env_var_name("generation.min_words") == "NEXTCHAP_GENERATION__MIN_WORDS"
    assert env_var_name("models") == "NEXTCHAP_MODELS"


def test_source_of_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    project = tmp_path / "nextchap.toml"
    user = tmp_path / "user" / "config.toml"
    monkeypatch.setattr(config_file, "PROJECT_CONFIG", Path("nextchap.toml"))
    monkeypatch.setattr(config_file, "user_config_path", lambda: user)
    key = "generation.min_words"

    make_layer(project, {key.split(".")[0]: {key.split(".")[1]: 1500}})
    make_layer(user, {key.split(".")[0]: {key.split(".")[1]: 1600}})
    (tmp_path / ".env").write_text(f"{env_var_name(key)}=1700\n", encoding="utf-8")
    env = {env_var_name(key): "1800"}

    assert source_of(key, cwd=tmp_path, environ=env) == "env"
    assert source_of(key, cwd=tmp_path, environ={}) == "dotenv"

    (tmp_path / ".env").unlink()
    assert source_of(key, cwd=tmp_path, environ={}) == "project"

    project.unlink()
    assert source_of(key, cwd=tmp_path, environ={}) == "user"

    user.unlink()
    assert source_of(key, cwd=tmp_path, environ={}) == "default"


def test_layer_path_uses_project_or_user(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    user = tmp_path / "user" / "config.toml"
    monkeypatch.setattr(config_file, "user_config_path", lambda: user)
    assert layer_path(True, cwd=tmp_path) == tmp_path / "nextchap.toml"
    assert layer_path(True) == Path.cwd() / "nextchap.toml"
    assert layer_path(False) == user


def test_env_shadow_hit_and_miss(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    var = env_var_name("generation.min_words")
    monkeypatch.setenv(var, "800")
    assert env_shadow("generation.min_words", 900) == var
    # Loose normalisation: same value, different spelling is not a shadow.
    assert env_shadow("generation.min_words", "800") is None
    assert env_shadow("generation.min_words", "  800 ") is None

    # Shadow from .env when the real env is clean.
    monkeypatch.delenv(var)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(f'{var}="750"\n', encoding="utf-8")
    assert env_shadow("generation.min_words", 900) == var
    assert env_shadow("generation.min_words", 750) is None

    # Nothing set anywhere: no shadow.
    (tmp_path / ".env").unlink()
    assert env_shadow("generation.min_words", 900) is None


# --- load_layer error path ----------------------------------------------------


def test_load_layer_bad_toml_raises(tmp_path: Path) -> None:
    path = tmp_path / "nextchap.toml"
    path.write_text("[broken\n", encoding="utf-8")
    with pytest.raises(ConfigError) as excinfo:
        load_layer(path)
    assert "not valid TOML" in str(excinfo.value)
    assert "Fix or delete" in excinfo.value.hint
