"""Safe config-layer read-modify-write: load → merge → validate → atomic write → backup.

This module is the single read-modify-write path for the two config layers
(`./nextchap.toml` and the user `config.toml`). Every write goes through
``nextchap.storage.atomic.atomic_write_text``; overwrites keep a one-step
``.bak`` undo.

Layer precedence matches ``Settings.settings_customise_sources``:
init (unused here) > environment > ``.env`` > project TOML > user TOML > default.
"""

from __future__ import annotations

import copy
import os
import tomllib
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Literal

import tomli_w
from pydantic import ValidationError

from nextchap.config import PROJECT_CONFIG, Settings, user_config_path
from nextchap.errors import ConfigError
from nextchap.storage.atomic import atomic_write_text

__all__ = [
    "backup_path",
    "drop_value",
    "env_shadow",
    "env_var_name",
    "layer_path",
    "load_layer",
    "merge_value",
    "read_layer_value",
    "restore_layer",
    "source_of",
    "validate_layer",
    "write_layer",
]

_API_KEY_SEGMENT = "api_key"


def layer_path(project: bool, *, cwd: Path | None = None) -> Path:
    """Path of the config layer: ``./nextchap.toml`` for project, user config otherwise."""
    if project:
        return (cwd or Path.cwd()) / PROJECT_CONFIG
    return user_config_path()


def load_layer(path: Path) -> dict[str, Any]:
    """Load a TOML layer; ``{}`` when the file is missing.

    Raises :class:`ConfigError` when the file exists but is not valid TOML.
    """
    if not path.exists():
        return {}
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid TOML: {exc}",
            hint=(
                f"Fix or delete {path} by hand, then retry "
                "(`nextchap config path` shows both layer files)."
            ),
        ) from exc
    return data


def read_layer_value(data: dict[str, Any], key: str) -> tuple[bool, Any]:
    """``(found, value)`` for a dotted key; ``(False, None)`` when absent or blocked."""
    node: Any = data
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return False, None
        node = node[part]
    return True, node


def merge_value(data: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    """Return a new dict with ``key`` set to ``value``; pure, never mutates ``data``.

    Creates intermediate tables. Raises :class:`ConfigError` when a path segment
    already exists as a non-table value, when the key has empty segments or a
    trailing dot, or when the key would store the API key in a file.
    """
    parts = _split_key(key)
    if parts[0] == _API_KEY_SEGMENT:
        raise ConfigError(
            "Refusing to store the API key in a config file.",
            hint="Use `nextchap config set-key` or the OPENROUTER_API_KEY env var.",
        )
    result = copy.deepcopy(data)
    node = result
    for part in parts[:-1]:
        existing = node.get(part)
        if existing is None:
            node[part] = {}
        elif not isinstance(existing, dict):
            raise ConfigError(
                f"{key} conflicts with an existing non-table value.",
                hint="Remove the conflicting value first (`nextchap config unset <key>`) "
                "or pick a different key.",
            )
        node = node[part]
    node[parts[-1]] = copy.deepcopy(value)
    return result


def drop_value(data: dict[str, Any], key: str) -> tuple[dict[str, Any], bool]:
    """Remove a dotted key; pure. Prunes tables that become empty. Returns ``(data, found)``."""
    parts = _split_key(key)
    result = copy.deepcopy(data)
    node: dict[str, Any] = result
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, dict):
            return result, False
        node = child
    found = parts[-1] in node
    if not found:
        return result, False
    del node[parts[-1]]
    # Prune now-empty tables from the leaf's parent up to the top.
    chain: list[dict[str, Any]] = [result]
    for part in parts[:-1]:
        chain.append(chain[-1][part])
    for depth in range(len(chain) - 1, 0, -1):
        if chain[depth]:
            break
        del chain[depth - 1][parts[depth - 1]]
    return result, True


def validate_layer(data: dict[str, Any], key: str) -> Settings:
    """Validate a whole layer file with :class:`Settings`; raises :class:`ConfigError` on bad values."""
    try:
        return Settings(**data)
    except ValidationError as exc:
        first = exc.errors()[0]
        msg = first.get("msg", "invalid value")
        raise ConfigError(
            f"Invalid value for {key}: {msg}",
            hint=f"Check the type and allowed values for `{key}`: run `nextchap config keys`.",
        ) from exc


def write_layer(path: Path, data: dict[str, Any]) -> Path | None:
    """Atomically write a layer as TOML, backing up existing bytes to ``<name>.bak``.

    Returns the backup path when one was made, else ``None``. ``None`` values are
    dropped recursively so the file stays valid TOML.
    """
    backup: Path | None = None
    if path.exists():
        backup = backup_path(path)
        atomic_write_text(backup, path.read_text(encoding="utf-8"))
    atomic_write_text(path, tomli_w.dumps(_toml_safe(data)))
    return backup


def restore_layer(path: Path) -> bool:
    """Restore the ``.bak`` written by the last :func:`write_layer`; ``True`` when one existed."""
    backup = backup_path(path)
    if not backup.exists():
        return False
    atomic_write_text(path, backup.read_text(encoding="utf-8"))
    return True


def backup_path(path: Path) -> Path:
    """The one-step-undo sibling of a layer file: ``nextchap.toml`` → ``nextchap.toml.bak``."""
    return path.with_name(path.name + ".bak")


def env_var_name(key: str) -> str:
    """The env var that feeds a dotted key: ``generation.min_words`` → ``NEXTCHAP_GENERATION__MIN_WORDS``."""
    return "NEXTCHAP_" + key.replace(".", "__").upper()


def source_of(
    key: str,
    *,
    cwd: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Literal["env", "dotenv", "project", "user", "default"]:
    """Which layer currently supplies ``key`` (same precedence as ``Settings``)."""
    env = environ if environ is not None else os.environ
    for name in _env_names(key):
        if name in env:
            return "env"
    root = cwd or Path.cwd()
    for name, _value in _parse_dotenv(root / ".env"):
        if name in _env_names(key):
            return "dotenv"
    for project_flag in (True, False):
        found, _ = read_layer_value(load_layer(layer_path(project_flag, cwd=root)), key)
        if found:
            return "project" if project_flag else "user"
    return "default"


def env_shadow(key: str, written: Any) -> str | None:
    """Name of the env var that still overrides ``written`` after a successful write, else ``None``.

    Limitation: the comparison is loose — both sides are stringified, stripped of
    surrounding quotes and lowercased — so case- or quoting-only differences
    (``True`` vs ``true``) are *not* reported as shadows, and values that stringify
    identically (``1`` vs ``1.0``) can hide one. Checks ``os.environ`` first, then
    the ``.env`` in the current directory.
    """
    target = _loose_norm(written)
    env_names = _env_names(key)
    for name in env_names:
        value = os.environ.get(name)
        if value is None:
            value = next((v for n, v in _parse_dotenv(Path.cwd() / ".env") if n == name), None)
        if value is not None and _loose_norm(value) != target:
            return name
    return None


def _env_names(key: str) -> tuple[str, ...]:
    """Env vars that feed ``key`` (``api_key`` also honours ``OPENROUTER_API_KEY``)."""
    if key.split(".")[0] == _API_KEY_SEGMENT:
        return (env_var_name(key), "OPENROUTER_API_KEY")
    return (env_var_name(key),)


def _split_key(key: str) -> list[str]:
    parts = key.split(".")
    if not key or any(not part for part in parts):
        raise ConfigError(
            f"Invalid key {key!r}.",
            hint="Use a dotted key with no empty segments, e.g. `generation.min_words`.",
        )
    return parts


def _toml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _toml_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_toml_safe(v) for v in value]
    return value


def _loose_norm(value: Any) -> str:
    text = str(value).strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in ("'", '"'):
        text = text[1:-1]
    return text.lower()


def _parse_dotenv(path: Path) -> Iterator[tuple[str, str]]:
    """Minimal ``KEY=value`` parser (strips matching quotes, skips blanks and ``#`` comments)."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name = name.strip()
        if not name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        yield name, value
