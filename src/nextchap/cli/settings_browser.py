"""Interactive settings browser: browse and edit every setting from a keyed picker.

A shell over the config-file layer (the same idiom as ``cli/menu.py``: keys, not
arrow keys). It renders the descriptor registry with live values and writes
through ``config_file`` only — nothing here touches the network or the
generation pipeline. Non-interactive sessions get a compact changed-settings
table instead of prompts, so scripts and CI never block on a pipe.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.markup import escape
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from nextchap import config as config_mod
from nextchap import config_file
from nextchap.cli.context import AppContext
from nextchap.cli.prompt_value import ask_value, render_value
from nextchap.cli.ui import error_panel
from nextchap.config import Settings
from nextchap.config_schema import DESCRIPTORS, SECTION_TITLES, SettingSpec
from nextchap.errors import NextChapError
from nextchap.models import KNOWN_MODELS

__all__ = ["model_ids", "run_settings_browser"]

_VALUE_WIDTH = 28
_PICKER_CAP = 15
_SOURCE_STYLES: dict[str, str] = {
    "default": "dim",
    "user": "cyan",
    "project": "blue",
    "dotenv": "yellow",
    "env": "bold yellow",  # env wins over files, so it stands out
}


def effective_settings(app_ctx: AppContext) -> Settings:
    """Live settings for this session: the app's data dir overrides the config files."""
    overrides = {"data_dir": app_ctx.data_dir} if app_ctx.data_dir else {}
    return Settings(**overrides)  # type: ignore[arg-type]


def model_ids(settings: Settings) -> list[str]:
    """Model ids for the picker: the four configured role models first, then the rest.

    Reads the offline ``models.json`` cache when present (tolerating a corrupt or
    missing file) and unions it with ``KNOWN_MODELS``. Never contacts the network.
    """
    cached: set[str] = set()
    cache = settings.cache_dir / "models.json"
    try:
        raw: Any = json.loads(cache.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raw = None
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        cached = {
            item["id"]
            for item in raw["data"]
            if isinstance(item, dict) and isinstance(item.get("id"), str)
        }
    ids: list[str] = []
    for role in settings.models.roles().values():
        if role.model not in ids:
            ids.append(role.model)
    for model_id in sorted(cached | set(KNOWN_MODELS)):
        if model_id not in ids:
            ids.append(model_id)
    return ids[:_PICKER_CAP]


def run_settings_browser(
    app_ctx: AppContext,
    *,
    section: str | None = None,
    project: bool | None = None,
) -> None:
    """Show the settings table and loop on the ``Setting`` prompt until ``q``.

    ``section`` opens filtered to one section (sticky for the session);
    ``project`` pins writes to the project layer (``True``/``False``); ``None``
    asks once per session and remembers the answer. Never raises for bad values.
    """
    settings = effective_settings(app_ctx)
    if not app_ctx.interactive:
        _print_summary(app_ctx, settings, section)
        app_ctx.err.print("[dim]Non-interactive: use nextchap config list | get | set.[/]")
        return

    if section is not None and (section not in SECTION_TITLES or section == "security"):
        app_ctx.err.print(f"[yellow]! unknown section {escape(section)} — showing all settings[/]")
        section = None

    filter_text = ""
    done_key: str | None = None
    layer_choice: bool | None = None

    while True:
        rows = _rows(section, filter_text)
        _render(app_ctx, settings, rows, done_key=done_key, filter_text=filter_text)
        done_key = None
        raw = Prompt.ask("Setting", default="", console=app_ctx.err, show_default=False).strip()
        if raw == "q":
            return
        if raw == "?":
            _print_help(app_ctx.err)
            continue
        if raw == "w":
            _print_where(app_ctx, settings)
            continue
        if raw.startswith("/"):
            filter_text = raw[1:].strip()
            continue
        if raw.isdigit():
            index = int(raw)
            if not 1 <= index <= len(rows):
                app_ctx.err.print(f"[red]no such row: {index}[/]")
                continue
            spec = rows[index - 1]
            committed, value = ask_value(
                app_ctx.err, spec, _value_of(settings, spec.key), models=model_ids(settings)
            )
            if not committed:
                continue
            if project is not None:
                use_project = project
            elif layer_choice is None:
                use_project = _ask_layer(app_ctx.err)
                layer_choice = use_project
            else:
                use_project = layer_choice
            done_key = _write(app_ctx, spec, value, use_project)
            if done_key is not None:
                settings = effective_settings(app_ctx)
            continue
        app_ctx.err.print("[red]Enter a row number, /filter, w, ? or q.[/]")


# --- table ------------------------------------------------------------------


def _rows(section: str | None, filter_text: str) -> list[SettingSpec]:
    """Visible descriptors in registry order, minus the read-only security section."""
    specs = [s for s in DESCRIPTORS if s.section != "security"]
    if section is not None:
        specs = [s for s in specs if s.section == section]
    if filter_text:
        needle = filter_text.lower()
        specs = [s for s in specs if needle in s.key.lower() or needle in s.help.lower()]
    return specs


def _value_of(settings: Settings, key: str) -> Any:
    node: Any = settings
    for part in key.split("."):
        node = getattr(node, part)
    return node


def _value_cell(spec: SettingSpec, value: Any) -> str:
    if spec.secret:
        return "••••"
    text = render_value(spec.kind, value)
    if len(text) > _VALUE_WIDTH:
        text = text[: _VALUE_WIDTH - 1] + "…"
    return escape(text)


def _source_cell(key: str) -> str:
    source = config_file.source_of(key)
    return f"[{_SOURCE_STYLES[source]}]{source}[/]"


def _render(
    app_ctx: AppContext,
    settings: Settings,
    rows: list[SettingSpec],
    *,
    done_key: str | None,
    filter_text: str,
) -> None:
    console = app_ctx.err
    console.print(
        f"[bold]NextChap settings[/] · {len(rows)} keys · {config_mod.user_config_path()}"
    )
    if filter_text:
        console.print(f"[dim]filter: {escape(filter_text)}[/]")
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("Source")
    seen_section: str | None = None
    for number, spec in enumerate(rows, 1):
        if spec.section != seen_section:
            seen_section = spec.section
            title = SECTION_TITLES.get(spec.section, spec.section)
            table.add_row("", Text(f"── {title}", style="bold"), "", "")
        value = _value_of(settings, spec.key)
        key_cell = Text(spec.key)
        if spec.key == done_key:
            key_cell.append(" ✓", style="green")
        elif value != spec.default:
            key_cell.append(" *", style="yellow")
        table.add_row(str(number), key_cell, _value_cell(spec, value), _source_cell(spec.key))
    console.print(table)
    console.print(
        f"[dim]1-{len(rows)} · /filter · \\[w]here nextchap.toml is · ? help · \\[q]uit[/]"
    )


# --- prompt-loop helpers ----------------------------------------------------


def _print_help(console: Console) -> None:
    console.print(
        "[dim]number — edit that row · /text — filter by key or help · / — clear filter\n"
        "w — where settings are read from · q — quit\n"
        "Source: default = built-in · user = user config.toml · project = ./nextchap.toml\n"
        "env/dotenv = environment variable (wins over files) · blank value = unset[/]"
    )


def _print_where(app_ctx: AppContext, settings: Settings) -> None:
    """The same lines as ``nextchap config path``."""
    for label, path in (
        ("project", Path.cwd() / config_mod.PROJECT_CONFIG),
        ("user", config_mod.user_config_path()),
    ):
        state = "[green](exists)[/]" if path.exists() else "[dim](missing)[/]"
        app_ctx.err.print(f"{label:8} {path} {state}")
    app_ctx.err.print(f"{'data':8} {settings.data_root}")


def _print_summary(app_ctx: AppContext, settings: Settings, section: str | None) -> None:
    """Non-interactive view: only the settings that differ from their defaults."""
    rows = _rows(section, "")
    changed = [s for s in rows if _value_of(settings, s.key) != s.default]
    app_ctx.out.print(f"[bold]NextChap settings[/] · {len(changed)} changed from default")
    table = Table(show_header=True, box=None, pad_edge=False)
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("Source")
    for spec in changed:
        table.add_row(
            spec.key, _value_cell(spec, _value_of(settings, spec.key)), _source_cell(spec.key)
        )
    app_ctx.out.print(table)


def _ask_layer(console: Console) -> bool:
    """Ask which layer to write to (asked once per session); ``True`` = project."""
    answer = Prompt.ask(
        escape("Write to [u]ser config / [p]roject config"),
        choices=["u", "p"],
        default="u",
        console=console,
        show_choices=False,
    )
    return answer.strip().lower() == "p"


def _write(app_ctx: AppContext, spec: SettingSpec, value: Any, use_project: bool) -> str | None:
    """Write one value through the config-file layer; the key on success, None on error."""
    try:
        path = config_file.layer_path(use_project)
        data = config_file.load_layer(path)
        merged = config_file.merge_value(data, spec.key, value)
        config_file.validate_layer(merged, spec.key)
        backup = config_file.write_layer(path, merged)
    except (NextChapError, ValueError) as exc:
        error_panel(app_ctx.err, exc)
        return None
    rendered = render_value(spec.kind, value) or "(unset)"
    app_ctx.err.print(f"✓ {spec.key} = {escape(rendered)} in {path}")
    if backup is not None:
        app_ctx.err.print(
            f"[dim]backup: {backup} · nextchap config unset {spec.key} --restore undoes it[/]"
        )
    shadow = config_file.env_shadow(spec.key, value)
    if shadow is not None:
        app_ctx.err.print(f"[yellow]! written, but {shadow} still wins[/]")
    return spec.key
