"""`models`, `doctor`, `config`, and world-state renderers."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Any

import click
import httpx
import typer
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from nextchap import config_file, config_schema
from nextchap.cli.prompt_value import parse_value, render_value
from nextchap.cli.settings_browser import run_settings_browser
from nextchap.config import (
    KEYRING_SERVICE,
    PROJECT_CONFIG,
    Settings,
    resolve_api_key,
    user_config_path,
)
from nextchap.config_schema import SettingSpec
from nextchap.errors import ConfigError
from nextchap.memory.world_state import WorldState
from nextchap.storage.atomic import atomic_write_text

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output on stdout.")]

EXAMPLES_MODELS = "Examples:\n  nextchap models\n  nextchap models --refresh"
EXAMPLES_DOCTOR = "Examples:\n  nextchap doctor\n  nextchap doctor --json"
EXAMPLES_CONFIG = (
    "Examples:\n"
    "  nextchap config                          browse settings interactively\n"
    "  nextchap config --section models         open the browser on one section\n"
    "  nextchap config list --all\n"
    "  nextchap config get generation.min_words\n"
    "  nextchap config set defaults.chapters 3\n"
    "  nextchap config unset defaults.chapters --restore\n"
    "  nextchap config keys --json\n"
    "  nextchap config set-key"
)


def _app_ctx(ctx: typer.Context) -> Any:
    return ctx.find_root().obj


def _mask(key: str | None) -> str:
    if not key:
        return "(not set)"
    return f"{key[:6]}…{key[-4:]}" if len(key) > 12 else "****"


def _per_million(price: float) -> str:
    return f"${price * 1e6:.3f}"


# ── renderers ─────────────────────────────────────────────────────────────────


def world_tree(state: WorldState) -> Tree:
    tree = Tree(Text.assemble(("World state", "bold"), f" · after chapter {state.chapter_count}"))
    meta = tree.add("[bold]Story")
    for label, value in (
        ("Arc", state.current_arc),
        ("Tone", state.story_tone),
        ("POV", state.narrative_pov),
        ("Last hook", state.last_chapter_hook),
    ):
        if value:
            meta.add(f"[dim]{label}:[/] {value}")
    chars = tree.add(f"[bold]Characters[/] ({len(state.characters)})")
    for c in state.characters[:25]:
        bits = " · ".join(v for v in (c.role, c.status, c.location, c.power_level) if v)
        chars.add(f"[cyan]{c.name}[/] [dim]{c.id}[/]" + (f" — {bits}" if bits else ""))
    if len(state.characters) > 25:
        chars.add(f"[dim]… {len(state.characters) - 25} more[/]")
    resolved = len(state.threads) - len(state.active_threads)
    threads = tree.add(
        f"[bold]Active threads[/] ({len(state.active_threads)}, {resolved} resolved)"
    )
    for t in state.active_threads:
        origin = f" [dim](ch {t.introduced_in})[/]" if t.introduced_in else ""
        threads.add(f"[magenta]{t.id}[/] {t.description}{origin}")
    for label, entries in (("Factions", state.factions), ("Locations", state.locations)):
        if entries:
            node = tree.add(f"[bold]{label}[/] ({len(entries)})")
            for e in entries[:15]:
                node.add(f"{e.name}" + (f" — [dim]{e.description}[/]" if e.description else ""))
    for label, items in (("Key items", state.key_items), ("World rules", state.world_rules)):
        if items:
            node = tree.add(f"[bold]{label}[/] ({len(items)})")
            for i in items[:15]:
                node.add(i)
    return tree


def world_diff(a: WorldState, b: WorldState, ca: int, cb: int) -> Panel:
    lines: list[Text] = []
    old_chars = {c.id: c for c in a.characters}
    for c in b.characters:
        if c.id not in old_chars:
            lines.append(Text(f"+ character {c.name} ({c.role or 'new'})", style="green"))
            continue
        prev = old_chars[c.id]
        for field in (
            "role",
            "status",
            "location",
            "power_level",
            "goals",
            "relationships",
            "notes",
        ):
            ov, nv = getattr(prev, field), getattr(c, field)
            if ov != nv:
                lines.append(Text(f"~ {c.name}.{field}: {ov or '∅'} → {nv or '∅'}", style="yellow"))
    old_threads = {t.id: t for t in a.threads}
    for t in b.threads:
        if t.id not in old_threads:
            lines.append(Text(f"+ thread {t.id}: {t.description}", style="green"))
        elif old_threads[t.id].status != t.status:
            lines.append(Text(f"✓ resolved {t.id}: {t.description}", style="blue"))
    for label, old, new in (
        ("rule", a.world_rules, b.world_rules),
        ("item", a.key_items, b.key_items),
    ):
        for x in new:
            if x not in old:
                lines.append(Text(f"+ {label}: {x}", style="green"))
    for label, olde, newe in (
        ("location", a.locations, b.locations),
        ("faction", a.factions, b.factions),
    ):
        before = {e.name: e.description for e in olde}
        for e in newe:
            if e.name not in before:
                lines.append(Text(f"+ {label} {e.name}", style="green"))
            elif before[e.name] != e.description:
                lines.append(Text(f"~ {label} {e.name}: {e.description}", style="yellow"))
    for field in ("current_arc", "story_tone", "narrative_pov", "last_chapter_hook"):
        if getattr(a, field) != getattr(b, field):
            lines.append(Text(f"~ {field}: {getattr(b, field)}", style="yellow"))
    body = Text("\n").join(lines) if lines else Text("No changes.", style="dim")
    return Panel(body, title=f"World state: chapter {ca} → {cb}", border_style="cyan", expand=False)


# ── commands ──────────────────────────────────────────────────────────────────


def models_cmd(
    ctx: typer.Context,
    refresh: Annotated[
        bool, typer.Option("--refresh", help="Re-fetch the OpenRouter model list.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Show the model used for each role, with live prices and availability."""
    app_ctx = _app_ctx(ctx)
    reg = app_ctx.registry(refresh=refresh)
    rows = []
    for role, rc in app_ctx.settings.models.roles().items():
        info = reg.get(rc.model)
        rows.append(
            {
                "role": role,
                "model": rc.model,
                "reasoning": rc.reasoning,
                "fallbacks": rc.fallbacks,
                "available": (rc.model in reg.models) if reg.live else None,
                "input_per_m": info.prompt_price * 1e6 if info else None,
                "output_per_m": info.completion_price * 1e6 if info else None,
                "context": info.context_length if info else None,
            }
        )
    if as_json:
        app_ctx.out.print_json(json.dumps({"live": reg.live, "roles": rows}))
        return
    t = Table(
        title="Models by role" + ("" if reg.live else " (offline: built-in prices)"),
        title_justify="left",
        show_edge=False,
    )
    for col in (
        "Role",
        "Model",
        "Reasoning",
        "In $/M",
        "Out $/M",
        "Context",
        "Status",
        "Fallbacks",
    ):
        t.add_column(col)
    for r in rows:
        status = (
            "[green]listed[/]"
            if r["available"]
            else "[red]not listed[/]"
            if r["available"] is False
            else "[dim]?[/]"
        )
        t.add_row(
            r["role"],
            r["model"],
            str(r["reasoning"]),
            f"{r['input_per_m']:.3f}" if r["input_per_m"] is not None else "?",
            f"{r['output_per_m']:.3f}" if r["output_per_m"] is not None else "?",
            f"{r['context']:,}" if r["context"] else "?",
            status,
            ", ".join(r["fallbacks"]),
        )
    app_ctx.out.print(t)
    app_ctx.err.print('[dim]Change a role in nextchap.toml, e.g. [models.writer] model = "…"[/]')


def _hf_cached(model: str) -> bool:
    hub = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model.replace('/', '--')}"
    return hub.exists()


def _completion_file() -> Path | None:
    """Best-effort lookup of an installed shell completion script."""
    home = Path.home()
    candidates = (
        home / ".zfunc" / "_nextchap",
        home / ".zsh" / "completions" / "_nextchap",
        home / ".oh-my-zsh" / "completions" / "_nextchap",
        home / ".local" / "share" / "bash-completion" / "completions" / "nextchap",
        home / ".config" / "fish" / "completions" / "nextchap.fish",
    )
    return next((path for path in candidates if path.exists()), None)


def doctor_cmd(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """Check Python, API key, credits, models, embeddings and storage."""
    app_ctx = _app_ctx(ctx)
    s: Settings = app_ctx.settings
    checks: list[dict[str, str]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    py = sys.version_info
    add("Python", "ok" if py >= (3, 12) else "fail", f"{py.major}.{py.minor}.{py.micro}")

    root = s.data_root
    try:
        root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(root).free / 1e9
        add("Data directory", "ok" if free > 2 else "warn", f"{root} ({free:.1f} GB free)")
    except OSError as exc:
        add("Data directory", "fail", f"{root}: {exc}")

    key = resolve_api_key(s)
    if not key:
        add("API key", "fail", "Set OPENROUTER_API_KEY or run `nextchap config set-key`.")
    else:
        try:
            resp = httpx.get(
                f"{s.network.base_url}/key",
                headers={"Authorization": f"Bearer {key}"},
                timeout=s.network.connect_timeout,
            )
            if resp.status_code == 200:
                data = resp.json().get("data", {})
                remaining = data.get("limit_remaining")
                usage = data.get("usage")
                detail = f"{_mask(key)} · used ${float(usage or 0):.2f}"
                if remaining is not None:
                    detail += f" · ${float(remaining):.2f} left on key limit"
                add("API key", "ok", detail)
            else:
                add("API key", "fail", f"{_mask(key)} rejected (HTTP {resp.status_code})")
        except httpx.HTTPError as exc:
            add("API key", "warn", f"{_mask(key)} (could not reach OpenRouter: {exc})")

    reg = app_ctx.registry(refresh=True)
    if not reg.live:
        add("Model list", "warn", "OpenRouter /models unreachable; using built-in model data.")
    for role, rc in s.models.roles().items():
        if reg.live:
            ok = rc.model in reg.models
            add(
                f"Model: {role}",
                "ok" if ok else "fail",
                rc.model if ok else f"{rc.model} is not listed on OpenRouter",
            )

    r = s.retrieval
    if r.embed_backend == "local":
        has = importlib.util.find_spec("sentence_transformers") is not None
        if not has:
            add(
                "Embeddings",
                "fail",
                "Install the extra: uv sync --extra local-embed (or use remote)",
            )
        else:
            cached = _hf_cached(r.embed_model)
            add(
                "Embeddings",
                "ok",
                f"{r.embed_model} " + ("(cached)" if cached else "(downloads on first use)"),
            )
    else:
        add(
            "Embeddings",
            "ok" if key or r.embed_backend == "hash" else "fail",
            f"{r.embed_backend}: {r.remote_embed_model if r.embed_backend == 'remote' else 'hash'}",
        )
    if r.rerank:
        has = importlib.util.find_spec("transformers") is not None
        add(
            "Reranker", "ok" if has else "fail", r.rerank_model if has else "uv sync --extra rerank"
        )

    add(
        "epubcheck",
        "ok" if shutil.which("epubcheck") else "info",
        "available" if shutil.which("epubcheck") else "optional; enables `export --check`",
    )

    shadowing = [
        name
        for sp in config_schema.DESCRIPTORS
        if sp.key != "api_key"  # reported by the API key row; never stored in a config file
        and (name := config_file.env_var_name(sp.key)) in os.environ
    ]
    if shadowing:
        shown = ", ".join(shadowing[:3])
        if len(shadowing) > 3:
            shown += f" +{len(shadowing) - 3} more"
        add("settings", "warn", f"{shown} override the config files")
    else:
        add("settings", "ok", "no environment variables override the config files")
    completion = _completion_file()
    add(
        "Shell completion",
        "ok" if completion else "info",
        str(completion) if completion else "not installed; run `nextchap --install-completion`",
    )
    if as_json:
        app_ctx.out.print_json(json.dumps(checks))
    else:
        t = Table(show_edge=False)
        t.add_column("Check")
        t.add_column("Status")
        t.add_column("Detail")
        style = {
            "ok": "[green]ok[/]",
            "warn": "[yellow]warn[/]",
            "fail": "[red]fail[/]",
            "info": "[dim]info[/]",
        }
        for c in checks:
            t.add_row(c["check"], style[c["status"]], c["detail"])
        app_ctx.out.print(t)
    if any(c["status"] == "fail" for c in checks):
        raise typer.Exit(1)


config_app = typer.Typer(help="Show, browse and change settings.", epilog=EXAMPLES_CONFIG)


@config_app.callback(invoke_without_command=True)
def config_main(
    ctx: typer.Context,
    section: Annotated[
        str | None,
        typer.Option("--section", help="Open the browser on one section (models, defaults, …)."),
    ] = None,
    project: Annotated[
        bool | None,
        typer.Option(
            "--project",
            help="Pin writes to ./nextchap.toml; without it the browser asks once.",
        ),
    ] = None,
) -> None:
    """Browse and edit every setting interactively (non-interactive: print changed ones)."""
    if ctx.invoked_subcommand is not None:
        return
    run_settings_browser(_app_ctx(ctx), section=section, project=project)


@config_app.command("show", epilog="Example:\n  nextchap config show --json")
def config_show(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """Print the resolved settings (API key masked)."""
    app_ctx = _app_ctx(ctx)
    data = app_ctx.settings.model_dump(mode="json")
    data["api_key"] = _mask(resolve_api_key(app_ctx.settings))
    data["data_root"] = str(app_ctx.settings.data_root)
    if as_json:
        app_ctx.out.print_json(json.dumps(data))
        return
    import tomli_w
    from rich.syntax import Syntax

    app_ctx.out.print(Syntax(tomli_w.dumps(_toml_safe(data)), "toml"))


@config_app.command("path", epilog="Example:\n  nextchap config path")
def config_path(ctx: typer.Context) -> None:
    """Show where settings are read from."""
    app_ctx = _app_ctx(ctx)
    for label, p in (("project", Path.cwd() / PROJECT_CONFIG), ("user", user_config_path())):
        app_ctx.out.print(
            f"{label:8} {p} {'[green](exists)[/]' if p.exists() else '[dim](missing)[/]'}"
        )
    app_ctx.out.print(f"{'data':8} {app_ctx.settings.data_root}")


def _toml_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _toml_safe(v) for k, v in value.items() if v is not None}
    if isinstance(value, list):
        return [_toml_safe(v) for v in value]
    return value


def _spec_or_error(key: str) -> SettingSpec:
    """The descriptor for ``key``, or a :class:`ConfigError` with a discovery hint."""
    try:
        return config_schema.spec(key)
    except KeyError:
        raise ConfigError(
            f"unknown setting: {key}",
            hint="Run `nextchap config keys` to list every setting.",
        ) from None


def _setting_value(settings: Settings, key: str) -> Any:
    node: Any = settings
    for part in key.split("."):
        node = getattr(node, part)
    return node


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)


def _record(sp: SettingSpec, settings: Settings) -> dict[str, Any]:
    """The ``get``/``list`` record for one setting; secrets never carry a value."""
    value = _setting_value(settings, sp.key)
    return {
        "key": sp.key,
        "kind": sp.kind,
        "value": None if sp.secret else _jsonable(value),
        "default": _jsonable(sp.default),
        "source": config_file.source_of(sp.key),
        "help": sp.help,
        "section": sp.section,
        "changed": value != sp.default,
    }


def _value_text(sp: SettingSpec, value: Any) -> str:
    if sp.secret:
        return "••••"
    return render_value(sp.kind, value) or "(unset)"


@config_app.command("get", epilog="Example:\n  nextchap config get generation.min_words")
def config_get(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted key, e.g. generation.min_words")],
    as_json: JsonOpt = False,
) -> None:
    """Show one setting: effective value, declared default, and which layer won."""
    app_ctx = _app_ctx(ctx)
    sp = _spec_or_error(key)
    record = _record(sp, app_ctx.settings)
    if as_json:
        app_ctx.out.print_json(json.dumps(record))
        return
    app_ctx.out.print(
        f"{sp.key} = {_value_text(sp, _setting_value(app_ctx.settings, key))}"
        f" ({record['source']}, default {_value_text(sp, sp.default)})"
    )
    app_ctx.out.print(f"[dim]{sp.help}[/]")


@config_app.command(
    "list",
    epilog="Examples:\n  nextchap config list --all\n  nextchap config list --section models --json",
)
def config_list(
    ctx: typer.Context,
    all_settings: Annotated[
        bool, typer.Option("--all", help="List every editable setting, not just changed ones.")
    ] = False,
    section: Annotated[str | None, typer.Option("--section", help="Only this section.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """List settings that differ from their defaults (``--all``: every editable one)."""
    app_ctx = _app_ctx(ctx)
    if section is not None and section not in config_schema.SECTION_TITLES:
        raise ConfigError(
            f"unknown section: {section}",
            hint="Sections: " + ", ".join(config_schema.SECTION_TITLES),
        )
    # data_dir is provided per invocation (--data-dir / NEXTCHAP_HOME), not edited in a
    # layer file; its effective value is still visible via `config get data_dir`.
    specs = [s for s in config_schema.DESCRIPTORS if s.editable and s.key != "data_dir"]
    if section is not None:
        specs = [s for s in specs if s.section == section]
    if not all_settings:
        specs = [s for s in specs if _setting_value(app_ctx.settings, s.key) != s.default]
    records = [_record(s, app_ctx.settings) for s in specs]
    if as_json:
        app_ctx.out.print_json(json.dumps(records))
        return
    if not records:
        app_ctx.out.print("Nothing changed from the defaults.")
        app_ctx.out.print("[dim]Hint: nextchap config list --all shows every setting.[/]")
        return
    table = Table(show_edge=False, title="Changed settings" if not all_settings else None)
    table.add_column("Key")
    table.add_column("Value")
    table.add_column("Source")
    if all_settings:
        table.add_column("Default", style="dim")
    for sp, record in zip(specs, records, strict=False):
        row = [sp.key, _value_text(sp, _setting_value(app_ctx.settings, sp.key)), record["source"]]
        if all_settings:
            row.append(_value_text(sp, sp.default))
        table.add_row(*row)
    app_ctx.out.print(table)


@config_app.command(
    "keys", epilog="Examples:\n  nextchap config keys\n  nextchap config keys --json"
)
def config_keys(
    ctx: typer.Context,
    section: Annotated[str | None, typer.Option("--section", help="Only this section.")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Dump the setting registry: type, default, choices and help for every key."""
    app_ctx = _app_ctx(ctx)
    if section is not None and section not in config_schema.SECTION_TITLES:
        raise ConfigError(
            f"unknown section: {section}",
            hint="Sections: " + ", ".join(config_schema.SECTION_TITLES),
        )
    specs = [s for s in config_schema.DESCRIPTORS if section is None or s.section == section]
    if as_json:
        rows = [
            {
                "key": s.key,
                "kind": s.kind,
                "default": _jsonable(s.default),
                "choices": list(s.choices),
                "minimum": s.minimum,
                "maximum": s.maximum,
                "optional": s.optional,
                "editable": s.editable,
                "secret": s.secret,
                "section": s.section,
                "help": s.help,
            }
            for s in specs
        ]
        app_ctx.out.print_json(json.dumps(rows))
        return

    def type_cell(s: SettingSpec) -> str:
        parts: list[str] = [s.kind]
        if s.choices:
            parts.append("one of: " + "|".join(s.choices))
        if s.minimum is not None or s.maximum is not None:
            lo = "" if s.minimum is None else f"{_format_number(s.minimum)}≤"
            hi = "" if s.maximum is None else f"≤{_format_number(s.maximum)}"
            parts.append(f"({lo}x{hi})")
        if s.optional:
            parts.append("optional")
        if not s.editable:
            parts.append("read-only")
        return " ".join(parts)

    table = Table(show_edge=False)
    table.add_column("Key")
    table.add_column("Type")
    table.add_column("Default")
    table.add_column("Help")
    for s in specs:
        default = "••••" if s.secret else render_value(s.kind, s.default) or "(unset)"
        table.add_row(s.key, type_cell(s), default, s.help)
    app_ctx.out.print(table)


def _format_number(value: float) -> str:
    return str(int(value)) if float(value).is_integer() else str(value)


@config_app.command("set", epilog="Example:\n  nextchap config set defaults.chapters 3 --project")
def config_set(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted key, e.g. models.writer.model")],
    value: Annotated[str, typer.Argument(help="Value (see the type in `nextchap config keys`)")],
    project: Annotated[
        bool, typer.Option("--project", help="Write ./nextchap.toml instead of the user config.")
    ] = False,
) -> None:
    """Set a value in the user (or project) config file."""
    app_ctx = _app_ctx(ctx)
    sp = _spec_or_error(key)
    try:
        parsed = parse_value(sp, value)
    except ValueError as exc:
        raise ConfigError(
            f"Invalid value for {key}: {exc}",
            hint=f"Check the type and allowed values for `{key}`: run `nextchap config keys`.",
        ) from exc
    path = config_file.layer_path(project)
    data = config_file.load_layer(path)
    merged = config_file.merge_value(data, key, parsed)  # refuses api_key
    config_file.validate_layer(merged, key)
    backup = config_file.write_layer(path, merged)
    app_ctx.err.print(f"[green]✓[/] {key} = {parsed!r} in {path}")
    if backup is not None:
        app_ctx.err.print(
            f"[dim]backup: {backup} · `nextchap config unset {key} --restore` undoes it[/]"
        )
    shadow = config_file.env_shadow(key, parsed)
    if shadow is not None:
        app_ctx.err.print(f"[yellow]![/] written, but {shadow} still wins")


@config_app.command(
    "unset",
    epilog=(
        "Examples:\n"
        "  nextchap config unset defaults.chapters\n"
        "  nextchap config unset defaults.chapters --restore"
    ),
)
def config_unset(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted key, e.g. models.writer.model")],
    project: Annotated[
        bool, typer.Option("--project", help="Edit ./nextchap.toml instead of the user config.")
    ] = False,
    restore: Annotated[
        bool,
        typer.Option("--restore", help="Restore the .bak from the last write to this layer."),
    ] = False,
) -> None:
    """Remove a key from the user (or project) config file."""
    app_ctx = _app_ctx(ctx)
    path = config_file.layer_path(project)
    if restore:
        if not config_file.restore_layer(path):
            raise ConfigError(
                f"No backup at {config_file.backup_path(path)}.",
                hint="--restore only undoes the last `config set`/`unset` write in this layer.",
            )
        app_ctx.err.print(f"[green]✓[/] Restored {path} from {config_file.backup_path(path)}.")
        return
    data = config_file.load_layer(path)
    new_data, found = config_file.drop_value(data, key)
    if not found:
        app_ctx.out.print(f"{key} is not set in {path}")
        return
    config_file.validate_layer(new_data, key)
    backup = config_file.write_layer(path, new_data)
    app_ctx.err.print(f"[green]✓[/] Removed {key} from {path}.")
    if backup is not None:
        app_ctx.err.print(
            f"[dim]backup: {backup} · `nextchap config unset {key} --restore` undoes it[/]"
        )


@config_app.command("edit", epilog="Example:\n  nextchap config edit --project")
def config_edit(
    ctx: typer.Context,
    project: Annotated[
        bool, typer.Option("--project", help="Edit ./nextchap.toml instead of the user config.")
    ] = False,
) -> None:
    """Open the config file in $EDITOR; the edited TOML is validated before saving."""
    app_ctx = _app_ctx(ctx)
    path = config_file.layer_path(project)
    original = path.read_text(encoding="utf-8") if path.exists() else ""
    edited = click.edit(text=original, extension=".toml")
    if edited is None or edited == original:
        app_ctx.err.print("[dim]No changes.[/]")
        return
    try:
        data = tomllib.loads(edited)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"{path} would not be valid TOML: {exc}",
            hint="Nothing was written; fix the edit and run `nextchap config edit` again.",
        ) from exc
    if "api_key" in data:
        raise ConfigError(
            "Refusing to store the API key in a config file.",
            hint="Use `nextchap config set-key` or the OPENROUTER_API_KEY env var.",
        )
    config_file.validate_layer(data, str(path))
    atomic_write_text(path, edited)
    app_ctx.err.print(f"[green]✓[/] Saved {path}.")


@config_app.command("set-key", epilog="Example:\n  nextchap config set-key")
def config_set_key(ctx: typer.Context) -> None:
    """Store the OpenRouter API key in the OS keyring."""
    app_ctx = _app_ctx(ctx)
    try:
        import keyring
    except ImportError as exc:
        raise ConfigError(
            "The keyring extra is not installed.",
            hint="uv sync --extra keyring, or export OPENROUTER_API_KEY / put it in a .env file.",
        ) from exc
    secret = Prompt.ask("OpenRouter API key", password=True, console=app_ctx.err).strip()
    if not secret.startswith("sk-"):
        raise ConfigError("That does not look like an OpenRouter key (expected sk-or-…).")
    keyring.set_password(KEYRING_SERVICE, "default", secret)
    app_ctx.err.print(f"[green]✓[/] Stored {_mask(secret)} in the OS keyring.")


def register(app: typer.Typer) -> None:
    app.command("models", epilog=EXAMPLES_MODELS)(models_cmd)
    app.command("doctor", epilog=EXAMPLES_DOCTOR)(doctor_cmd)
    app.add_typer(config_app, name="config")
