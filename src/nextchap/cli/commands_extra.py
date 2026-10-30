"""`models`, `doctor`, `config`, `migrate`, and world-state renderers."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import tomllib
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer
from pydantic import ValidationError
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from nextchap.config import (
    KEYRING_SERVICE,
    PROJECT_CONFIG,
    Settings,
    resolve_api_key,
    user_config_path,
)
from nextchap.errors import ConfigError
from nextchap.memory.world_state import WorldState
from nextchap.storage.migrate import LEGACY_BOOKS, find_legacy, migrate

JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output on stdout.")]


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
        "available" if shutil.which("epubcheck") else "optional; enables `export epub --check`",
    )
    if find_legacy(Path.cwd()):
        add("Legacy data", "warn", f"./{LEGACY_BOOKS} found; run `nextchap migrate`")

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


config_app = typer.Typer(help="Show and change settings.", no_args_is_help=True)


@config_app.command("show")
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

    app_ctx.out.print(
        Syntax(tomli_w.dumps({k: v for k, v in data.items() if v is not None}), "toml")
    )


@config_app.command("path")
def config_path(ctx: typer.Context) -> None:
    """Show where settings are read from."""
    app_ctx = _app_ctx(ctx)
    for label, p in (("project", Path.cwd() / PROJECT_CONFIG), ("user", user_config_path())):
        app_ctx.out.print(
            f"{label:8} {p} {'[green](exists)[/]' if p.exists() else '[dim](missing)[/]'}"
        )
    app_ctx.out.print(f"{'data':8} {app_ctx.settings.data_root}")


def _parse_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except ValueError:
        return raw


@config_app.command("set")
def config_set(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted key, e.g. models.writer.model")],
    value: Annotated[str, typer.Argument(help="Value (JSON for numbers/lists/booleans)")],
    project: Annotated[
        bool, typer.Option("--project", help="Write ./nextchap.toml instead of the user config.")
    ] = False,
) -> None:
    """Set a value in the user (or project) config file."""
    import tomli_w

    app_ctx = _app_ctx(ctx)
    if key.split(".")[0] == "api_key":
        raise ConfigError(
            "Refusing to store the API key in a config file.",
            hint="Use `nextchap config set-key` or the OPENROUTER_API_KEY env var.",
        )
    path = (Path.cwd() / PROJECT_CONFIG) if project else user_config_path()
    data: dict[str, Any] = tomllib.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    node = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node.setdefault(part, {})
        if not isinstance(node, dict):
            raise ConfigError(f"{key} conflicts with an existing non-table value.")
    node[parts[-1]] = _parse_value(value)
    try:
        Settings(**data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid value for {key}: {exc.errors()[0]['msg']}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tomli_w.dumps(data), encoding="utf-8")
    app_ctx.err.print(f"[green]✓[/] {key} = {node[parts[-1]]!r} in {path}")


@config_app.command("set-key")
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


def migrate_cmd(
    ctx: typer.Context,
    from_dir: Annotated[
        Path, typer.Option("--from", help="Folder containing novel_books/.")
    ] = Path(),
    move: Annotated[
        bool, typer.Option("--move", help="Delete the old folders after copying.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Move 1.x books (./novel_books) into the data directory."""
    app_ctx = _app_ctx(ctx)
    found = find_legacy(from_dir.resolve())
    if not found:
        app_ctx.err.print(f"No 1.x books found in {from_dir.resolve() / LEGACY_BOOKS}.")
        return
    app_ctx.err.print(f"Found {len(found)} book(s): {', '.join(p.name for p in found)}")
    app_ctx.err.print(f"Destination: {app_ctx.settings.data_root / 'books'}")
    if not yes and not Confirm.ask(
        "Migrate them?" + (" (old folders are deleted)" if move else ""),
        default=True,
        console=app_ctx.err,
    ):
        raise typer.Exit(1)
    results = migrate(from_dir.resolve(), app_ctx.settings.data_root, copy=not move)
    t = Table(show_edge=False)
    t.add_column("Old ID")
    t.add_column("Book ID")
    t.add_column("Result")
    for r in results:
        t.add_row(r.legacy_id, r.book_id, r.status)
    app_ctx.err.print(t)
    app_ctx.err.print(
        "World state and summaries are upgraded on first use. The retrieval index is "
        "rebuilt locally on the next run (no API cost)."
    )


def register(app: typer.Typer) -> None:
    app.command("models")(models_cmd)
    app.command("doctor")(doctor_cmd)
    app.command("migrate")(migrate_cmd)
    app.add_typer(config_app, name="config")
