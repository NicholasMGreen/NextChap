"""`nextchap` command line."""

from __future__ import annotations

import json
import logging
import signal
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import FrameType
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table
from rich.text import Text

from nextchap import __version__
from nextchap.cli import commands_extra
from nextchap.cli.context import AppContext
from nextchap.cli.ui import error_panel, money
from nextchap.config import Settings
from nextchap.errors import BookExists, IndexMismatch, NextChapError, ParseError
from nextchap.ingest import master_book
from nextchap.ingest.cleanup import NoiseFilter
from nextchap.ingest.epub import EpubMeta, read_epub
from nextchap.ingest.text import DetectionReport, parse_text_file
from nextchap.llm.cost import CostMeter, logged_spend
from nextchap.pipeline import manage
from nextchap.pipeline.book import Book, BookMeta
from nextchap.pipeline.engine import Engine
from nextchap.pipeline.engine import export_epub as write_book_epub
from nextchap.pipeline.estimate import EstimateRow, estimate_generation, estimate_import, total
from nextchap.retrieval.context import estimate_tokens
from nextchap.storage.lock import book_lock
from nextchap.storage.paths import all_book_ids, slugify, validate_book_id

log = logging.getLogger("nextchap")

app = typer.Typer(
    name="nextchap",
    help="Continue web novels from EPUB/TXT with long-term story memory.",
    no_args_is_help=True,
    rich_markup_mode="rich",
    pretty_exceptions_enable=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)
books_app = typer.Typer(help="List, inspect and delete books.", no_args_is_help=True)
state_app = typer.Typer(
    help="Inspect, rebuild and roll back the world state.", no_args_is_help=True
)
index_app = typer.Typer(help="Manage the retrieval index.", no_args_is_help=True)
export_app = typer.Typer(help="Export books.", no_args_is_help=True)
app.add_typer(books_app, name="books")
app.add_typer(state_app, name="state")
app.add_typer(index_app, name="index")
app.add_typer(export_app, name="export")
commands_extra.register(app)


def _ctx(ctx: typer.Context) -> AppContext:
    obj = ctx.find_root().obj
    assert isinstance(obj, AppContext)
    return obj


def complete_book_id(incomplete: str) -> list[str]:
    try:
        return [b for b in all_book_ids(Settings().data_root) if b.startswith(incomplete)]
    except Exception:
        return []


BookId = Annotated[
    str, typer.Argument(help="Book ID.", autocompletion=complete_book_id, show_default=False)
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Machine-readable output on stdout.")]


def _version(value: bool) -> None:
    if value:
        typer.echo(f"nextchap {__version__}")
        raise typer.Exit


@app.callback()
def root(
    ctx: typer.Context,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            envvar="NEXTCHAP_HOME",
            help="Data root; books go into <root>/novel_books (default: the project directory).",
        ),
    ] = None,
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="-v info, -vv debug.")
    ] = 0,
    plain: Annotated[bool, typer.Option("--plain", help="No live UI; plain log lines.")] = False,
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show the version.")
    ] = False,
) -> None:
    app_ctx = AppContext(data_dir=data_dir, verbose=verbose, plain=plain)
    ctx.obj = app_ctx
    app_ctx.setup_logging()


# ── helpers ───────────────────────────────────────────────────────────────────


def _open_book(app_ctx: AppContext, book_id: str) -> Book:
    return Book.open(app_ctx.settings, validate_book_id(book_id))


def _estimate_table(title: str, rows: list[EstimateRow]) -> Table:
    t = Table(title=title, title_justify="left", show_edge=False)
    for col, just in (
        ("Role", "left"),
        ("Model", "left"),
        ("Calls", "right"),
        ("Input tok", "right"),
        ("Cached", "right"),
        ("Output tok", "right"),
        ("Cost", "right"),
    ):
        t.add_column(col, justify=just)  # type: ignore[arg-type]
    for r in rows:
        t.add_row(
            r.role,
            r.model,
            f"{r.calls:g}",
            f"{r.prompt_tokens:,}",
            f"{r.cached_tokens:,}",
            f"{r.completion_tokens:,}",
            money(r.cost),
        )
    t.add_section()
    t.add_row("[bold]total", "", "", "", "", "", f"[bold]{money(total(rows))}")
    return t


@contextmanager
def _graceful_stop(engine: Engine, err: Console) -> Iterator[None]:
    """First Ctrl-C stops at the next safe point; the second aborts immediately."""
    previous = signal.getsignal(signal.SIGINT)

    def handler(signum: int, frame: FrameType | None) -> None:
        if engine.stop_requested:
            signal.signal(signal.SIGINT, signal.default_int_handler)
            raise KeyboardInterrupt
        engine.stop_requested = True
        err.print(
            "[yellow]![/] Stopping after the current step (progress is saved). "
            "Press Ctrl-C again to abort now."
        )

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)


def _run_with_index_guard(app_ctx: AppContext, book: Book, fn: Any) -> Any:
    try:
        return fn()
    except IndexMismatch as exc:
        app_ctx.err.print(f"[yellow]![/] {exc}")
        if not app_ctx.interactive or not Confirm.ask(
            "Rebuild the index now? (local only, no API cost)", default=True, console=app_ctx.err
        ):
            raise
        reporter = app_ctx.reporter(CostMeter())
        with reporter:
            manage.rebuild_index(book, lambda: app_ctx.store_factory(book), reporter)
        return fn()


def _detect(app_ctx: AppContext, file: Path) -> tuple[DetectionReport, EpubMeta, str]:
    noise = NoiseFilter(app_ctx.settings.cleanup.extra_noise_patterns)
    suffix = file.suffix.lower()
    if suffix == ".epub":
        report, meta = read_epub(file, noise)
        return report, meta, "epub"
    if suffix in (".txt", ".md", ".markdown"):
        text = file.read_text(encoding="utf-8", errors="replace")
        parsed = master_book.parse(text)
        if suffix != ".txt" and parsed.chapters:
            return DetectionReport(chapters=parsed.chapters), EpubMeta(title=file.stem), "md"
        return parse_text_file(file, noise), EpubMeta(title=file.stem), "txt"
    raise ParseError(
        f"Unsupported file type: {file.suffix}", hint="Use an .epub, .txt or .md file."
    )


def _detection_panel(report: DetectionReport, file: Path) -> Panel:
    chs = report.chapters
    t = Table(show_edge=False, box=None, pad_edge=False)
    t.add_column("#", justify="right", style="dim")
    t.add_column("Title")
    t.add_column("Words", justify="right")
    shown = chs if len(chs) <= 10 else [*chs[:6], None, *chs[-3:]]
    for c in shown:
        if c is None:
            t.add_row("…", "", "")
        else:
            t.add_row(str(c.number), c.title[:60], f"{c.word_count:,}")
    words = sum(c.word_count for c in chs)
    body = [
        Text.assemble((f"{len(chs)} chapters", "bold"), f" · {words:,} words · from {file.name}")
    ]
    if report.removed_noise:
        body.append(Text(f"Removed {report.removed_noise} promo/noise line(s).", style="dim"))
    from rich.console import Group

    items: list[Any] = [*body, t]
    for a in report.anomalies:
        items.append(Text(f"! {a}", style="yellow"))
    return Panel(Group(*items), title="Detected chapters", border_style="cyan", expand=False)


# ── import / generate ─────────────────────────────────────────────────────────


@app.command("import")
def import_cmd(
    ctx: typer.Context,
    file: Annotated[
        Path,
        typer.Argument(
            exists=True, dir_okay=False, readable=True, help=".epub, .txt or master_book-style .md"
        ),
    ],
    book_id: Annotated[
        str | None, typer.Option("--book-id", help="Defaults to a slug of the filename.")
    ] = None,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Skip confirmation prompts.")] = False,
    force: Annotated[
        bool, typer.Option("--force", help="Replace an existing book with this ID.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Detect chapters and estimate cost only.")
    ] = False,
    max_cost: Annotated[
        float | None, typer.Option("--max-cost", help="Stop if spend reaches this (USD).")
    ] = None,
) -> None:
    """Import a novel and build its story memory (summaries, world state, bible, index)."""
    app_ctx = _ctx(ctx)
    bid = validate_book_id(book_id or slugify(file.stem))
    book = Book.open(app_ctx.settings, bid, must_exist=False)
    if book.exists() and not force:
        raise BookExists(f"Book '{bid}' already exists.")

    report, meta, kind = _detect(app_ctx, file)
    app_ctx.err.print(_detection_panel(report, file))
    rows = estimate_import(app_ctx.settings, app_ctx.registry(), report.chapters)
    app_ctx.err.print(_estimate_table("Estimated import cost", rows))
    if dry_run:
        return
    if not yes:
        if not app_ctx.err.is_terminal:
            raise NextChapError(
                "Confirmation needed.", hint="Re-run with --yes to import non-interactively."
            )
        prompt = f"Import {len(report.chapters)} chapters as '{bid}' (about {money(total(rows))})?"
        if book.exists():
            prompt = f"Replace existing book '{bid}' and import {len(report.chapters)} chapters?"
        if not Confirm.ask(prompt, default=True, console=app_ctx.err):
            raise typer.Exit(1)
    if book.exists():
        manage.delete_book(book)

    meter = CostMeter(max_cost if max_cost is not None else app_ctx.settings.generation.max_cost)
    book.paths.root.mkdir(parents=True, exist_ok=True)
    meter.bind(book.paths.usage, bid)
    reporter = app_ctx.reporter(meter)
    bm = BookMeta(
        book_id=bid,
        source_name=file.name,
        source_kind=kind,  # type: ignore[arg-type]
        title=meta.title,
        author=meta.author,
        language=meta.language or "en",
    )
    with book_lock(book.paths.lock), reporter:
        engine = app_ctx.engine(book, meter, reporter)
        with _graceful_stop(engine, app_ctx.err):
            engine.import_chapters(report.chapters, bm, file if kind == "epub" else None)
    app_ctx.err.print(
        Panel(
            Text.assemble(
                ("Imported ", ""),
                (bid, "bold"),
                f" · {len(report.chapters)} chapters · spent ",
                (money(meter.total_cost), "bold"),
                "\n\nNext: ",
                (f"nextchap generate {bid} -n 3", "bold cyan"),
                "\nTweak voice first in: ",
                (str(book.paths.style), "dim"),
            ),
            border_style="green",
            expand=False,
        )
    )


@app.command()
def generate(
    ctx: typer.Context,
    book_id: BookId,
    chapters: Annotated[
        int, typer.Option("-n", "--chapters", min=1, help="How many chapters.")
    ] = 1,
    max_cost: Annotated[
        float | None, typer.Option("--max-cost", help="Stop before spend passes this (USD).")
    ] = None,
    review: Annotated[
        bool, typer.Option("--review", help="Approve/edit each plan and chapter.")
    ] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the estimated calls and cost only.")
    ] = False,
    no_preview: Annotated[
        bool, typer.Option("--no-preview", help="Hide the live prose preview.")
    ] = False,
    no_critic: Annotated[
        bool, typer.Option("--no-critic", help="Skip the continuity check.")
    ] = False,
    no_epub: Annotated[
        bool, typer.Option("--no-epub", help="Don't re-export the EPUB afterwards.")
    ] = False,
) -> None:
    """Plan, write and commit new chapters."""
    app_ctx = _ctx(ctx)
    s = app_ctx.settings
    if no_critic:
        s.generation.critic = False
    if no_epub:
        s.generation.export_epub = False
    book = _open_book(app_ctx, book_id)

    if dry_run:
        state = book.world.load()
        base = (
            estimate_tokens(book.style())
            + estimate_tokens(book.bible())
            + estimate_tokens(state.compact())
        )
        recent = book.master().chapters[-s.generation.recent_full_count :]
        writer_prefix = base + min(
            sum(estimate_tokens(c.content) for c in recent),
            int(s.generation.max_prompt_tokens * 0.25),
        )
        rows = estimate_generation(s, app_ctx.registry(), chapters, base, writer_prefix)
        app_ctx.err.print(_estimate_table(f"Estimated cost for {chapters} chapter(s)", rows))
        return

    if review and not app_ctx.err.is_terminal:
        raise NextChapError("--review needs an interactive terminal.")
    meter = CostMeter(max_cost if max_cost is not None else s.generation.max_cost)
    meter.bind(book.paths.usage, book.id)
    reporter = app_ctx.reporter(meter, preview=not no_preview)
    with book_lock(book.paths.lock), reporter:
        engine = app_ctx.engine(book, meter, reporter, review=review)
        with _graceful_stop(engine, app_ctx.err):
            done = _run_with_index_guard(app_ctx, book, lambda: engine.generate(chapters))

    app_ctx.err.print(f"[bold]{len(done)} chapter(s) added to {book.id}[/]")
    t = Table(show_edge=False)
    t.add_column("#", justify="right")
    t.add_column("Title")
    t.add_column("Words", justify="right")
    for c in done:
        t.add_row(str(c.number), c.title, f"{c.word_count:,}")
    app_ctx.err.print(t)
    by_role = ", ".join(f"{k} {money(v)}" for k, v in sorted(meter.by_role().items()))
    app_ctx.err.print(
        f"Spent [bold]{money(meter.total_cost)}[/] ({by_role}) · cache hit "
        f"{meter.cache_hit_rate:.0%}"
    )
    app_ctx.err.print(f"Book folder: [dim]{book.paths.root}[/]")


# ── books ─────────────────────────────────────────────────────────────────────


@books_app.command("list")
def books_list(ctx: typer.Context, as_json: JsonOpt = False) -> None:
    """List all books."""
    app_ctx = _ctx(ctx)
    rows = [
        manage.book_overview(Book.open(app_ctx.settings, b))
        for b in all_book_ids(app_ctx.settings.data_root)
    ]
    if as_json:
        app_ctx.out.print_json(json.dumps(rows))
        return
    if not rows:
        app_ctx.err.print(
            f"No books yet in {app_ctx.settings.data_root}. "
            "Start with [bold]nextchap import <file>[/]."
        )
        return
    t = Table(show_edge=False)
    for col, just in (
        ("Book ID", "left"),
        ("Chapters", "right"),
        ("Generated", "right"),
        ("Size", "right"),
        ("Modified", "left"),
    ):
        t.add_column(col, justify=just)  # type: ignore[arg-type]
    for r in rows:
        t.add_row(
            r["book_id"],
            str(r["chapters"]),
            str(r["generated"]),
            f"{r['size_mb']:.1f} MB",
            r["modified"],
        )
    app_ctx.out.print(t)


@books_app.command("info")
def books_info(ctx: typer.Context, book_id: BookId, as_json: JsonOpt = False) -> None:
    """Show details, world state and spend for one book."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    ov = manage.book_overview(book)
    spend = logged_spend(book.paths.usage)
    ov["total_spend"] = round(spend, 6)
    if as_json:
        state = book.world.load().model_dump(mode="json") if book.world.exists() else None
        app_ctx.out.print_json(json.dumps({**ov, "world_state": state}))
        return
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim")
    grid.add_column()
    for label, key in (
        ("Title", "title"),
        ("Source", "source"),
        ("Chapters", "chapters"),
        ("Generated", "generated"),
        ("Disk", "size_mb"),
        ("Modified", "modified"),
        ("Folder", "path"),
    ):
        value = ov[key]
        grid.add_row(label, f"{value} MB" if key == "size_mb" else str(value))
    grid.add_row("Total spend", money(spend))
    app_ctx.out.print(Panel(grid, title=book.id, border_style="cyan", expand=False))
    if book.world.exists():
        app_ctx.out.print(commands_extra.world_tree(book.world.load()))


@books_app.command("delete")
def books_delete(
    ctx: typer.Context,
    book_id: BookId,
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Don't ask for confirmation.")] = False,
) -> None:
    """Permanently delete a book and everything derived from it."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    ov = manage.book_overview(book)
    app_ctx.err.print(
        f"Book [bold]{book.id}[/]: {ov['chapters']} chapters ({ov['generated']} generated), "
        f"{ov['size_mb']:.1f} MB at {ov['path']}"
    )
    if not yes and not Confirm.ask("Delete it permanently?", default=False, console=app_ctx.err):
        raise typer.Exit(1)
    with book_lock(book.paths.lock):
        pass
    manage.delete_book(book)
    app_ctx.err.print(f"[green]✓[/] Deleted {book.id}.")


@app.command()
def clean(
    ctx: typer.Context,
    book_id: BookId,
    no_backup: Annotated[
        bool, typer.Option("--no-backup", help="Don't write master_book.md.bak.")
    ] = False,
) -> None:
    """Remove promo lines and duplicate chapter headers from master_book.md."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    with book_lock(book.paths.lock):
        report = manage.clean_book(
            book, backup=not no_backup, extra_patterns=app_ctx.settings.cleanup.extra_noise_patterns
        )
    if not report.changed:
        app_ctx.err.print("Nothing to clean.")
        return
    app_ctx.err.print(
        f"[green]✓[/] Removed {report.removed_noise} noise line(s) and "
        f"{report.removed_duplicate_headers} duplicate header(s). Changed chapters are "
        "re-summarized on the next run."
    )


# ── state / index / export ────────────────────────────────────────────────────


@state_app.command("show")
def state_show(
    ctx: typer.Context,
    book_id: BookId,
    chapter: Annotated[
        int | None, typer.Option("--chapter", help="Show the snapshot after this chapter.")
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Show the world state (characters, threads, rules…)."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    state = book.world.load_snapshot(chapter) if chapter else book.world.load()
    if as_json:
        app_ctx.out.print_json(state.model_dump_json())
        return
    app_ctx.out.print(commands_extra.world_tree(state))


@state_app.command("diff")
def state_diff(
    ctx: typer.Context,
    book_id: BookId,
    from_ch: Annotated[
        int | None, typer.Option("--from", help="Older snapshot (default: second newest).")
    ] = None,
    to_ch: Annotated[
        int | None, typer.Option("--to", help="Newer snapshot (default: newest).")
    ] = None,
) -> None:
    """Show what changed in the world state between two chapters."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    snaps = book.world.snapshots()
    if len(snaps) < 2 and (from_ch is None or to_ch is None):
        raise NextChapError("Need at least two world-state snapshots to diff.")
    b = to_ch or snaps[-1]
    a = from_ch or max(n for n in snaps if n < b)
    app_ctx.out.print(
        commands_extra.world_diff(book.world.load_snapshot(a), book.world.load_snapshot(b), a, b)
    )


@state_app.command("rebuild")
def state_rebuild(
    ctx: typer.Context,
    book_id: BookId,
    no_renumber: Annotated[
        bool, typer.Option("--no-renumber", help="Keep chapter numbers.")
    ] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Report without writing.")] = False,
    no_backup: Annotated[bool, typer.Option("--no-backup")] = False,
) -> None:
    """Re-derive bookkeeping from master_book.md (fixes chapter-number drift)."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    with book_lock(book.paths.lock):
        r = manage.rebuild_state(
            book, renumber=not no_renumber, dry_run=dry_run, backup=not no_backup
        )
    verb = "Would renumber" if dry_run else "Renumbered"
    app_ctx.err.print(
        f"{r.chapters} chapters · {verb} {r.renumbered}"
        + (" · index reset (rebuilt on next run)" if r.index_reset and not dry_run else "")
    )


@state_app.command("rollback")
def state_rollback(
    ctx: typer.Context,
    book_id: BookId,
    to: Annotated[int, typer.Option("--to", help="Restore the state saved after this chapter.")],
    truncate: Annotated[
        bool, typer.Option("--truncate", help="Also delete chapters after it.")
    ] = False,
    yes: Annotated[bool, typer.Option("--yes", "-y")] = False,
) -> None:
    """Undo world-state changes (and optionally chapters) after a given chapter."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    if truncate and not yes:
        later = [c.number for c in book.master().chapters if c.number > to]
        if later and not Confirm.ask(
            f"Delete {len(later)} chapter(s) after {to}?", default=False, console=app_ctx.err
        ):
            raise typer.Exit(1)
    with book_lock(book.paths.lock):
        removed = manage.rollback(book, to=to, truncate=truncate)
    app_ctx.err.print(
        f"[green]✓[/] World state restored to chapter {to}"
        + (f"; removed chapters {removed[0]}-{removed[-1]}" if removed else "")
        + "."
    )


@index_app.command("rebuild")
def index_rebuild(ctx: typer.Context, book_id: BookId) -> None:
    """Re-embed every chapter (needed after changing the embedding model)."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    reporter = app_ctx.reporter(CostMeter())
    with book_lock(book.paths.lock), reporter:
        chunks = manage.rebuild_index(book, lambda: app_ctx.store_factory(book), reporter)
    app_ctx.err.print(f"[green]✓[/] Indexed {chunks} chunks.")


@export_app.command("epub")
def export_epub(
    ctx: typer.Context,
    book_id: BookId,
    output: Annotated[Path | None, typer.Option("--output", "-o", help="Output path.")] = None,
    template: Annotated[
        Path | None,
        typer.Option(
            "--template",
            exists=True,
            dir_okay=False,
            help="EPUB to copy metadata, cover and styles from.",
        ),
    ] = None,
    check: Annotated[
        bool, typer.Option("--check", help="Validate with epubcheck if installed.")
    ] = False,
) -> None:
    """Build an EPUB from master_book.md."""
    app_ctx = _ctx(ctx)
    book = _open_book(app_ctx, book_id)
    out = write_book_epub(book, output, template)
    app_ctx.err.print(f"[green]✓[/] Wrote {out}")
    if check:
        from nextchap.ingest.epub import check_epub

        result = check_epub(out)
        if result is None:
            app_ctx.err.print("[yellow]![/] epubcheck is not installed; skipped validation.")
        else:
            ok, output_text = result
            app_ctx.err.print(
                "[green]✓ epubcheck passed[/]" if ok else "[red]✗ epubcheck failed[/]"
            )
            if not ok:
                app_ctx.err.print(output_text)
                raise typer.Exit(1)


# ── entry point ───────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    err = Console(stderr=True)
    try:
        rv = app(args=argv, standalone_mode=False)
        if isinstance(rv, int):
            return rv
    except typer.Exit as exc:
        return int(exc.exit_code or 0)
    except NextChapError as exc:
        error_panel(err, exc)
        return 1
    except KeyboardInterrupt:
        err.print("\n[yellow]Interrupted.[/] Progress up to the last completed step is saved.")
        return 130
    except Exception as exc:
        import click

        if isinstance(exc, click.exceptions.Abort):
            err.print("[yellow]Aborted.[/]")
            return 1
        click_exc: tuple[type[Any], ...] = (click.ClickException,)
        try:
            # Typer >= 0.20 vendors its own click, so its UsageError does not
            # subclass the installed click's ClickException.
            import typer._click as _typer_click

            click_exc = (click.ClickException, _typer_click.ClickException)
        except ImportError:
            pass
        if isinstance(exc, click_exc):
            exc.show()
            return int(exc.exit_code)
        log.exception("Unexpected error")
        log_file = next(
            (
                h.baseFilename
                for h in logging.getLogger().handlers
                if isinstance(h, logging.FileHandler)
            ),
            None,
        )
        if any(a in ("-v", "-vv", "--verbose") for a in (argv or sys.argv[1:])):
            err.print_exception()
        error_panel(err, exc, log_file)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
