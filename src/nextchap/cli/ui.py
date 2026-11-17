"""Rich terminal UI: live progress, streaming preview, cost footer, review prompts."""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from types import TracebackType

import click
from rich.console import Console, Group, RenderableType
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.prompt import Prompt
from rich.text import Text

from nextchap.config import Settings
from nextchap.errors import NextChapError
from nextchap.llm.cost import CostMeter
from nextchap.pipeline.reporter import ReviewAction

PREVIEW_LINES = 12


def error_panel(console: Console, exc: BaseException, log_path: str | None = None) -> None:
    if isinstance(exc, NextChapError):
        body = Text(str(exc), style="bold")
        if exc.hint:
            body.append("\n\n→ ", style="cyan").append(exc.hint)
        console.print(Panel(body, title=type(exc).__name__, border_style="red", expand=False))
        return
    body = Text(f"{type(exc).__name__}: {exc}", style="bold")
    if log_path:
        body.append(
            f"\n\nFull traceback written to {log_path}. Re-run with -v to print it.", style="dim"
        )
    console.print(Panel(body, title="Unexpected error", border_style="red", expand=False))


def money(value: float) -> str:
    return f"${value:.4f}" if value < 1 else f"${value:.2f}"


ROLE_STYLES = {"writer": "magenta", "planner": "yellow", "analyst": "cyan", "critic": "blue"}

# U+F0CA9 (nf-md-gauge), dimmed segment prefix when the terminal has Nerd Fonts.
FOOTER_ICON = "\U000f0ca9"


def fmt_tokens(n: int) -> str:
    """Compact token counts: 999 → ``999``, 41200 → ``41.2k``, 1300000 → ``1.3M``."""
    if n < 1_000:
        return str(n)
    if n < 1_000_000:
        return f"{n / 1_000:.1f}k"
    return f"{n / 1_000_000:.1f}M"


class _Dashboard:
    def __init__(self, reporter: RichReporter) -> None:
        self.r = reporter

    def __rich__(self) -> RenderableType:
        r = self.r
        parts: list[RenderableType] = [r.progress]
        if r.show_preview and r.preview_text:
            words = len(r.preview_text.split())
            style = "green" if words >= r.min_words else "yellow"
            tail = "\n".join(r.preview_text.rstrip().splitlines()[-PREVIEW_LINES:])
            parts.append(
                Panel(
                    Text(tail, overflow="fold"),
                    title="Live preview",
                    subtitle=Text(f"{words} words", style=style),
                    border_style="dim",
                    height=PREVIEW_LINES + 2,
                )
            )
        parts.append(r.footer())
        return Group(*parts)


class RichReporter:
    """Implements the engine's Reporter protocol."""

    def __init__(
        self, console: Console, meter: CostMeter, settings: Settings, *, show_preview: bool = True
    ) -> None:
        self.console = console
        self.meter = meter
        self.settings = settings
        self.show_preview = show_preview
        self.min_words = settings.generation.min_words
        self.preview_text = ""
        self._preview_newlines = 0
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(bar_width=28),
            MofNCompleteColumn(),
            TextColumn("[dim]{task.fields[detail]}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
        self._phase: TaskID | None = None
        self._step: TaskID | None = None
        self._step_label = ""
        self._step_started = 0.0
        self._live = Live(_Dashboard(self), console=console, refresh_per_second=8, transient=True)

    def __enter__(self) -> RichReporter:
        self._live.start()
        return self

    def __exit__(
        self, et: type[BaseException] | None, ev: BaseException | None, tb: TracebackType | None
    ) -> None:
        self._finish_step(ok=et is None)
        self._live.stop()

    @contextmanager
    def paused(self) -> Iterator[None]:
        self._live.stop()
        try:
            yield
        finally:
            self._live.start()

    def footer(self) -> Text:
        m = self.meter
        t = Text()

        def sep() -> None:
            t.append(" · ", style="dim")

        # Token counter: \u2191 prompt, \u2193 completion.
        if self.settings.ui.footer_icon:
            t.append(f" {FOOTER_ICON} ", style="dim")
            t.append("\u2191", style="green").append(fmt_tokens(m.prompt_tokens), style="green")
            t.append(" ")
            t.append("\u2193", style="magenta").append(
                fmt_tokens(m.completion_tokens), style="magenta"
            )
        else:
            t.append(" in ", style="dim").append(fmt_tokens(m.prompt_tokens), style="green")
            t.append(" out ", style="dim").append(fmt_tokens(m.completion_tokens), style="magenta")

        rate = m.cache_hit_rate
        cache_style = "dim" if rate < 0.10 else "cyan" if rate < 0.50 else "green"
        sep()
        t.append("cache ", style="dim").append(f"{rate:.0%}", style=cache_style)

        sep()
        t.append("spend ", style="dim")
        if m.max_cost:
            frac = m.total_cost / m.max_cost
            spend_style = (
                "bold green" if frac < 0.5 else "bold yellow" if frac < 0.8 else "bold red"
            )
            t.append(money(m.total_cost), style=spend_style)
            t.append("/", style="dim").append(money(m.max_cost), style="dim")
        else:
            t.append(money(m.total_cost), style="bold yellow")

        # What actually did the work in this run (zero-call roles stay hidden).
        usage = m.by_role_model()
        if usage:
            order = list(self.settings.models.roles())
            for role, model in sorted(
                usage,
                key=lambda k: (order.index(k[0]) if k[0] in order else len(order), k[0], k[1]),
            ):
                sep()
                t.append(role, style=ROLE_STYLES.get(role, "cyan"))
                t.append(f" {model}", style="dim")
                t.append(f" ({usage[(role, model)].calls})", style="dim")
        else:
            sep()
            t.append("writer ", style="dim").append(self.settings.models.writer.model, style="cyan")
        return t

    # ── Reporter protocol ─────────────────────────────────────────────────────

    def phase(self, name: str, total: int | None = None) -> None:
        self._finish_step()
        self.end_phase()
        self._phase = self.progress.add_task(name, total=total, detail="")

    def advance(self, n: int = 1, detail: str = "") -> None:
        if self._phase is not None:
            self.progress.update(self._phase, advance=n, detail=detail)

    def end_phase(self) -> None:
        if self._phase is not None:
            task = next((t for t in self.progress.tasks if t.id == self._phase), None)
            if task is not None:
                self.console.print(
                    f"[green]✓[/] {task.description} [dim]({task.completed:.0f}, "
                    f"{task.elapsed or 0:.0f}s)[/]"
                )
            self.progress.remove_task(self._phase)
            self._phase = None

    def _finish_step(self, ok: bool = True) -> None:
        if self._step is None:
            return
        elapsed = time.monotonic() - self._step_started
        mark = "[green]✓[/]" if ok else "[red]✗[/]"
        self.console.print(f"  {mark} {self._step_label} [dim]{elapsed:.1f}s[/]")
        self.progress.remove_task(self._step)
        self._step = None

    def step(self, label: str) -> None:
        self._finish_step()
        if not label.startswith("Writing") and not label.startswith("Extending"):
            self.preview_text = ""
            self._preview_newlines = 0
        self._step_label = label
        self._step_started = time.monotonic()
        self._step = self.progress.add_task(label, total=None, detail="")

    def preview(self, text: str) -> None:
        # The engine streams one delta per token, but rich's Live repaints the
        # whole dashboard on every frame with no diffing. Updating the panel
        # per token makes it strobe; only redraw on completed lines instead.
        n = text.count("\n")
        if n <= self._preview_newlines:
            return
        self._preview_newlines = n
        self.preview_text = text

    def info(self, message: str) -> None:
        self.console.print(f"[cyan]•[/] {message}")

    def warn(self, message: str) -> None:
        self.console.print(f"[yellow]![/] {message}")

    def chapter_done(self, number: int, title: str, words: int, cost: float) -> None:
        self._finish_step()
        self.preview_text = ""
        self._preview_newlines = 0
        self.console.print(
            Panel(
                Text.assemble(
                    (f"Chapter {number}: {title}", "bold"),
                    "  ",
                    (f"{words:,} words", "green" if words >= self.min_words else "yellow"),
                    "  ",
                    (money(cost), "dim"),
                ),
                border_style="green",
                expand=False,
            )
        )


class PlainReporter:
    """Line-oriented output for pipes, CI and --plain."""

    def __init__(self, console: Console) -> None:
        self.console = console
        self._total: int | None = None
        self._done = 0
        self._name = ""

    def __enter__(self) -> PlainReporter:
        return self

    def __exit__(self, *exc: object) -> None:
        pass

    @contextmanager
    def paused(self) -> Iterator[None]:
        yield

    def phase(self, name: str, total: int | None = None) -> None:
        self._name, self._total, self._done = name, total, 0
        self.console.print(f"== {name}" + (f" ({total})" if total else ""))

    def advance(self, n: int = 1, detail: str = "") -> None:
        self._done += n
        if self._total and (
            self._done == self._total or self._done % max(1, self._total // 10) == 0
        ):
            self.console.print(f"   {self._name}: {self._done}/{self._total}")

    def end_phase(self) -> None:
        pass

    def step(self, label: str) -> None:
        self.console.print(f"-> {label}")

    def preview(self, text: str) -> None:
        pass

    def info(self, message: str) -> None:
        self.console.print(f"info: {message}")

    def warn(self, message: str) -> None:
        self.console.print(f"warning: {message}")

    def chapter_done(self, number: int, title: str, words: int, cost: float) -> None:
        self.console.print(f"done: chapter {number} '{title}' {words} words {money(cost)}")


class RichReview:
    """Interactive plan/chapter review (--review)."""

    def __init__(self, console: Console, reporter: RichReporter | PlainReporter) -> None:
        self.console = console
        self.reporter = reporter

    def _ask(self) -> str:
        return Prompt.ask(
            "[bold][a][/]ccept  [bold][e][/]dit in $EDITOR  [bold][r][/]egenerate  [bold][q][/]uit",
            choices=["a", "e", "r", "q"],
            default="a",
            console=self.console,
            show_choices=False,
        )

    def _review(
        self, title: str, body: RenderableType, text: str, ext: str
    ) -> tuple[ReviewAction, str]:
        with self.reporter.paused():
            while True:
                self.console.print(Panel(body, title=title, border_style="cyan"))
                choice = self._ask()
                if choice == "a":
                    return "accept", text
                if choice == "q":
                    return "quit", ""
                if choice == "r":
                    note = Prompt.ask(
                        "Notes for the next attempt (optional)", default="", console=self.console
                    )
                    return "regenerate", note
                edited = click.edit(text, extension=ext)
                if edited is not None and edited.strip():
                    return "accept", edited

    def review_plan(self, chapter: int, plan_markdown: str) -> tuple[ReviewAction, str]:
        return self._review(
            f"Plan for chapter {chapter}", Markdown(plan_markdown), plan_markdown, ".md"
        )

    def review_chapter(self, chapter: int, prose: str) -> tuple[ReviewAction, str]:
        lines = prose.splitlines()
        shown = prose if len(lines) <= 80 else "\n".join([*lines[:40], "", "[…]", "", *lines[-30:]])
        title = f"Chapter {chapter} · {len(prose.split()):,} words"
        return self._review(title, Text(shown), prose, ".txt")
