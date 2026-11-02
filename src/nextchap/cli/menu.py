"""Bare `nextchap` interactive menu: pick a book, pick an action, run the verb.

This module is only a shell around the existing commands.  It reads book
overviews for display and returns the argv for the chosen command; it never
runs generation or writes book files itself.
"""

from __future__ import annotations

from typing import Any

from rich.console import Group
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from nextchap.cli.context import AppContext
from nextchap.cli.overview import overview_rows, overview_table
from nextchap.cli.resolve import available_books
from nextchap.cli.ui import money


def _book_panel(rows: list[dict[str, Any]], count: int) -> Panel:
    total_spend = sum(float(row["total_spend"]) for row in rows)
    header = Text.assemble(
        (f"{count} book{'s' if count != 1 else ''}", "bold"),
        " · ",
        (money(total_spend), "bold"),
        " total spend",
    )
    return Panel(
        Group(header, overview_table(rows)),
        title="NextChap",
        border_style="cyan",
        expand=False,
    )


def _action_panel(chosen: str, actions: list[str]) -> Panel:
    action_table = Table(show_header=False, box=None, pad_edge=False)
    action_table.add_column("#", justify="right", style="dim")
    action_table.add_column("Action")
    for i, label in enumerate(actions, 1):
        action_table.add_row(str(i), label)
    return Panel(action_table, title=chosen, border_style="cyan", expand=False)


def run_menu(app_ctx: AppContext) -> list[str] | None:
    """Show the menu and return an argv tail for the chosen command, or None."""
    ids = available_books(app_ctx.settings)
    if not ids:
        app_ctx.err.print(
            Panel(
                Text.assemble(
                    ("No books yet. ", "bold"),
                    "Start with ",
                    ("nextchap import <file>", "bold cyan"),
                    ".",
                ),
                title="NextChap",
                border_style="cyan",
                expand=False,
            )
        )
        return None
    rows = overview_rows(app_ctx)
    app_ctx.err.print(_book_panel(rows, len(ids)))

    def choose_book() -> str:
        if len(ids) == 1:
            return ids[0]
        answer = Prompt.ask(
            "Book",
            choices=[str(i) for i in range(1, len(ids) + 1)],
            default="1",
            console=app_ctx.err,
            show_choices=False,
        )
        return ids[int(answer) - 1]

    chosen = choose_book()
    actions = [
        "generate next chapter(s)",
        "status / continuity report",
        "fix (rebuild index, clean, roll back…)",
        "export epub",
        "switch book",
        "quit",
    ]
    while True:
        app_ctx.err.print(_action_panel(chosen, actions))
        answer = Prompt.ask(
            "Action",
            choices=[str(i) for i in range(1, len(actions) + 1)],
            default="1",
            console=app_ctx.err,
            show_choices=False,
        )
        choice = int(answer)
        if choice == 1:
            default_n = str(app_ctx.settings.defaults.chapters)
            count = Prompt.ask("How many chapters?", default=default_n, console=app_ctx.err)
            try:
                if int(count) < 1:
                    raise ValueError
            except ValueError:
                app_ctx.err.print("[yellow]![/] Enter a whole number of chapters.")
                continue
            return ["generate", chosen, "-n", count]
        if choice == 2:
            return ["status", chosen]
        if choice == 3:
            return ["fix", chosen]
        if choice == 4:
            return ["export", chosen]
        if choice == 5:
            app_ctx.err.print(_book_panel(rows, len(ids)))
            chosen = choose_book()
            continue
        return None
