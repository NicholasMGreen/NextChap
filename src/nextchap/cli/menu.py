"""Bare `nextchap` interactive menu: pick a book, pick an action, run the verb.

This module is only a shell around the existing commands.  It reads book
overviews for display and returns the argv for the chosen command; it never
runs generation or writes book files itself.
"""

from __future__ import annotations

from typing import Any

from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text

from nextchap.cli.context import AppContext
from nextchap.cli.resolve import available_books
from nextchap.cli.ui import money
from nextchap.llm.cost import logged_spend
from nextchap.pipeline.book import Book
from nextchap.pipeline.manage import book_overview


def _overview_rows(app_ctx: AppContext, ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for book_id in ids:
        book = Book.open(app_ctx.settings, book_id)
        row = book_overview(book)
        row["spend"] = logged_spend(book.paths.usage)
        rows.append(row)
    return rows


def _book_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Book ID", style="bold")
    table.add_column("Chapters", justify="right")
    table.add_column("Generated", justify="right", style="dim")
    table.add_column("Spend", justify="right", style="dim")
    for i, row in enumerate(rows, 1):
        table.add_row(
            str(i),
            str(row["book_id"]),
            str(row["chapters"]),
            str(row["generated"]),
            money(float(row["spend"])),
        )
    return table


def run_menu(app_ctx: AppContext) -> list[str] | None:
    """Show the menu and return an argv tail for the chosen command, or None."""
    ids = available_books(app_ctx.settings)
    if not ids:
        app_ctx.err.print(
            Text.assemble(
                ("No books yet. ", "bold"),
                "Start with ",
                ("nextchap import <file>", "bold cyan"),
                ".",
            )
        )
        return None
    rows = _overview_rows(app_ctx, ids)
    total_spend = sum(float(row["spend"]) for row in rows)
    app_ctx.err.print(
        Text.assemble(
            ("NextChap", "bold"),
            f" — {len(ids)} book{'s' if len(ids) != 1 else ''} · {money(total_spend)} total spend",
        )
    )
    app_ctx.err.print(_book_table(rows))

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
        app_ctx.err.print(Text(f"Book: {chosen}", style="dim"))
        action_table = Table(show_header=False, box=None, pad_edge=False)
        action_table.add_column("#", justify="right", style="dim")
        action_table.add_column("Action")
        for i, label in enumerate(actions, 1):
            action_table.add_row(str(i), label)
        app_ctx.err.print(action_table)
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
            app_ctx.err.print(_book_table(rows))
            chosen = choose_book()
            continue
        return None
