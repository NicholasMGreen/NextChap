"""Shared book-list rows and table for `status` and the interactive menu."""

from __future__ import annotations

from typing import Any

from rich.table import Table

from nextchap.cli.context import AppContext
from nextchap.cli.resolve import available_books
from nextchap.cli.ui import money
from nextchap.llm.cost import logged_spend
from nextchap.pipeline.book import Book
from nextchap.pipeline.manage import book_overview


def overview_rows(app_ctx: AppContext) -> list[dict[str, Any]]:
    rows = []
    for book_id in available_books(app_ctx.settings):
        book = Book.open(app_ctx.settings, book_id)
        row = book_overview(book)
        row["total_spend"] = round(logged_spend(book.paths.usage), 6)
        rows.append(row)
    return rows


def overview_table(rows: list[dict[str, Any]]) -> Table:
    table = Table(show_edge=False)
    for col, just in (
        ("Book ID", "left"),
        ("Chapters", "right"),
        ("Generated", "right"),
        ("Spend", "right"),
        ("Size", "right"),
        ("Modified", "left"),
    ):
        table.add_column(col, justify=just)  # type: ignore[arg-type]
    for row in rows:
        table.add_row(
            str(row["book_id"]),
            str(row["chapters"]),
            str(row["generated"]),
            money(float(row["total_spend"])),
            f"{row['size_mb']:.1f} MB",
            str(row["modified"]),
        )
    return table
