"""Resolve a partial or mistyped book ID to a real book.

The CLI should never make someone copy-paste an exact ID just to run a command.
Resolution falls through exact -> prefix -> fuzzy matches, and asks the user to
pick when several books fit.  Paths are still built only from validated IDs.
"""

from __future__ import annotations

from collections.abc import Sequence
from difflib import get_close_matches

from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from nextchap.config import Settings
from nextchap.errors import BookNotFound, NextChapError
from nextchap.storage.paths import BookPaths, all_book_ids, book_paths, validate_book_id

FUZZY_CUTOFF = 0.6


class AmbiguousBook(NextChapError):
    """Several books matched the requested partial ID."""

    hint = "Pass a longer, more specific ID, or run in a terminal to pick from a list."


def available_books(settings: Settings) -> list[str]:
    return all_book_ids(settings.data_root)


def _available_hint(ids: Sequence[str], *, limit: int = 12) -> str:
    shown = ", ".join(ids[:limit])
    if len(ids) > limit:
        shown += f", … ({len(ids) - limit} more)"
    return f"Available: {shown}. Run `nextchap status` to see them."


def match_book_ids(available: Sequence[str], partial: str) -> list[str]:
    """Return matching IDs in priority order: exact, prefix, then fuzzy."""
    if partial in available:
        return [partial]
    prefixed = [book for book in available if book.startswith(partial)]
    if prefixed:
        return prefixed
    return get_close_matches(partial, available, n=5, cutoff=FUZZY_CUTOFF)


def pick_book(console: Console, ids: Sequence[str], *, title: str = "Book") -> str:
    """Numbered Rich picker.  Returns the chosen book ID."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Book ID", style="bold")
    for i, book_id in enumerate(ids, 1):
        table.add_row(str(i), book_id)
    console.print(table)
    answer = Prompt.ask(
        title,
        choices=[str(i) for i in range(1, len(ids) + 1)],
        default="1",
        console=console,
        show_choices=False,
    )
    return ids[int(answer) - 1]


def resolve_book(
    settings: Settings,
    partial: str | None,
    *,
    console: Console | None = None,
    interactive: bool | None = None,
) -> BookPaths:
    """Resolve ``partial`` to a book, picking one interactively when needed."""
    err = console or Console(stderr=True)
    is_interactive = err.is_terminal if interactive is None else interactive
    query = partial.strip() if partial is not None else ""
    if query:
        validate_book_id(query)
    available = available_books(settings)
    if not available:
        if query:
            raise BookNotFound(
                f"Book {query!r} not found (no books in {settings.data_root}).",
                hint="Import one first: nextchap import <book.epub|book.txt>",
            )
        raise BookNotFound(
            f"No books found in {settings.data_root}.",
            hint="Import one first: nextchap import <book.epub|book.txt>",
        )

    if not query:
        if len(available) == 1:
            chosen = available[0]
        elif is_interactive:
            chosen = pick_book(err, available)
        else:
            raise AmbiguousBook(
                "Which book? Pass a book ID.",
                hint=_available_hint(available),
            )
        err.print(f"[dim]using book:[/] [bold]{chosen}[/]")
        return book_paths(settings.data_root, chosen)

    matches = match_book_ids(available, query)
    if not matches:
        raise BookNotFound(
            f"Book {query!r} not found (no exact, prefix or close match).",
            hint=_available_hint(available),
        )
    if len(matches) == 1:
        chosen = matches[0]
        if chosen != query:
            err.print(f"[dim]using book:[/] [bold]{chosen}[/] [dim](matched {query!r})[/]")
        return book_paths(settings.data_root, chosen)
    if not is_interactive:
        raise AmbiguousBook(
            f"{query!r} matches several books: {', '.join(matches)}.",
            hint=_available_hint(available),
        )
    chosen = pick_book(err, matches, title=f"Books matching {query!r}")
    return book_paths(settings.data_root, chosen)
