"""`nextchap query`: free-text passage search over a book's retrieval index."""

from __future__ import annotations

import json
import re
from typing import Annotated

import typer
from rich.prompt import Prompt
from rich.text import Text

from nextchap.cli.context import AppContext
from nextchap.cli.resolve import available_books, resolve_book
from nextchap.config import Settings
from nextchap.errors import IndexMismatch, NextChapError
from nextchap.pipeline.book import Book
from nextchap.retrieval.bm25 import tokenize
from nextchap.retrieval.embed import embedder_name
from nextchap.retrieval.store import Hit, VectorStore

EXAMPLES_QUERY = (
    "Examples:\n"
    '  nextchap query "what does Walter know about Wade\'s powers"\n'
    "  nextchap query deadp -k 10    # fuzzy book ID, more results\n"
    "  nextchap query --json         # machine-readable results"
)

EXCERPT_CHARS = 360


def _ctx(ctx: typer.Context) -> AppContext:
    obj = ctx.find_root().obj
    assert isinstance(obj, AppContext)
    return obj


def _best_excerpt(text: str, question: str, max_chars: int = EXCERPT_CHARS) -> str:
    """Show the sentences that overlap the question most, in document order."""
    parts = [p.strip() for p in re.split(r"\n+|(?<=[.!?])\s+", text) if p.strip()]
    if not parts:
        return ""
    wanted = set(tokenize(question))
    ranked = sorted(
        range(len(parts)),
        key=lambda i: (-len(wanted & set(tokenize(parts[i]))), i),
    )
    keep = sorted(ranked[:3])
    out = " ".join(parts[i] for i in keep)
    if len(out) > max_chars:
        out = out[:max_chars]
        cut = out.rfind(" ")
        if cut > max_chars // 2:
            out = out[:cut]
        out = out.rstrip() + "…"
    elif keep[-1] != len(parts) - 1:
        out += "…"
    if keep[0] > 0:
        out = "…" + out
    return out


def _mode(store: VectorStore) -> str:
    if store.reranker is not None:
        return "hybrid search + rerank"
    return "hybrid search" if store.cfg.hybrid else "dense search"


def _open_book(app_ctx: AppContext, book_id: str | None) -> Book:
    paths = resolve_book(
        app_ctx.settings,
        book_id,
        console=app_ctx.err,
        interactive=app_ctx.interactive,
    )
    return Book(paths, app_ctx.settings)


def _search(
    app_ctx: AppContext, book: Book, question: str, top_k: int | None
) -> tuple[list[Hit], str]:
    info = VectorStore.inspect(book.paths.index)
    fix_hint = f"Run `nextchap fix {book.id} --rebuild-index` (local only, no API cost)."
    if info is None or info.chunks == 0:
        raise NextChapError(
            f"Book '{book.id}' has no retrieval index.",
            hint=f"Build it first. {fix_hint}",
        )
    if info.unreadable:
        raise NextChapError(
            f"The retrieval index for '{book.id}' is unreadable.",
            hint=fix_hint,
        )
    actual = str(info.metadata.get("embed_model", ""))
    expected = embedder_name(app_ctx.settings)
    if actual != expected:
        raise IndexMismatch(
            f"Index was built with embed_model={actual!r}, settings say {expected!r}.",
            hint=fix_hint,
        )
    try:
        store = app_ctx.store_factory(book)
    except IndexMismatch as exc:
        raise IndexMismatch(str(exc), hint=fix_hint) from exc
    return store.query(question, top_k=top_k or app_ctx.settings.retrieval.top_k), _mode(store)


def run_query(
    app_ctx: AppContext,
    book: Book,
    question: str,
    *,
    top_k: int | None = None,
    as_json: bool = False,
) -> list[Hit]:
    """Search one book and print the passages. Never calls a model."""
    hits, mode = _search(app_ctx, book, question, top_k)
    if as_json:
        payload = [
            {
                "chapter": hit.chapter,
                "score": round(hit.score, 6),
                "excerpt": _best_excerpt(hit.text, question),
            }
            for hit in hits
        ]
        app_ctx.out.print_json(json.dumps(payload))
        return hits

    app_ctx.out.print(Text.assemble(("Query: ", "dim"), question))
    if not hits:
        app_ctx.out.print(
            Text.assemble(
                ("Book: ", "dim"),
                book.id,
                " · ",
                ("No passages matched", "yellow"),
                " — try fewer words or another phrasing.",
            )
        )
        return hits
    app_ctx.out.print(
        Text.assemble(
            ("Book: ", "dim"),
            book.id,
            f" · {len(hits)} result{'s' if len(hits) != 1 else ''} ({mode})",
        )
    )
    app_ctx.out.print()
    for rank, hit in enumerate(hits, 1):
        line = Text()
        line.append(f"{rank:2}) ", style="bold")
        line.append(f"Ch {hit.chapter:02d} ", style="cyan")
        line.append(f"· ({hit.score:.2f})  ", style="dim")
        line.append(_best_excerpt(hit.text, question))
        app_ctx.out.print(line)
    return hits


def _asks_for_book(settings: Settings, token: str) -> bool:
    if not token or " " in token:
        return False
    available = available_books(settings)
    return token in available or any(book.startswith(token) for book in available)


def _split_args(
    settings: Settings, book_id: str | None, question: str | None
) -> tuple[str | None, str | None]:
    """Allow `query "a question"` as well as `query BOOK "a question"`."""
    if question is not None:
        return book_id, question
    if book_id is None:
        return None, None
    if _asks_for_book(settings, book_id):
        return book_id, None
    return None, book_id


def _ask_question(app_ctx: AppContext) -> str:
    if not app_ctx.interactive:
        raise NextChapError(
            "No question given.",
            hint='Usage: nextchap query [BOOK] "your question"',
        )
    answer = Prompt.ask("Question", default="", console=app_ctx.err).strip()
    if not answer:
        raise NextChapError(
            "No question given.",
            hint='Usage: nextchap query [BOOK] "your question"',
        )
    return answer


def query_cmd(
    ctx: typer.Context,
    book_id: Annotated[
        str | None,
        typer.Argument(
            help="Optional book ID (exact, prefix or close typo). Omit to pick a book.",
            show_default=False,
        ),
    ] = None,
    question: Annotated[
        str | None,
        typer.Argument(help="Passage request in plain words.", show_default=False),
    ] = None,
    top_k: Annotated[
        int | None,
        typer.Option("-k", "--top-k", min=1, help="How many passages to show."),
    ] = None,
    as_json: Annotated[
        bool, typer.Option("--json", help="Machine-readable output on stdout.")
    ] = False,
) -> None:
    """Find passages that match a free-text question (no model calls, no cost)."""
    app_ctx = _ctx(ctx)
    book_id, question = _split_args(app_ctx.settings, book_id, question)
    text = question or _ask_question(app_ctx)
    run_query(app_ctx, _open_book(app_ctx, book_id), text, top_k=top_k, as_json=as_json)
