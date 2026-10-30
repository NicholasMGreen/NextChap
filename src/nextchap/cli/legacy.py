"""Translate the 1.x scripts (novel_generator.py, book_manager.py) to `nextchap` commands."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nextchap.cli.app import main
from nextchap.config import Settings
from nextchap.storage.migrate import LEGACY_BOOKS
from nextchap.storage.paths import books_dir, slugify

NOTICE = "note: {old} is deprecated; use `nextchap {new}`."


def _warn(old: str, new: str) -> None:
    print(NOTICE.format(old=old, new=new), file=sys.stderr)


def _unmigrated(book_id: str) -> bool:
    legacy = Path.cwd() / LEGACY_BOOKS / book_id
    return legacy.is_dir() and not (books_dir(Settings().data_root) / slugify(book_id)).exists()


def novel_generator(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="novel_generator.py")
    ap.add_argument("novel_file")
    ap.add_argument("num_chapters", type=int)
    ap.add_argument("--api-key")
    ap.add_argument("--book-id")
    args = ap.parse_args(argv)
    if args.api_key:
        os.environ["NEXTCHAP_API_KEY"] = args.api_key
    raw_id = args.book_id or Path(args.novel_file).stem
    book_id = slugify(raw_id)
    if _unmigrated(raw_id):
        print(
            f"Book '{raw_id}' is still in ./{LEGACY_BOOKS}. Run `nextchap migrate` first.",
            file=sys.stderr,
        )
        return 1
    _warn(
        "novel_generator.py",
        f"import {args.novel_file} && nextchap generate {book_id} -n {args.num_chapters}",
    )
    if not (books_dir(Settings().data_root) / book_id).exists():
        code = main(["import", args.novel_file, "--book-id", book_id, "--yes"])
        if code:
            return code
    return main(["generate", book_id, "-n", str(args.num_chapters)])


def book_manager(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="book_manager.py")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list")
    for name in ("info", "rebuild-chroma"):
        sub.add_parser(name).add_argument("book_id")
    p = sub.add_parser("delete")
    p.add_argument("book_id")
    p.add_argument("--yes", action="store_true")
    p = sub.add_parser("cleanup")
    p.add_argument("book_id")
    p.add_argument("--no-backup", action="store_true")
    p = sub.add_parser("rebuild-state")
    p.add_argument("book_id")
    for flag in ("--no-renumber", "--no-backup", "--dry-run"):
        p.add_argument(flag, action="store_true")
    p = sub.add_parser("reconvert-epub")
    p.add_argument("book_id")
    p.add_argument("--output")
    p.add_argument("--template")
    a = ap.parse_args(argv)

    bid = getattr(a, "book_id", "")
    if bid and _unmigrated(bid):
        print(
            f"Book '{bid}' is still in ./{LEGACY_BOOKS}. Run `nextchap migrate` first.",
            file=sys.stderr,
        )
        return 1
    bid = slugify(bid) if bid else ""

    def flags(*names: str) -> list[str]:
        return [f"--{n}" for n in names if getattr(a, n.replace("-", "_"), False)]

    new: list[str]
    if a.command == "list":
        new = ["books", "list"]
    elif a.command == "info":
        new = ["books", "info", bid]
    elif a.command == "delete":
        new = ["books", "delete", bid, *flags("yes")]
    elif a.command == "cleanup":
        new = ["clean", bid, *flags("no-backup")]
    elif a.command == "rebuild-state":
        new = ["state", "rebuild", bid, *flags("no-renumber", "no-backup", "dry-run")]
    elif a.command == "rebuild-chroma":
        new = ["index", "rebuild", bid]
    else:
        new = ["export", "epub", bid]
        if a.output:
            new += ["--output", a.output]
        if a.template:
            new += ["--template", a.template]
    _warn(f"book_manager.py {a.command}", " ".join(new))
    return main(new)
