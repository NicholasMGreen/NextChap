#!/usr/bin/env python3
"""
Book Manager
============
Manages the per-book data created by novel_generator.py.

Commands:
  python book_manager.py list              -- list all tracked books
  python book_manager.py info <book-id>    -- show details for one book
  python book_manager.py delete <book-id>  -- delete ALL data for a book (with confirmation)
  python book_manager.py delete <book-id> --yes  -- skip confirmation prompt

Data locations:
  novel_books/<book-id>/    -- summaries, world state, bible, generated chapters
  chroma_books/<book-id>/   -- vector embeddings
"""

import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


BOOKS_ROOT  = Path("./novel_books")
CHROMA_ROOT = Path("./chroma_books")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dir_size_mb(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1_048_576, 2)


def _count_generated(book_root: Path) -> int:
    gen_dir = book_root / "generated_chapters"
    if not gen_dir.exists():
        return 0
    return len([f for f in gen_dir.glob("chapter_*.md") if "_plan" not in f.name])


def _get_chapter_count(book_root: Path) -> int:
    state_file = book_root / "world_state.json"
    if not state_file.exists():
        return 0
    try:
        data = json.loads(state_file.read_text())
        return data.get("chapter_count", 0)
    except Exception:
        return 0


def _last_modified(path: Path) -> str:
    if not path.exists():
        return "never"
    files = list(path.rglob("*"))
    if not files:
        return "never"
    latest = max(f.stat().st_mtime for f in files if f.is_file())
    return datetime.fromtimestamp(latest).strftime("%Y-%m-%d %H:%M")


def _all_book_ids() -> list:
    ids = set()
    if BOOKS_ROOT.exists():
        ids.update(p.name for p in BOOKS_ROOT.iterdir() if p.is_dir())
    if CHROMA_ROOT.exists():
        ids.update(p.name for p in CHROMA_ROOT.iterdir() if p.is_dir())
    return sorted(ids)


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_list(_args):
    book_ids = _all_book_ids()

    if not book_ids:
        print("No books found.")
        print(f"  Books are stored in: {BOOKS_ROOT.resolve()}")
        return

    print(f"\n  {'BOOK ID':<30} {'CHAPTERS':>9} {'GENERATED':>10} {'SIZE':>8}  LAST MODIFIED")
    print("  " + "-" * 74)

    for bid in book_ids:
        book_root   = BOOKS_ROOT  / bid
        chroma_root = CHROMA_ROOT / bid
        chapters    = _get_chapter_count(book_root)
        generated   = _count_generated(book_root)
        size        = _dir_size_mb(book_root) + _dir_size_mb(chroma_root)
        modified    = _last_modified(book_root)
        print(f"  {bid:<30} {chapters:>9} {generated:>10} {size:>7.1f}MB  {modified}")

    print()


def cmd_info(args):
    bid = args.book_id

    book_root   = BOOKS_ROOT  / bid
    chroma_root = CHROMA_ROOT / bid

    if not book_root.exists() and not chroma_root.exists():
        print(f"Book '{bid}' not found.")
        sys.exit(1)

    print(f"\n  Book ID       : {bid}")
    print(f"  Book folder   : {book_root.resolve()}")
    print(f"  Chroma folder : {chroma_root.resolve()}")

    state_file = book_root / "world_state.json"
    if state_file.exists():
        try:
            ws = json.loads(state_file.read_text())
            print(f"\n  World State")
            print(f"    Chapters processed : {ws.get('chapter_count', 0)}")
            print(f"    Current arc        : {ws.get('current_arc', 'N/A')}")
            print(f"    Story tone         : {ws.get('story_tone', 'N/A')}")
            print(f"    Narrative POV      : {ws.get('narrative_pov', 'N/A')}")
            chars = ws.get("characters", {})
            if chars:
                print(f"    Characters         : {', '.join(list(chars.keys())[:8])}")
            threads = ws.get("active_plot_threads", [])
            if threads:
                print(f"    Active threads     : {len(threads)}")
                for t in threads[:4]:
                    print(f"      - {t[:80]}")
            hook = ws.get("last_chapter_hook", "")
            if hook:
                print(f"    Last hook          : {hook[:100]}")
        except Exception as exc:
            print(f"  (Could not read world state: {exc})")

    gen_dir = book_root / "generated_chapters"
    if gen_dir.exists():
        chapter_files = sorted(f for f in gen_dir.glob("chapter_*.md") if "_plan" not in f.name)
        plan_files    = list(gen_dir.glob("chapter_*_plan.md"))
        print(f"\n  Generated Chapters")
        print(f"    Count  : {len(chapter_files)}")
        print(f"    Plans  : {len(plan_files)}")
        if chapter_files:
            print(f"    Range  : {chapter_files[0].stem} to {chapter_files[-1].stem}")

    bible_file = book_root / "master_bible.md"
    if bible_file.exists():
        size = bible_file.stat().st_size
        print(f"\n  Master Bible   : {size:,} bytes")

    total_mb = _dir_size_mb(book_root) + _dir_size_mb(chroma_root)
    print(f"\n  Total disk use : {total_mb:.1f} MB")
    print(f"  Last modified  : {_last_modified(book_root)}")
    print()


def cmd_delete(args):
    bid = args.book_id

    book_root   = BOOKS_ROOT  / bid
    chroma_root = CHROMA_ROOT / bid

    if not book_root.exists() and not chroma_root.exists():
        print(f"Book '{bid}' not found. Nothing to delete.")
        sys.exit(1)

    total_mb = _dir_size_mb(book_root) + _dir_size_mb(chroma_root)
    generated = _count_generated(book_root)

    print(f"\n  Book to delete : {bid}")
    print(f"  Folders        :")
    if book_root.exists():
        print(f"    {book_root.resolve()}  ({_dir_size_mb(book_root):.1f} MB)")
    if chroma_root.exists():
        print(f"    {chroma_root.resolve()}  ({_dir_size_mb(chroma_root):.1f} MB)")
    print(f"  Generated chs  : {generated}")
    print(f"  Total size     : {total_mb:.1f} MB")
    print()

    if not args.yes:
        answer = input(f"  Permanently delete all data for '{bid}'? [yes/N]: ").strip()
        if answer.lower() != "yes":
            print("  Cancelled.")
            return

    deleted = []
    if book_root.exists():
        shutil.rmtree(book_root)
        deleted.append(str(book_root))
    if chroma_root.exists():
        shutil.rmtree(chroma_root)
        deleted.append(str(chroma_root))

    # Clean up empty root dirs
    for root in [BOOKS_ROOT, CHROMA_ROOT]:
        if root.exists() and not any(root.iterdir()):
            root.rmdir()

    print(f"  Deleted:")
    for d in deleted:
        print(f"    {d}")
    print(f"  Book '{bid}' has been fully removed.\n")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Manage per-book data for novel_generator.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python book_manager.py list\n"
            "  python book_manager.py info my_story\n"
            "  python book_manager.py delete my_story\n"
            "  python book_manager.py delete my_story --yes\n"
        ),
    )

    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all books with size and chapter counts")

    p_info = sub.add_parser("info", help="Show detailed info for one book")
    p_info.add_argument("book_id", help="Book ID (slug)")

    p_del = sub.add_parser("delete", help="Permanently delete all data for a book")
    p_del.add_argument("book_id", help="Book ID (slug)")
    p_del.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    args = ap.parse_args()
    {"list": cmd_list, "info": cmd_info, "delete": cmd_delete}[args.command](args)


if __name__ == "__main__":
    main()
