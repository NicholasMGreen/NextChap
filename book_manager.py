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
import re
from pathlib import Path
from datetime import datetime


BOOKS_ROOT  = Path("./novel_books")
CHROMA_ROOT = Path("./chroma_books")
RECENT_FULL_COUNT = 3


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


_CHAPTER_HEADER_RE = re.compile(r"^##\s+Chapter\s+(\d+)\s*[:\-]?\s*(.*)$", re.IGNORECASE)
_PROMO_LINE_RE = re.compile(
    r"(patre\.?on|patreon\.com|chapters?\s+in\s+advance|^\s*\[?\d+\s+chapters?\s+in\s+advance\]?\s*$)",
    re.IGNORECASE,
)
_DOTS_RE = re.compile(r"^\s*\.{3,}\s*$")


def _is_removable_noise_line(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    if _PROMO_LINE_RE.search(s):
        return True
    if _DOTS_RE.match(s):
        return True
    return False


def _normalize_header(line: str) -> str:
    m = _CHAPTER_HEADER_RE.match(line.strip())
    if not m:
        return ""
    num = int(m.group(1))
    title = (m.group(2) or "").strip().lower()
    title = re.sub(r"\s+", " ", title)
    return f"{num}:{title}"


def _clean_master_book_text(text: str) -> tuple[str, int, int]:
    """
    Returns (cleaned_text, removed_noise_lines, removed_duplicate_headers).
    Duplicate headers are removed when they repeat before any substantive prose.
    """
    lines = text.splitlines()
    out = []
    removed_noise = 0
    removed_dupes = 0

    last_header_key = None
    saw_substantive_since_header = False

    for line in lines:
        if _is_removable_noise_line(line):
            removed_noise += 1
            continue

        header_key = _normalize_header(line)
        if header_key:
            if header_key == last_header_key and not saw_substantive_since_header:
                removed_dupes += 1
                continue
            out.append(line)
            last_header_key = header_key
            saw_substantive_since_header = False
            continue

        out.append(line)
        if line.strip():
            saw_substantive_since_header = True

    cleaned = "\n".join(out)
    if text.endswith("\n"):
        cleaned += "\n"
    return cleaned, removed_noise, removed_dupes


def _split_master_book(text: str) -> tuple[list[str], list[dict]]:
    """Parse master_book markdown into frontmatter lines and chapter blocks."""
    lines = text.splitlines()
    frontmatter = []
    chapters = []

    cur_num = None
    cur_title = ""
    cur_lines = []
    in_chapters = False

    for line in lines:
        m = _CHAPTER_HEADER_RE.match(line.strip())
        if m:
            in_chapters = True
            if cur_num is not None:
                chapters.append(
                    {"number": cur_num, "title": cur_title, "content": "\n".join(cur_lines).strip()}
                )
            cur_num = int(m.group(1))
            cur_title = (m.group(2) or "").strip() or f"Chapter {cur_num}"
            cur_lines = []
            continue

        if not in_chapters:
            frontmatter.append(line)
        elif cur_num is not None:
            cur_lines.append(line)

    if cur_num is not None:
        chapters.append({"number": cur_num, "title": cur_title, "content": "\n".join(cur_lines).strip()})

    return frontmatter, chapters


def _rebuild_master_book_text(frontmatter: list[str], chapters: list[dict]) -> str:
    fm = list(frontmatter)
    for i, line in enumerate(fm):
        if line.strip().lower().startswith("chapters:"):
            fm[i] = f"chapters: {len(chapters)}"

    parts = []
    if fm:
        parts.append("\n".join(fm).rstrip())

    for ch in chapters:
        parts.append(f"## Chapter {ch['number']}: {ch['title']}\n\n{ch['content']}".rstrip())

    return "\n\n".join(parts).rstrip() + "\n"


def _safe_title_for_number(title: str, number: int) -> str:
    if re.match(r"^chapter\s+\d+\s*$", title.strip(), re.IGNORECASE):
        return f"Chapter {number}"
    return title


def _frontmatter_value(frontmatter: list[str], key: str) -> str:
    prefix = f"{key.lower()}:"
    for line in frontmatter:
        s = line.strip()
        if s.lower().startswith(prefix):
            return s[len(prefix):].strip()
    return ""


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


def cmd_cleanup(args):
    bid = args.book_id
    book_root = BOOKS_ROOT / bid
    master_path = book_root / "master_book.md"

    if not book_root.exists():
        print(f"Book '{bid}' not found.")
        sys.exit(1)
    if not master_path.exists():
        print(f"No master_book.md found for '{bid}' at {master_path}.")
        sys.exit(1)

    original = master_path.read_text()
    cleaned, removed_noise, removed_dupes = _clean_master_book_text(original)

    if cleaned == original:
        print(f"No cleanup needed for '{bid}'.")
        return

    if not args.no_backup:
        backup = master_path.with_suffix(".md.bak")
        backup.write_text(original)
        print(f"  Backup saved: {backup}")

    master_path.write_text(cleaned)
    print(f"  Cleaned: {master_path}")
    print(f"    Removed promo/noise lines      : {removed_noise}")
    print(f"    Removed duplicate chapter heads: {removed_dupes}")
    print()


def cmd_rebuild_state(args):
    bid = args.book_id
    book_root = BOOKS_ROOT / bid
    master_path = book_root / "master_book.md"
    ch_sum_path = book_root / "chapter_summaries.json"
    recent_path = book_root / "recent_chapters.json"
    arc_path = book_root / "arc_summaries.json"
    ws_path = book_root / "world_state.json"

    if not book_root.exists() or not master_path.exists():
        print(f"Book '{bid}' missing required files (master_book.md).")
        sys.exit(1)

    original_text = master_path.read_text()
    frontmatter, chapters = _split_master_book(original_text)
    if not chapters:
        print("No chapters found in master_book.md; aborting.")
        sys.exit(1)

    original_numbers = [c["number"] for c in chapters]
    changed_numbers = 0
    if not args.no_renumber:
        for idx, ch in enumerate(chapters, 1):
            if ch["number"] != idx:
                changed_numbers += 1
            ch["number"] = idx
            ch["title"] = _safe_title_for_number(ch["title"], idx)

    rebuilt_text = _rebuild_master_book_text(frontmatter, chapters)
    if not args.dry_run:
        if not args.no_backup:
            master_path.with_suffix(".md.pre_rebuild.bak").write_text(original_text)
        master_path.write_text(rebuilt_text)

    old_summaries = []
    if ch_sum_path.exists():
        try:
            old_summaries = json.loads(ch_sum_path.read_text())
        except Exception:
            old_summaries = []

    new_summaries = []
    for i, ch in enumerate(chapters):
        summary = old_summaries[i]["summary"] if i < len(old_summaries) and isinstance(old_summaries[i], dict) else ""
        new_summaries.append({"number": ch["number"], "title": ch["title"], "summary": summary})

    recent = []
    for ch, s in zip(chapters[-RECENT_FULL_COUNT:], new_summaries[-RECENT_FULL_COUNT:]):
        recent.append(
            {"number": ch["number"], "title": ch["title"], "content": ch["content"], "summary": s["summary"]}
        )

    if ws_path.exists():
        try:
            ws = json.loads(ws_path.read_text())
        except Exception:
            ws = {}
    else:
        ws = {}
    ws["chapter_count"] = len(chapters)

    if arc_path.exists():
        try:
            arc_data = json.loads(arc_path.read_text())
            if not isinstance(arc_data, list):
                arc_data = []
        except Exception:
            arc_data = []
    else:
        arc_data = []

    if not args.dry_run:
        ch_sum_path.write_text(json.dumps(new_summaries, indent=2))
        recent_path.write_text(json.dumps(recent, indent=2))
        ws_path.write_text(json.dumps(ws, indent=2))
        arc_path.write_text(json.dumps(arc_data, indent=2))

    print(f"  Rebuilt state for '{bid}'")
    print(f"    Chapters in master_book : {len(chapters)}")
    if not args.no_renumber:
        print(f"    Chapter numbers fixed   : {changed_numbers}")
    print(f"    world_state.chapter_count -> {len(chapters)}")
    if args.dry_run:
        print("    (dry-run mode: no files written)")
    print()


def cmd_reconvert_epub(args):
    bid = args.book_id
    book_root = BOOKS_ROOT / bid
    master_path = book_root / "master_book.md"
    if not book_root.exists() or not master_path.exists():
        print(f"Book '{bid}' missing required file: {master_path}")
        sys.exit(1)

    try:
        import novel_generator as ng
    except Exception as exc:
        print("Could not import novel_generator.py dependencies.")
        print(f"  Details: {exc}")
        print("Run from the same environment you use for novel generation.")
        sys.exit(1)

    master_text = master_path.read_text()
    frontmatter, _ = _split_master_book(master_text)
    chapters = ng.parse_markdown_book(master_text)
    if not chapters:
        print("No chapters found in master_book.md; cannot build EPUB.")
        sys.exit(1)

    source_name = _frontmatter_value(frontmatter, "source")
    default_stem = Path(source_name).stem if source_name else bid
    out_path = Path(args.output) if args.output else (book_root / f"{default_stem}_with_generated.epub")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    source_candidate = Path(source_name) if source_name else None
    if source_candidate and not source_candidate.is_absolute():
        source_candidate = Path.cwd() / source_candidate

    template_path = None
    if args.template:
        template_path = Path(args.template)
    elif source_candidate and source_candidate.exists():
        template_path = source_candidate
    else:
        merged_candidate = book_root / f"{default_stem}_with_generated.epub"
        if merged_candidate.exists():
            template_path = merged_candidate

    if not template_path or not template_path.exists():
        print("Could not locate an EPUB template to preserve metadata/structure.")
        print("Provide one explicitly with --template /path/to/original.epub")
        sys.exit(1)

    handler = ng.EpubHandler()
    _, original_book = handler.read(str(template_path))
    handler.write(original_book, chapters, str(out_path))

    print(f"  Reconverted EPUB saved:")
    print(f"    {out_path.resolve()}")
    print(f"    Chapters: {len(chapters)}")
    print(f"    Template: {template_path}")
    print()


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
            "  python book_manager.py cleanup my_story\n"
            "  python book_manager.py rebuild-state my_story\n"
            "  python book_manager.py reconvert-epub my_story\n"
        ),
    )

    sub = ap.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="List all books with size and chapter counts")

    p_info = sub.add_parser("info", help="Show detailed info for one book")
    p_info.add_argument("book_id", help="Book ID (slug)")

    p_del = sub.add_parser("delete", help="Permanently delete all data for a book")
    p_del.add_argument("book_id", help="Book ID (slug)")
    p_del.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    p_clean = sub.add_parser("cleanup", help="Clean master_book.md (duplicate headers + promo lines)")
    p_clean.add_argument("book_id", help="Book ID (slug)")
    p_clean.add_argument("--no-backup", action="store_true", help="Do not write master_book.md.bak")

    p_rebuild = sub.add_parser("rebuild-state", help="Rebuild session state from master_book.md")
    p_rebuild.add_argument("book_id", help="Book ID (slug)")
    p_rebuild.add_argument("--no-renumber", action="store_true", help="Keep existing chapter numbers")
    p_rebuild.add_argument("--no-backup", action="store_true", help="Do not write master_book.md.pre_rebuild.bak")
    p_rebuild.add_argument("--dry-run", action="store_true", help="Show results without writing files")

    p_epub = sub.add_parser("reconvert-epub", help="Build a fresh EPUB from master_book.md")
    p_epub.add_argument("book_id", help="Book ID (slug)")
    p_epub.add_argument("--output", help="Output EPUB path (default: novel_books/<id>/<source>_with_generated.epub)")
    p_epub.add_argument("--template", help="Template EPUB path (metadata/structure source)")

    args = ap.parse_args()
    {
        "list": cmd_list,
        "info": cmd_info,
        "delete": cmd_delete,
        "cleanup": cmd_cleanup,
        "rebuild-state": cmd_rebuild_state,
        "reconvert-epub": cmd_reconvert_epub,
    }[args.command](args)


if __name__ == "__main__":
    main()
