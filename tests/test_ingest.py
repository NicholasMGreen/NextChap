from __future__ import annotations

from pathlib import Path

import ebooklib
from ebooklib import epub

from nextchap.ingest import master_book
from nextchap.ingest.cleanup import NoiseFilter, clean_master_markdown, header_key
from nextchap.ingest.epub import prose_to_html, read_epub, write_epub
from nextchap.ingest.master_book import Chapter, MasterBook
from nextchap.ingest.text import chinese_to_int, parse_text, roman_to_int, words_to_int
from tests.conftest import build_epub, novel_text


def test_number_parsers() -> None:
    assert roman_to_int("XIV") == 14
    assert roman_to_int("IIII") is None
    assert words_to_int("twenty-one") == 21
    assert words_to_int("one hundred and five") == 105
    assert words_to_int("banana") is None
    assert chinese_to_int("十二") == 12
    assert chinese_to_int("一百零五") == 105


def test_prose_lines_are_not_chapter_headers() -> None:
    text = (
        "Chapter 1: Start\n\nHe counted.\n\n3. He ran to the door.\n\n1984: the year it began\n\n"
        "2. Then the second item of a list in prose continues here.\n\nChapter 2 - Next\n\nMore.\n"
    )
    r = parse_text(text)
    assert [(c.number, c.title) for c in r.chapters] == [(1, "Start"), (2, "Next")]
    assert "3. He ran to the door." in r.chapters[0].content


def test_header_variants() -> None:
    text = (
        "Chapter One: Dawn\n\naaa\n\nCHAPTER II — Dusk\n\nbbb\n\nChapter Three\n\nccc\n\n"
        "4: Bare Number\n\nddd\n\n第五章 雨\n\neee\n"
    )
    r = parse_text(text)
    assert [(c.number, c.title) for c in r.chapters] == [
        (1, "Dawn"),
        (2, "Dusk"),
        (3, "Chapter 3"),
        (4, "Bare Number"),
        (5, "雨"),
    ]


def test_restarting_numbers_are_renumbered() -> None:
    text = "Chapter 1\n\na\n\nChapter 2\n\nb\n\nPrologue: Volume Two\n\nc\n\nChapter 1\n\nd\n"
    r = parse_text(text)
    assert [c.number for c in r.chapters] == [1, 2, 3, 4]
    assert r.chapters[2].title == "Prologue: Volume Two"
    assert r.renumbered and any("renumbered" in a for a in r.anomalies)


def test_repeated_header_merges() -> None:
    r = parse_text("Chapter 3\n\nChapter 3: The Fall\n\nbody text\n")
    assert len(r.chapters) == 1
    assert r.chapters[0].title == "The Fall"


def test_whole_file_fallback() -> None:
    r = parse_text("Just a story with no headings.\n")
    assert len(r.chapters) == 1
    assert r.anomalies


def test_cleanup_whitespace_normalization_and_promo() -> None:
    assert header_key("## Chapter 3:  The   Fall") == header_key("## Chapter 3: the fall")
    md = (
        "## Chapter 3:  The   Fall\n## Chapter 3: The Fall\n\nBody line.\n"
        "Read 20 chapters in advance on Patreon!\n.....\n"
        "A long sentence that mentions patreon in passing but is really prose " + "x " * 120 + "\n"
    )
    cleaned, rep = clean_master_markdown(md)
    assert rep.removed_duplicate_headers == 1
    assert rep.removed_noise == 2
    assert "mentions patreon in passing" in cleaned


def test_custom_noise_patterns() -> None:
    f = NoiseFilter([r"^translator'?s note"])
    assert f.is_noise("Translator's note: thanks!")
    assert not f.is_noise("He noted the translator's mistake.")


def test_master_book_roundtrip_and_upsert(tmp_path: Path) -> None:
    book = MasterBook(
        {"source": "x.epub"}, [Chapter(1, "One", "Hello."), Chapter(2, "Two", "World.")]
    )
    text = master_book.render(book)
    assert "chapters: 2" in text
    again = master_book.parse(text)
    assert [(c.number, c.title, c.content) for c in again.chapters] == [
        (1, "One", "Hello."),
        (2, "Two", "World."),
    ]
    p = tmp_path / "master_book.md"
    master_book.save(p, again)
    assert master_book.upsert_chapter(p, Chapter(3, "Three", "## Chapter 9 sneaky\nText"))
    assert not master_book.upsert_chapter(p, Chapter(3, "Three", "Chapter 9 sneaky\nText"))
    loaded = master_book.load(p)
    assert [c.number for c in loaded.chapters] == [1, 2, 3]
    assert loaded.frontmatter["chapters"] == "3"


def test_epub_read(tmp_path: Path) -> None:
    report, meta = read_epub(build_epub(tmp_path / "b.epub", chapters=3))
    assert meta.title == "Test Book" and meta.author == "Author"
    assert [(c.number, c.title) for c in report.chapters] == [
        (1, "Part 1"),
        (2, "Part 2"),
        (3, "Part 3"),
    ]
    assert report.removed_noise == 3
    assert "Patreon" not in report.chapters[0].content


def test_epub_write_keeps_cover_and_builds_nav(tmp_path: Path) -> None:
    src = build_epub(tmp_path / "b.epub", chapters=2)
    report, meta = read_epub(src)
    chapters = [
        *report.chapters,
        Chapter(3, "New one", "Rank #1 hero *smiled*.\n\n* * *\n\nLater **that** day."),
    ]
    out = tmp_path / "out.epub"
    write_epub(chapters, out, src, meta)
    book = epub.read_epub(str(out))
    names = [i.file_name for i in book.get_items()]
    assert "cover.xhtml" in names and "toc.ncx" in names and "nav.xhtml" in names
    assert [sid for sid, _ in book.spine][:2] == ["cover", "nav"]
    ch3 = next(
        i
        for i in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)
        if i.file_name.endswith("nc-ch00003.xhtml")
    )
    html = ch3.get_content().decode()
    assert "Rank #1 hero <em>smiled</em>." in html
    assert '<hr class="scene-break"/>' in html
    assert "<strong>that</strong>" in html
    assert "Chapter 3: New one" in html


def test_epub_write_without_template(tmp_path: Path) -> None:
    from nextchap.ingest.epub import EpubMeta

    out = tmp_path / "plain.epub"
    report = parse_text(novel_text(2, 50))
    write_epub(report.chapters, out, None, EpubMeta(title="Plain", author="Me"))
    again, meta = read_epub(out)
    assert meta.title == "Plain"
    assert len(again.chapters) == 2


def test_prose_to_html_escapes() -> None:
    html = prose_to_html("A & B", "x < y & z\nnext line\n\nsecond para")
    assert "A &amp; B" in html and "<p>x &lt; y &amp; z<br/>next line</p>" in html
    single_block = prose_to_html("T", "line one\nline two")
    assert "<p>line one</p>\n<p>line two</p>" in single_block
