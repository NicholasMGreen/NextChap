from __future__ import annotations

from pathlib import Path

import pytest
from ebooklib import epub

from nextchap import config as config_mod
from nextchap.config import RetrievalConfig, Settings
from nextchap.ingest.text import parse_text
from nextchap.llm.cost import CostMeter
from nextchap.pipeline.book import Book, BookMeta
from nextchap.pipeline.engine import Engine
from nextchap.retrieval.embed import HashEmbedder
from nextchap.retrieval.store import VectorStore
from tests.fakes import FakeLLM, prose


@pytest.fixture(autouse=True)
def _isolate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests away from the real user config, env keys and data dirs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(config_mod, "user_config_path", lambda: tmp_path / "no-user-config.toml")
    for var in ("OPENROUTER_API_KEY", "NEXTCHAP_API_KEY"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path / "data", retrieval=RetrievalConfig(embed_backend="hash"))


def novel_text(chapters: int = 6, words: int = 320) -> str:
    parts = []
    for i in range(1, chapters + 1):
        parts.append(f"Chapter {i}: Title {i}\n\n{prose(words, f'src{i}')}\n")
    return "\n".join(parts)


@pytest.fixture
def novel_txt(tmp_path: Path) -> Path:
    p = tmp_path / "novel.txt"
    p.write_text(novel_text(), encoding="utf-8")
    return p


def build_epub(path: Path, chapters: int = 4) -> Path:
    book = epub.EpubBook()
    book.set_identifier("test-book")
    book.set_title("Test Book")
    book.set_language("en")
    book.add_author("Author")
    cover = epub.EpubHtml(uid="cover", file_name="cover.xhtml", title="Cover")
    cover.content = "<h1>Test Book</h1><p>by Author</p>"
    items = [cover]
    for i in range(1, chapters + 1):
        c = epub.EpubHtml(uid=f"c{i}", file_name=f"c{i}.xhtml", title=f"C{i}")
        body = "".join(f"<p>{p}</p>" for p in prose(300, f"ep{i}").split("\n\n"))
        promo = "<p>Read 10 chapters in advance on Patreon!</p>"
        c.content = f"<h1>Chapter {i}</h1><p>Chapter {i}: Part {i}</p>{body}{promo}"
        items.append(c)
    for it in items:
        book.add_item(it)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    book.spine = ["nav", *items]
    book.toc = items[1:]
    epub.write_epub(str(path), book, {})
    return path


@pytest.fixture
def novel_epub(tmp_path: Path) -> Path:
    return build_epub(tmp_path / "novel.epub")


def make_engine(
    settings: Settings, book_id: str = "test", llm: FakeLLM | None = None, **kw: object
) -> tuple[Engine, FakeLLM, Book]:
    meter = CostMeter(kw.pop("max_cost", None))  # type: ignore[arg-type]
    fake = llm or FakeLLM(meter)
    fake.meter = meter
    book = Book.open(settings, book_id, must_exist=False)
    engine = Engine(
        book,
        fake,
        meter,
        lambda: VectorStore(book.paths.index, HashEmbedder(), settings.retrieval),
        **kw,
    )  # type: ignore[arg-type]
    return engine, fake, book


@pytest.fixture
def imported(settings: Settings, novel_txt: Path) -> tuple[Engine, FakeLLM, Book]:
    engine, fake, book = make_engine(settings)
    report = parse_text(novel_txt.read_text(encoding="utf-8"))
    engine.import_chapters(
        report.chapters, BookMeta(book_id="test", source_name="novel.txt", source_kind="txt"), None
    )
    return engine, fake, book
