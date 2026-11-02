from __future__ import annotations

import json
from pathlib import Path

import pytest

from nextchap.config import RetrievalConfig
from nextchap.errors import BookLocked, IndexMismatch, InvalidBookId
from nextchap.ingest.master_book import Chapter
from nextchap.retrieval.bm25 import BM25, rrf
from nextchap.retrieval.context import fit_items, fit_text
from nextchap.retrieval.embed import HashEmbedder
from nextchap.retrieval.store import VectorStore, chunk_chapter
from nextchap.storage.atomic import atomic_write_json, read_json
from nextchap.storage.lock import book_lock
from nextchap.storage.paths import book_paths, slugify, validate_book_id


@pytest.mark.parametrize("bad", ["../etc", "a/b", "", "UPPER", "-lead", "x" * 65, "..", "a b"])
def test_invalid_book_ids(bad: str) -> None:
    with pytest.raises(InvalidBookId):
        validate_book_id(bad)
    with pytest.raises(InvalidBookId):
        book_paths(Path("/tmp/data"), bad)


def test_slugify() -> None:
    assert slugify("Martial Peak (Vol. 2).epub") == "martial_peak_vol_2epub"
    assert slugify("Émile & Co") == "emile_co"
    assert slugify("???") == "book"
    assert validate_book_id(slugify("x" * 200))


def test_atomic_write_leaves_no_temp(tmp_path: Path) -> None:
    p = tmp_path / "d" / "f.json"
    atomic_write_json(p, {"ä": 1})
    atomic_write_json(p, {"ä": 2})
    assert read_json(p) == {"ä": 2}
    assert [x.name for x in p.parent.iterdir()] == ["f.json"]


def test_lock_is_exclusive(tmp_path: Path) -> None:
    lock = tmp_path / ".lock"
    with book_lock(lock), pytest.raises(BookLocked), book_lock(lock):
        pass


def test_fit_text_and_items() -> None:
    long = "\n".join(f"line {i}" for i in range(1000))
    head = fit_text(long, 50, keep="head")
    tail = fit_text(long, 50, keep="tail")
    assert head.startswith("line 0") and head.endswith("[…]")
    assert tail.startswith("[…]") and tail.endswith("line 999")
    assert fit_items(["a" * 40, "b" * 40, "c" * 40], 25) == ["b" * 40, "c" * 40]


def test_bm25_and_rrf() -> None:
    bm = BM25(["the azure dragon fist strikes", "a quiet morning tea", "dragon tea"])
    assert bm.top("Azure Dragon", 2) == [0, 2]
    fused = rrf([["a", "b"], ["b", "c"]], [0.5, 0.5])
    assert fused[0][0] == "b"


def test_chunker_respects_scene_breaks() -> None:
    body = "\n\n".join(["word " * 100] * 6) + "\n\n* * *\n\nShort final scene."
    chunks = chunk_chapter(Chapter(5, "T", body), words=250, overlap=50)
    assert chunks[-1].text == "Short final scene."
    assert all(c.id.startswith("ch00005-s") for c in chunks)
    assert len({c.id for c in chunks}) == len(chunks)


def test_store_upsert_idempotent_and_hybrid(tmp_path: Path) -> None:
    store = VectorStore(tmp_path / "idx", HashEmbedder(), RetrievalConfig())
    chapters = [
        Chapter(i, f"T{i}", "\n\n".join(["filler text about the weather " * 10] * 5))
        for i in range(1, 6)
    ]
    chapters[2] = Chapter(
        3, "T3", chapters[2].content + "\n\nLin Feng unleashed the Azure Dragon Fist."
    )
    store.upsert_chapters(chapters)
    n = store.count()
    store.upsert_chapters(chapters)
    assert store.count() == n
    assert store.query("Azure Dragon Fist")[0].chapter == 3
    assert all(h.chapter != 3 for h in store.query("Azure Dragon Fist", exclude_chapters={3}))
    assert store.indexed()[3] == chapters[2].content_hash
    store.delete_chapters([3])
    assert 3 not in store.indexed()


def test_store_refuses_other_embedding(tmp_path: Path) -> None:
    VectorStore(tmp_path / "idx", HashEmbedder(), RetrievalConfig())
    with pytest.raises(IndexMismatch):
        VectorStore(tmp_path / "idx", HashEmbedder(dim=64), RetrievalConfig())


def test_store_inspect_distinguishes_missing_and_unreadable(tmp_path: Path) -> None:
    assert VectorStore.inspect(tmp_path / "missing") is None
    broken = tmp_path / "broken"
    broken.write_text("not a chroma directory", encoding="utf-8")
    info = VectorStore.inspect(broken)
    assert info is not None and info.unreadable and info.chunks == 0


def test_estimates_scale(settings: object) -> None:
    from nextchap.models import ModelRegistry
    from nextchap.pipeline.estimate import estimate_generation, estimate_import, total

    reg = ModelRegistry.offline()
    one = total(estimate_generation(settings, reg, 1, 8000, 25000))  # type: ignore[arg-type]
    ten = total(estimate_generation(settings, reg, 10, 8000, 25000))  # type: ignore[arg-type]
    assert 0.001 < one < 0.05
    assert ten == pytest.approx(one * 10, rel=0.01)
    imp = total(estimate_import(settings, reg, [Chapter(i, "t", "w " * 2500) for i in range(500)]))  # type: ignore[arg-type]
    assert 0.05 < imp < 2.0
    assert json.dumps(one)
