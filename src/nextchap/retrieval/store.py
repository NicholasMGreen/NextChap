"""Per-book Chroma index of scene-sized chunks with hybrid (dense + BM25) retrieval."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import chromadb
from chromadb.config import Settings as ChromaSettings

from nextchap.config import RetrievalConfig
from nextchap.errors import IndexMismatch
from nextchap.ingest.master_book import Chapter
from nextchap.llm.base import Embedder
from nextchap.retrieval.bm25 import BM25, rrf

COLLECTION = "chapters"
CHUNKER_VERSION = "scene-v1"
MAX_HITS_PER_CHAPTER = 2
_SCENE_BREAK_RE = re.compile(r"^\s*(?:(?:\*\s*){3,}|(?:-\s*){3,}|(?:#\s*){3}|~{3,}|◇+|◆+)\s*$")


@dataclass
class Chunk:
    id: str
    chapter: int
    title: str
    index: int
    text: str


@dataclass
class Hit:
    id: str
    chapter: int
    title: str
    text: str
    score: float


class Reranker(Protocol):
    def scores(self, query: str, docs: list[str]) -> list[float]: ...


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if len(paras) <= 1 and "\n" in text:
        paras = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return paras


def chunk_chapter(ch: Chapter, words: int = 400, overlap: int = 60) -> list[Chunk]:
    scenes: list[list[str]] = [[]]
    for p in _paragraphs(ch.content):
        if _SCENE_BREAK_RE.match(p):
            if scenes[-1]:
                scenes.append([])
            continue
        if len(p.split()) > words * 2:
            toks = p.split()
            scenes[-1].extend(" ".join(toks[i : i + words]) for i in range(0, len(toks), words))
        else:
            scenes[-1].append(p)

    texts: list[str] = []
    for scene in scenes:
        buf: list[str] = []
        count = 0
        fresh = False
        for p in scene:
            buf.append(p)
            count += len(p.split())
            fresh = True
            if count >= words:
                texts.append("\n\n".join(buf))
                carry: list[str] = []
                carried = 0
                for q in reversed(buf):
                    n = len(q.split())
                    if carried + n > overlap:
                        break
                    carry.insert(0, q)
                    carried += n
                buf, count, fresh = carry, carried, False
        if buf and fresh:
            texts.append("\n\n".join(buf))

    return [
        Chunk(f"ch{ch.number:05d}-s{i:03d}", ch.number, ch.title, i, t)
        for i, t in enumerate(t for t in texts if t.strip())
    ]


class VectorStore:
    def __init__(
        self, path: Path, embedder: Embedder, cfg: RetrievalConfig, reranker: Reranker | None = None
    ) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.embedder = embedder
        self.cfg = cfg
        self.reranker = reranker
        self._client = chromadb.PersistentClient(
            path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        expected = {"embed_model": embedder.name, "chunker": CHUNKER_VERSION}
        self._col: Any = self._client.get_or_create_collection(
            COLLECTION, metadata=expected, embedding_function=None
        )
        actual = self._col.metadata or {}
        for key, value in expected.items():
            if actual.get(key) != value:
                raise IndexMismatch(
                    f"Index was built with {key}={actual.get(key)!r}, settings say {value!r}."
                )
        self._bm25: tuple[BM25, list[str], list[str], list[dict[str, Any]]] | None = None

    @staticmethod
    def reset(path: Path) -> None:
        # Chroma caches one client per path per process; dropping the collection through it
        # avoids stale SQLite handles that deleting the directory would leave behind.
        if not path.exists():
            return
        try:
            client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
            if any(c.name == COLLECTION for c in client.list_collections()):
                client.delete_collection(COLLECTION)
        except Exception:
            from chromadb.api.shared_system_client import SharedSystemClient

            shutil.rmtree(path, ignore_errors=True)
            SharedSystemClient.clear_system_cache()

    @staticmethod
    def inspect(path: Path) -> tuple[int, dict[str, Any]] | None:
        """Read index metadata without constructing an embedder (for status output)."""
        if not path.exists():
            return None
        try:
            client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
            col = client.get_collection(COLLECTION, embedding_function=None)
        except Exception:
            return None
        return int(col.count()), dict(col.metadata or {})

    def count(self) -> int:
        return int(self._col.count())

    def indexed(self) -> dict[int, str]:
        """chapter number -> content hash of what is indexed."""
        got = self._col.get(include=["metadatas"])
        out: dict[int, str] = {}
        for meta in got.get("metadatas") or []:
            if meta:
                out[int(meta["chapter"])] = str(meta.get("hash", ""))
        return out

    def upsert_chapters(self, chapters: list[Chapter]) -> int:
        added = 0
        for ch in chapters:
            chunks = chunk_chapter(ch, self.cfg.chunk_words, self.cfg.chunk_overlap_words)
            self._col.delete(where={"chapter": ch.number})
            if not chunks:
                continue
            embeddings = self.embedder.embed_documents([c.text for c in chunks])
            self._col.add(
                ids=[c.id for c in chunks],
                embeddings=embeddings,
                documents=[c.text for c in chunks],
                metadatas=[
                    {
                        "chapter": c.chapter,
                        "title": c.title,
                        "index": c.index,
                        "hash": ch.content_hash,
                    }
                    for c in chunks
                ],
            )
            added += len(chunks)
        self._bm25 = None
        return added

    def delete_chapters(self, numbers: list[int]) -> None:
        for n in numbers:
            self._col.delete(where={"chapter": n})
        self._bm25 = None

    def _lexical(self) -> tuple[BM25, list[str], list[str], list[dict[str, Any]]]:
        if self._bm25 is None:
            got = self._col.get(include=["documents", "metadatas"])
            ids = list(got.get("ids") or [])
            docs = [d or "" for d in (got.get("documents") or [])]
            metas = [m or {} for m in (got.get("metadatas") or [])]
            self._bm25 = (BM25(docs), ids, docs, metas)
        return self._bm25

    def query(
        self, text: str, top_k: int | None = None, exclude_chapters: set[int] | None = None
    ) -> list[Hit]:
        top_k = top_k or self.cfg.top_k
        total = self.count()
        if total == 0 or not text.strip():
            return []
        exclude = exclude_chapters or set()
        n = min(self.cfg.candidates, total)
        where = {"chapter": {"$nin": sorted(exclude)}} if exclude else None
        dense = self._col.query(
            query_embeddings=[self.embedder.embed_query(text)],
            n_results=n,
            where=where,
            include=["documents", "metadatas"],
        )
        pool: dict[str, tuple[str, dict[str, Any]]] = {}
        dense_ids = list(dense["ids"][0])
        for cid, doc, meta in zip(
            dense_ids, dense["documents"][0], dense["metadatas"][0], strict=True
        ):
            pool[cid] = (doc or "", meta or {})

        rankings, weights = [dense_ids], [self.cfg.dense_weight]
        if self.cfg.hybrid:
            bm25, ids, docs, metas = self._lexical()
            lexical = []
            for i in bm25.top(text, n * 2):
                if int(metas[i].get("chapter", -1)) in exclude:
                    continue
                lexical.append(ids[i])
                pool.setdefault(ids[i], (docs[i], metas[i]))
                if len(lexical) >= n:
                    break
            rankings.append(lexical)
            weights.append(1 - self.cfg.dense_weight)
        fused = rrf(rankings, weights)

        if self.reranker is not None and fused:
            cand = [cid for cid, _ in fused[:n]]
            scores = self.reranker.scores(text, [pool[c][0] for c in cand])
            fused = sorted(zip(cand, scores, strict=True), key=lambda kv: kv[1], reverse=True)

        hits: list[Hit] = []
        per_chapter: dict[int, int] = {}
        for cid, score in fused:
            doc, meta = pool[cid]
            chapter = int(meta.get("chapter", 0))
            if per_chapter.get(chapter, 0) >= MAX_HITS_PER_CHAPTER:
                continue
            per_chapter[chapter] = per_chapter.get(chapter, 0) + 1
            hits.append(Hit(cid, chapter, str(meta.get("title", "")), doc, float(score)))
            if len(hits) >= top_k:
                break
        return hits
