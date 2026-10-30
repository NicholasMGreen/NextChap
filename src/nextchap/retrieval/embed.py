"""Embedding backends: local sentence-transformers, remote (OpenRouter), and a test hash."""

from __future__ import annotations

import hashlib
import math
from typing import Any

from nextchap.config import Settings
from nextchap.errors import ConfigError
from nextchap.llm.base import Embedder
from nextchap.retrieval.bm25 import tokenize

QUERY_INSTRUCTION = (
    "Given a story passage request, retrieve the scenes of the novel that are most relevant"
)


class LocalEmbedder:
    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ConfigError(
                "Local embeddings need the 'local-embed' extra.",
                hint='uv sync --extra local-embed   (or set retrieval.embed_backend = "remote")',
            ) from exc
        self.name = model_name
        self._model: Any = SentenceTransformer(model_name)
        self._is_qwen = "qwen3-embedding" in model_name.lower()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self._model.encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False
        )
        return [list(map(float, v)) for v in vecs]

    def embed_query(self, text: str) -> list[float]:
        if self._is_qwen:
            text = f"Instruct: {QUERY_INSTRUCTION}\nQuery: {text}"
        vec = self._model.encode([text], normalize_embeddings=True, show_progress_bar=False)[0]
        return list(map(float, vec))


class RemoteEmbedder:
    BATCH = 64

    def __init__(self, client: Any, model: str) -> None:
        self.name = f"remote:{model}"
        self._client = client
        self._model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH):
            out.extend(
                _normalize(v) for v in self._client.embed(self._model, texts[i : i + self.BATCH])
            )
        return out

    def embed_query(self, text: str) -> list[float]:
        return _normalize(self._client.embed(self._model, [text])[0])


class HashEmbedder:
    """Deterministic bag-of-words hashing embedder. For tests and offline smoke runs only."""

    def __init__(self, dim: int = 256) -> None:
        self.name = f"hash-{dim}"
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for tok in tokenize(text):
            h = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
            v[h % self.dim] += 1.0 if (h >> 63) else -1.0
        return _normalize(v)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vec(text)


def _normalize(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def make_embedder(settings: Settings, client: Any | None = None) -> Embedder:
    r = settings.retrieval
    if r.embed_backend == "hash":
        return HashEmbedder()
    if r.embed_backend == "remote":
        if client is None:
            raise ConfigError("Remote embeddings need an API key.")
        return RemoteEmbedder(client, r.remote_embed_model)
    return LocalEmbedder(r.embed_model)
