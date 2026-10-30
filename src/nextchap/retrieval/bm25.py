"""Small Okapi BM25 for keyword recall of names, plus reciprocal rank fusion."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"[\w'-]+", re.UNICODE)
_STOP = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "his",
        "i",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "she",
        "so",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "when",
        "which",
        "who",
        "will",
        "with",
        "you",
        "your",
        "not",
        "no",
        "yes",
    ]
)


def tokenize(text: str) -> list[str]:
    return [t for t in (m.lower() for m in _TOKEN_RE.findall(text)) if t not in _STOP]


class BM25:
    def __init__(self, docs: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.tfs = [Counter(tokenize(d)) for d in docs]
        self.lens = [sum(tf.values()) for tf in self.tfs]
        self.avg = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter[str] = Counter()
        for tf in self.tfs:
            df.update(tf.keys())
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        terms = tokenize(query)
        out = []
        for tf, length in zip(self.tfs, self.lens, strict=True):
            s = 0.0
            for t in terms:
                f = tf.get(t)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * length / (self.avg or 1))
                s += self.idf.get(t, 0.0) * f * (self.k1 + 1) / denom
            out.append(s)
        return out

    def top(self, query: str, k: int) -> list[int]:
        scored = [(s, i) for i, s in enumerate(self.scores(query)) if s > 0]
        scored.sort(reverse=True)
        return [i for _, i in scored[:k]]


def rrf(rankings: list[list[str]], weights: list[float], k: int = 60) -> list[tuple[str, float]]:
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranking):
            fused[item] = fused.get(item, 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
