"""Retrieval: lexical, vector, and hybrid fusion.

BM25 is implemented here in pure Python — it is the baseline every vector
index must beat, and you can't demand that comparison if standing it up
costs a dependency. Embedders are callables you supply, so the vector side
is provider-agnostic and tests run on deterministic fakes.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Sequence

from .corpus import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class BM25Index:
    """Okapi BM25 (k1/b standard defaults)."""

    def __init__(self, chunks: Sequence[Chunk], k1: float = 1.5, b: float = 0.75):
        if not chunks:
            raise ValueError("empty corpus")
        self.chunks = list(chunks)
        self.k1, self.b = k1, b
        self._tf: list[Counter] = [Counter(tokenize(c.text)) for c in self.chunks]
        self._doc_len = [sum(tf.values()) for tf in self._tf]
        self._avg_len = sum(self._doc_len) / len(self._doc_len)
        df: Counter = Counter()
        for tf in self._tf:
            df.update(tf.keys())
        n = len(self.chunks)
        # BM25+-style floor keeps idf non-negative for very common terms.
        self._idf = {
            term: max(0.0, math.log((n - d + 0.5) / (d + 0.5) + 1.0))
            for term, d in df.items()
        }

    def search(self, query: str, k: int = 10) -> list[Hit]:
        q_terms = tokenize(query)
        scores: list[float] = []
        for i, tf in enumerate(self._tf):
            s = 0.0
            norm = self.k1 * (1 - self.b + self.b * self._doc_len[i] / self._avg_len)
            for term in q_terms:
                f = tf.get(term, 0)
                if f:
                    s += self._idf.get(term, 0.0) * (f * (self.k1 + 1)) / (f + norm)
            scores.append(s)
        ranked = sorted(range(len(scores)), key=lambda i: (-scores[i], self.chunks[i].id))
        return [Hit(self.chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]


Embedder = Callable[[list[str]], list[list[float]]]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class VectorIndex:
    """Exact cosine search over an embedder you supply. Exact beats
    approximate until your corpus proves otherwise — measure first."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        if not chunks:
            raise ValueError("empty corpus")
        self.chunks = list(chunks)
        self._embedder = embedder
        self._vectors = embedder([c.text for c in self.chunks])

    def search(self, query: str, k: int = 10) -> list[Hit]:
        qv = self._embedder([query])[0]
        scored = [(_cosine(qv, v), c) for v, c in zip(self._vectors, self.chunks)]
        scored.sort(key=lambda t: (-t[0], t[1].id))
        return [Hit(c, s) for s, c in scored[:k]]


def reciprocal_rank_fusion(
    result_lists: Sequence[Sequence[Hit]], k: int = 10, rrf_k: int = 60
) -> list[Hit]:
    """Combine ranked lists by RRF — robust to incomparable score scales,
    which is exactly the situation when fusing BM25 with cosine."""
    fused: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for hits in result_lists:
        for rank, hit in enumerate(hits, 1):
            fused[hit.chunk.id] = fused.get(hit.chunk.id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[hit.chunk.id] = hit.chunk
    ranked = sorted(fused.items(), key=lambda t: (-t[1], t[0]))
    return [Hit(by_id[cid], score) for cid, score in ranked[:k]]


class HybridRetriever:
    """BM25 + vectors, fused. The default architecture worth measuring
    against — lexical catches exact terms (names, codes, numbers), vectors
    catch paraphrase; each covers the other's blind spot."""

    def __init__(self, chunks: Sequence[Chunk], embedder: Embedder):
        self.bm25 = BM25Index(chunks)
        self.vectors = VectorIndex(chunks, embedder)

    def search(self, query: str, k: int = 10) -> list[Hit]:
        return reciprocal_rank_fusion(
            [self.bm25.search(query, k=k * 2), self.vectors.search(query, k=k * 2)], k=k
        )
