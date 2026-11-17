"""Retrieval metrics — measured separately from answer quality.

When a RAG answer is wrong, the first question is "was the evidence even in
the context?" If not, no prompt will save you. A labeled set of
(query -> relevant chunk ids) makes retrieval a measured component with its
own regression story, independent of the model on top.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from .retrieval import Hit


@dataclass(frozen=True)
class RetrievalCase:
    query: str
    relevant_ids: frozenset[str]  # chunk ids a correct retrieval must surface

    @staticmethod
    def make(query: str, relevant_ids: Sequence[str]) -> "RetrievalCase":
        if not relevant_ids:
            raise ValueError("a retrieval case needs at least one relevant id")
        return RetrievalCase(query, frozenset(relevant_ids))


def recall_at_k(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    got = {h.chunk.id for h in hits[:k]}
    return len(got & relevant) / len(relevant)


def hit_rate_at_k(hits: Sequence[Hit], relevant: frozenset[str], k: int) -> float:
    """1.0 if ANY relevant chunk appears in the top k — the metric that maps
    directly to 'could the model have answered correctly?'"""
    return 1.0 if any(h.chunk.id in relevant for h in hits[:k]) else 0.0


def mrr(hits: Sequence[Hit], relevant: frozenset[str]) -> float:
    for rank, h in enumerate(hits, 1):
        if h.chunk.id in relevant:
            return 1.0 / rank
    return 0.0


@dataclass(frozen=True)
class RetrievalReport:
    n: int
    k: int
    mean_recall_at_k: float
    mean_hit_rate_at_k: float
    mean_mrr: float
    worst_queries: tuple[str, ...]  # queries with zero relevant hits in top k

    def summary(self) -> str:
        s = (
            f"n={self.n}  recall@{self.k}={self.mean_recall_at_k:.3f}  "
            f"hit_rate@{self.k}={self.mean_hit_rate_at_k:.3f}  mrr={self.mean_mrr:.3f}"
        )
        if self.worst_queries:
            s += f"\n  zero-hit queries ({len(self.worst_queries)}): "
            s += "; ".join(self.worst_queries[:5])
        return s


Searcher = Callable[[str, int], Sequence[Hit]]


def evaluate_retrieval(
    search: Searcher, cases: Sequence[RetrievalCase], k: int = 5
) -> RetrievalReport:
    """Run the labeled set through any searcher — BM25, vector, hybrid —
    and get comparable numbers. This is how chunking sizes, indexes, and
    fusion strategies compete: on the same cases, not on anecdotes."""
    if not cases:
        raise ValueError("no retrieval cases")
    recalls, hit_rates, mrrs, worst = [], [], [], []
    for case in cases:
        hits = list(search(case.query, k))
        recalls.append(recall_at_k(hits, case.relevant_ids, k))
        hr = hit_rate_at_k(hits, case.relevant_ids, k)
        hit_rates.append(hr)
        mrrs.append(mrr(hits, case.relevant_ids))
        if hr == 0.0:
            worst.append(case.query)
    n = len(cases)
    return RetrievalReport(
        n=n,
        k=k,
        mean_recall_at_k=sum(recalls) / n,
        mean_hit_rate_at_k=sum(hit_rates) / n,
        mean_mrr=sum(mrrs) / n,
        worst_queries=tuple(worst),
    )
