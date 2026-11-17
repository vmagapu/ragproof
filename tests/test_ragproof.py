"""Deterministic throughout: hash-based fake embedder, scripted models, no network."""

import hashlib
import math

import pytest

from ragproof.answer import GroundedAnswer, answer
from ragproof.corpus import chunk_corpus, chunk_document
from ragproof.metrics import RetrievalCase, evaluate_retrieval, mrr, recall_at_k
from ragproof.retrieval import (
    BM25Index,
    Hit,
    HybridRetriever,
    VectorIndex,
    reciprocal_rank_fusion,
    tokenize,
)


def fake_embedder(texts):
    """Deterministic 'embeddings': token-hash buckets. Texts sharing words
    land near each other — enough structure to test ranking honestly."""
    dim = 32
    out = []
    for t in texts:
        v = [0.0] * dim
        for tok in tokenize(t):
            h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
            v[h % dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        out.append([x / norm for x in v])
    return out


DOCS = {
    "loans": "The mortgage payment on a fixed loan is constant. Amortization "
    "schedules show interest and principal per month.\n\nEarly payments are "
    "mostly interest; later payments are mostly principal.",
    "evals": "Evaluation gates in CI stop regressions in model behavior. "
    "A judge must be calibrated against human labels before its scores are trusted.",
    "retire": "Retirement projections should always be shown in today's dollars. "
    "Inflation adjustment converts nominal futures into real purchasing power.",
}
CHUNKS = chunk_corpus(DOCS, max_words=30, overlap_words=5)


# ---------- chunking ----------

def test_chunking_is_deterministic_and_overlapping():
    a = chunk_document("d", "one two three four five six seven eight", max_words=4, overlap_words=2)
    b = chunk_document("d", "one two three four five six seven eight", max_words=4, overlap_words=2)
    assert [c.text for c in a] == [c.text for c in b]
    assert a[0].text == "one two three four"
    assert a[1].text.startswith("three four")  # overlap preserves the boundary
    assert [c.id for c in a] == ["d#0", "d#1", "d#2"]


def test_chunking_rejects_bad_params():
    with pytest.raises(ValueError):
        chunk_document("d", "x", max_words=10, overlap_words=10)


def test_empty_document_yields_no_chunks():
    assert chunk_document("d", "   \n\n  ") == []


# ---------- BM25 ----------

def test_bm25_ranks_on_topic_chunk_first():
    idx = BM25Index(CHUNKS)
    hits = idx.search("mortgage amortization interest")
    assert hits and hits[0].chunk.doc_id == "loans"


def test_bm25_returns_nothing_for_alien_query():
    idx = BM25Index(CHUNKS)
    assert idx.search("zebra quantum volcano") == []


def test_bm25_idf_floor_nonnegative():
    idx = BM25Index(CHUNKS)
    assert all(v >= 0 for v in idx._idf.values())


# ---------- vectors & fusion ----------

def test_vector_index_finds_shared_vocabulary():
    idx = VectorIndex(CHUNKS, fake_embedder)
    hits = idx.search("calibrated judge labels", k=3)
    assert hits[0].chunk.doc_id == "evals"


def test_rrf_rewards_agreement():
    c1, c2, c3 = CHUNKS[0], CHUNKS[1], CHUNKS[2]
    list_a = [Hit(c1, 10.0), Hit(c2, 5.0)]
    list_b = [Hit(c1, 0.9), Hit(c3, 0.8)]
    fused = reciprocal_rank_fusion([list_a, list_b], k=3)
    assert fused[0].chunk.id == c1.id  # ranked #1 by both lists


def test_hybrid_covers_lexical_blind_spot():
    retriever = HybridRetriever(CHUNKS, fake_embedder)
    hits = retriever.search("inflation purchasing power")
    assert hits and hits[0].chunk.doc_id == "retire"


# ---------- metrics ----------

def test_recall_and_mrr_math():
    rel = frozenset({"a", "b"})
    hits = [Hit(c, 1.0) for c in CHUNKS[:3]]
    assert recall_at_k(hits, rel, 3) == 0.0
    assert mrr(hits, frozenset({CHUNKS[1].id})) == 0.5  # found at rank 2


def test_evaluate_retrieval_end_to_end():
    idx = BM25Index(CHUNKS)
    loans_ids = [c.id for c in CHUNKS if c.doc_id == "loans"]
    evals_ids = [c.id for c in CHUNKS if c.doc_id == "evals"]
    cases = [
        RetrievalCase.make("mortgage amortization schedule", loans_ids),
        RetrievalCase.make("judge calibration human labels", evals_ids),
        RetrievalCase.make("zebra quantum volcano", ["loans#0"]),  # guaranteed miss
    ]
    report = evaluate_retrieval(lambda q, k: idx.search(q, k), cases, k=3)
    assert report.mean_hit_rate_at_k == pytest.approx(2 / 3)
    assert report.worst_queries == ("zebra quantum volcano",)


# ---------- grounded answering ----------

def _hits(n=2):
    return [Hit(CHUNKS[i], 5.0 - i) for i in range(n)]


def test_refuses_before_model_call_when_retrieval_weak():
    calls = {"n": 0}

    def model(prompt):
        calls["n"] += 1
        return "should never run"

    result = answer("q", [], model)
    assert not result.answered and calls["n"] == 0

    result = answer("q", _hits(), model, min_top_score=100.0)
    assert not result.answered and calls["n"] == 0
    assert "retrieval too weak" in result.refusal_reason


def test_enforces_citation_contract():
    uncited = answer("q", _hits(), lambda p: "Confident answer with no citations.")
    assert not uncited.answered
    assert "no valid citations" in uncited.refusal_reason

    cited = answer("q", _hits(), lambda p: "The payment is constant [1].")
    assert cited.answered
    assert cited.citations == (CHUNKS[0].id,)


def test_model_refusal_is_respected():
    result = answer("q", _hits(), lambda p: "INSUFFICIENT CONTEXT")
    assert not result.answered
    assert result.refusal_reason == "model: insufficient context"


def test_out_of_range_citations_rejected():
    result = answer("q", _hits(2), lambda p: "Made up a source [7].")
    assert not result.answered


def test_context_contains_numbered_chunk_ids():
    seen = {}

    def model(prompt):
        seen["prompt"] = prompt
        return "ok [1]"

    answer("q", _hits(), model)
    assert f"[1] ({CHUNKS[0].id})" in seen["prompt"]
