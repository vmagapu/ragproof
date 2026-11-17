# ragproof

**RAG you can measure — because when a grounded answer is wrong, it's usually the retrieval, and "usually" deserves a number.**

[![tests](https://github.com/vmagapu/ragproof/actions/workflows/ci.yml/badge.svg)](https://github.com/vmagapu/ragproof/actions)

Most RAG content stops at "it retrieved something and the answer sounded right." In production — I run RAG at [MoneyCompass](https://getmoneycompass.com) on PostgreSQL + pgvector, and shipped it at marketplace scale before that — the discipline that matters is different: **retrieval is a measured component with its own regression story, and grounding is an enforced contract: cite or refuse.** This repo is that discipline at its smallest honest size.

## The two theses

**1. Measure retrieval separately from answers.** If the evidence never reached the context window, no prompt will save the answer. So retrieval gets its own labeled cases (`query → relevant chunk ids`) and its own metrics — and chunk sizes, indexes, and fusion strategies compete on those numbers, not on anecdotes:

```python
report = evaluate_retrieval(retriever.search, cases, k=5)
print(report.summary())
# n=180  recall@5=0.872  hit_rate@5=0.944  mrr=0.761
#   zero-hit queries (10): "roth conversion deadline"; ...
```

`hit_rate@k` is the one to watch first — it answers "could the model even have gotten this right?" The zero-hit list is your work queue.

**2. Grounding is a contract, enforced.** The answer path refuses *before* the model call when retrieval is too weak (why pay for an answer the context can't support?), requires citations against the numbered context, and **converts an uncited answer into a refusal** — plausible-but-unsupported is treated as the worst failure mode, because in financial software it is:

```python
result = answer(question, hits, model, min_top_score=0.35)
if result.answered:
    print(result.text, result.citations)   # ('policies#3', 'policies#7')
else:
    print(result.refusal_reason)           # "retrieval too weak: 2 hits, top score 0.2100"
```

## What's inside

| Module | What it does |
|---|---|
| `corpus` | Deterministic paragraph-aware chunking with overlap (boundary loss is the quietest retrieval killer) |
| `retrieval` | Pure-Python Okapi BM25 · exact-cosine vector index over any embedder callable · hybrid fusion via reciprocal rank fusion |
| `metrics` | recall@k, hit_rate@k, MRR over labeled cases; comparable across any searcher |
| `answer` | The retrieval gate + the citation contract |

Design choices worth arguing about (please do): **BM25 is implemented, not imported** — it's the baseline every vector index must beat, and you can't demand that comparison if standing it up costs a dependency. **Exact search over approximate** until your corpus proves otherwise — measure first. **RRF for fusion** because BM25 scores and cosine similarities live on incomparable scales, and rank fusion sidesteps the calibration problem entirely. **Embedders are callables** — provider-agnostic, and the tests run on a deterministic hash-based fake: no network, no flakes, 16 tests in 20ms.

## Install

```bash
pip install git+https://github.com/vmagapu/ragproof   # zero dependencies
```

## The family

This is the retrieval leg of a set of small, zero-dependency pieces extracted from the same production philosophy: [finance-engines-mcp](https://github.com/vmagapu/finance-engines-mcp) (deterministic tools for model judgment) · [evalgate](https://github.com/vmagapu/evalgate) (eval gates — point it at `answer()` outputs for the answer-quality side) · [modelgate](https://github.com/vmagapu/modelgate) (the metered model call behind `Model`) · [agentic-sdlc-playbook](https://github.com/vmagapu/agentic-sdlc-playbook) (how an organization adopts all of this). The model gets judgment; deterministic code gets the numbers; everything gets measured.

— [Vish Magapu](https://vmagapu.github.io) · MIT
