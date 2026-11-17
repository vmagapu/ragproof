"""Grounded answering: cite or refuse.

The contract: every answer is assembled from numbered context chunks, must
cite them, and when retrieval is too weak to support an answer the system
refuses BEFORE the model is called. An unsupported answer is a worse
failure than no answer — and the refusal is also cheaper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Sequence

from .retrieval import Hit

Model = Callable[[str], str]

_PROMPT = """Answer the question using ONLY the numbered context below.

Rules:
- Cite every claim with its source number in square brackets, e.g. [1] or [2][3].
- If the context does not contain the answer, reply exactly: INSUFFICIENT CONTEXT
- Do not use knowledge outside the context.

[CONTEXT]
{context}

[QUESTION]
{question}

[ANSWER]"""

_CITATION = re.compile(r"\[(\d+)\]")


@dataclass(frozen=True)
class GroundedAnswer:
    answered: bool
    text: str
    citations: tuple[str, ...]      # chunk ids actually cited
    refusal_reason: str | None = None


def format_context(hits: Sequence[Hit]) -> str:
    return "\n\n".join(f"[{i}] ({h.chunk.id}) {h.chunk.text}" for i, h in enumerate(hits, 1))


def answer(
    question: str,
    hits: Sequence[Hit],
    model: Model,
    min_hits: int = 1,
    min_top_score: float = 0.0,
) -> GroundedAnswer:
    """The retrieval gate runs first: too few hits, or a top score below the
    floor you calibrated for your index, means refuse without spending a
    model call. Then the citation contract is ENFORCED, not requested:
    an answer with no valid citations is converted to a refusal."""
    if not hits or len(hits) < min_hits or hits[0].score < min_top_score:
        top = f"{hits[0].score:.4f}" if hits else "n/a"
        return GroundedAnswer(
            answered=False,
            text="",
            citations=(),
            refusal_reason=f"retrieval too weak: {len(hits)} hits, top score {top}",
        )

    raw = model(_PROMPT.format(context=format_context(hits), question=question)).strip()

    if raw.upper().startswith("INSUFFICIENT CONTEXT"):
        return GroundedAnswer(False, "", (), refusal_reason="model: insufficient context")

    cited_numbers = {int(n) for n in _CITATION.findall(raw)}
    valid = {n for n in cited_numbers if 1 <= n <= len(hits)}
    if not valid:
        return GroundedAnswer(
            False, "", (), refusal_reason="answer carried no valid citations — rejected"
        )

    return GroundedAnswer(
        answered=True,
        text=raw,
        citations=tuple(hits[n - 1].chunk.id for n in sorted(valid)),
    )
