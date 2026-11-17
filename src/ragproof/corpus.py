"""Documents and chunking.

Chunking is where most RAG quality is won or lost, and it is pure,
deterministic string processing — so it should be tested like it matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    id: str          # "<doc_id>#<n>" — stable, citable
    doc_id: str
    text: str
    position: int    # 0-based order within the document


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")
_WORD = re.compile(r"\S+")


def chunk_document(
    doc_id: str,
    text: str,
    max_words: int = 200,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Paragraph-aware sliding window.

    Paragraphs are packed into windows of at most `max_words`; a paragraph
    longer than the window is split on word boundaries. Consecutive chunks
    overlap by `overlap_words` so facts near a boundary appear whole in at
    least one chunk — boundary loss is the quietest retrieval killer.
    """
    if max_words <= 0 or not (0 <= overlap_words < max_words):
        raise ValueError("require max_words > 0 and 0 <= overlap_words < max_words")

    words: list[str] = []
    for para in _PARAGRAPH_SPLIT.split(text.strip()):
        para = para.strip()
        if para:
            words.extend(_WORD.findall(para))
            words.append("\n")  # soft paragraph marker, kept out of counts below

    tokens = [w for w in words if w != "\n"]
    if not tokens:
        return []

    chunks: list[Chunk] = []
    step = max_words - overlap_words
    start = 0
    position = 0
    while start < len(tokens):
        window = tokens[start : start + max_words]
        chunks.append(
            Chunk(
                id=f"{doc_id}#{position}",
                doc_id=doc_id,
                text=" ".join(window),
                position=position,
            )
        )
        if start + max_words >= len(tokens):
            break
        start += step
        position += 1
    return chunks


def chunk_corpus(docs: dict[str, str], max_words: int = 200, overlap_words: int = 40) -> list[Chunk]:
    out: list[Chunk] = []
    for doc_id in sorted(docs):  # deterministic order
        out.extend(chunk_document(doc_id, docs[doc_id], max_words, overlap_words))
    return out
