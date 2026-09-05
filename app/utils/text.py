"""Text splitting for retrieval.

Chunks are cut on paragraph and sentence boundaries rather than at a fixed
character offset. Slicing mid-sentence - as this did previously - strands the
subject of a clause in one chunk and its object in another, which both weakens
the embedding and produces quoted context that reads as truncated nonsense.
"""
from __future__ import annotations

import re

# Split after ., ! or ? when followed by whitespace, keeping the punctuation.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(paragraph: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_END.split(paragraph) if s.strip()]


def _hard_split(text: str, size: int) -> list[str]:
    """Last resort for a single sentence longer than the chunk size."""
    return [text[i : i + size] for i in range(0, len(text), size)]


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks that end on sentence boundaries.

    `overlap` carries whole trailing sentences from the previous chunk into the
    next one, so a fact that straddles a boundary stays retrievable from both.
    """
    if not text.strip():
        return []

    units: list[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _split_sentences(paragraph):
            units.extend(_hard_split(sentence, chunk_size) if len(sentence) > chunk_size else [sentence])

    chunks: list[str] = []
    current: list[str] = []
    length = 0

    for unit in units:
        # +1 for the space joining units.
        if current and length + len(unit) + 1 > chunk_size:
            chunks.append(" ".join(current))
            # Carry whole trailing sentences back as the overlap.
            carried: list[str] = []
            carried_len = 0
            for previous in reversed(current):
                if carried_len + len(previous) + 1 > overlap:
                    break
                carried.insert(0, previous)
                carried_len += len(previous) + 1
            current = carried
            length = carried_len
        current.append(unit)
        length += len(unit) + 1

    if current:
        chunks.append(" ".join(current))
    return chunks
