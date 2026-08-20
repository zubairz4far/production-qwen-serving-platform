from __future__ import annotations

import hashlib

from .types import Chunk


def _stable_chunk_id(source: str, index: int, text: str) -> str:
    digest = hashlib.sha256(f"{source}\n{index}\n{text}".encode()).hexdigest()[:16]
    return f"{source}:{index}:{digest}"


def chunk_text(
    text: str,
    *,
    source: str,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Split text into deterministic overlapping word windows."""

    if chunk_size_words <= 0:
        raise ValueError("chunk_size_words must be positive")
    if overlap_words < 0:
        raise ValueError("overlap_words cannot be negative")
    if overlap_words >= chunk_size_words:
        raise ValueError("overlap_words must be smaller than chunk_size_words")

    words = text.split()
    if not words:
        return []

    step = chunk_size_words - overlap_words
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + chunk_size_words]
        if not window:
            break
        chunk_text_value = " ".join(window)
        chunks.append(
            Chunk(
                chunk_id=_stable_chunk_id(source, index, chunk_text_value),
                text=chunk_text_value,
                source=source,
                metadata={
                    "chunk_index": index,
                    "start_word": start,
                    "end_word": start + len(window),
                },
            )
        )
        if start + chunk_size_words >= len(words):
            break
    return chunks
