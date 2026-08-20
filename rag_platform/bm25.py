from __future__ import annotations

import math
import re
from collections import Counter

from .types import Chunk, RetrievalHit

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


class BM25Index:
    """Small transparent BM25 implementation used as the sparse baseline."""

    def __init__(self, chunks: list[Chunk], *, k1: float = 1.5, b: float = 0.75) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in self.chunks]
        self.doc_term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = (
            sum(self.doc_lengths) / len(self.doc_lengths) if self.doc_lengths else 0.0
        )

        doc_freq: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            doc_freq.update(set(tokens))
        self.doc_freq = doc_freq

    def _idf(self, term: str) -> float:
        n = len(self.chunks)
        df = self.doc_freq.get(term, 0)
        return math.log(1.0 + (n - df + 0.5) / (df + 0.5))

    def score(self, query: str, doc_index: int) -> float:
        if not self.chunks or self.avg_doc_length == 0:
            return 0.0

        score = 0.0
        frequencies = self.doc_term_freqs[doc_index]
        doc_length = self.doc_lengths[doc_index]
        for term in tokenize(query):
            tf = frequencies.get(term, 0)
            if tf == 0:
                continue
            denominator = tf + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += self._idf(term) * (tf * (self.k1 + 1)) / denominator
        return score

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        if limit <= 0:
            return []

        scored = [(index, self.score(query, index)) for index in range(len(self.chunks))]
        scored = [item for item in scored if item[1] > 0]
        scored.sort(key=lambda item: (-item[1], self.chunks[item[0]].chunk_id))

        return [
            RetrievalHit(
                chunk=self.chunks[index],
                score=score,
                rank=rank,
                channel="bm25",
            )
            for rank, (index, score) in enumerate(scored[:limit], start=1)
        ]
