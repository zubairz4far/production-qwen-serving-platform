from __future__ import annotations

from typing import Protocol

from .fusion import reciprocal_rank_fusion
from .types import RetrievalHit


class Retriever(Protocol):
    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]: ...


class HybridRetriever:
    """Combine dense and sparse retrieval using reciprocal-rank fusion."""

    def __init__(self, *, sparse: Retriever, dense: Retriever) -> None:
        self.sparse = sparse
        self.dense = dense

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        candidate_limit: int = 30,
    ) -> list[RetrievalHit]:
        sparse_hits = self.sparse.search(query, limit=candidate_limit)
        dense_hits = self.dense.search(query, limit=candidate_limit)
        return reciprocal_rank_fusion([sparse_hits, dense_hits], limit=limit)
