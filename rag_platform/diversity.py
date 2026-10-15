from __future__ import annotations

from .fusion import diversify_by_source
from .types import RetrievalHit


class SourceDiversityRetriever:
    """Apply source diversification to the final ranking of any retriever."""

    def __init__(
        self,
        *,
        retriever,
        candidate_limit: int = 20,
        max_per_source: int = 2,
    ) -> None:
        if candidate_limit <= 0:
            raise ValueError("candidate_limit must be positive")
        if max_per_source <= 0:
            raise ValueError("max_per_source must be positive")
        self.retriever = retriever
        self.candidate_limit = candidate_limit
        self.max_per_source = max_per_source

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        candidates = self.retriever.search(
            query,
            limit=max(limit, self.candidate_limit),
        )
        return diversify_by_source(
            candidates,
            limit=limit,
            max_per_source=self.max_per_source,
        )
