from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import RetrievalHit


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]: ...


class CrossEncoderScorer:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def predict(self, pairs: Sequence[tuple[str, str]]) -> Sequence[float]:
        return self.model.predict(list(pairs)).tolist()


class Reranker:
    """Rerank first-stage candidates with a query-document pair scorer."""

    def __init__(self, scorer: PairScorer) -> None:
        self.scorer = scorer

    def rerank(
        self,
        query: str,
        hits: Sequence[RetrievalHit],
        *,
        limit: int = 10,
    ) -> list[RetrievalHit]:
        if limit <= 0 or not hits:
            return []
        scores = self.scorer.predict([(query, hit.chunk.text) for hit in hits])
        if len(scores) != len(hits):
            raise ValueError("reranker returned a different number of scores than candidates")
        ordered = sorted(
            zip(hits, scores, strict=True),
            key=lambda item: float(item[1]),
            reverse=True,
        )
        return [
            RetrievalHit(
                chunk=hit.chunk,
                score=float(score),
                rank=rank,
                channel="rerank",
            )
            for rank, (hit, score) in enumerate(ordered[:limit], start=1)
        ]


class RerankingRetriever:
    def __init__(self, *, retriever, reranker: Reranker, candidate_limit: int = 30) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.candidate_limit = candidate_limit

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        candidates = self.retriever.search(query, limit=self.candidate_limit)
        return self.reranker.rerank(query, candidates, limit=limit)
