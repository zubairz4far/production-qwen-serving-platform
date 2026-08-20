from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    query_count: int


def evaluate_retrieval(
    expected: list[set[str]],
    retrieved: list[list[str]],
    *,
    k: int = 5,
) -> RetrievalMetrics:
    """Measure retrieval recall, hit rate and MRR on a frozen query set."""

    if len(expected) != len(retrieved):
        raise ValueError("expected and retrieved must have the same length")
    if k <= 0:
        raise ValueError("k must be positive")
    if not expected:
        return RetrievalMetrics(0.0, 0.0, 0.0, 0)

    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []

    for relevant, ranking in zip(expected, retrieved, strict=True):
        top_k = ranking[:k]
        matched = relevant.intersection(top_k)
        recalls.append(len(matched) / len(relevant) if relevant else 1.0)
        hits.append(1.0 if matched else 0.0)

        first_relevant_rank = next(
            (rank for rank, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks.append(1.0 / first_relevant_rank if first_relevant_rank else 0.0)

    return RetrievalMetrics(
        recall_at_k=mean(recalls),
        hit_rate_at_k=mean(hits),
        mrr=mean(reciprocal_ranks),
        query_count=len(expected),
    )
