from __future__ import annotations

from collections import defaultdict

from .types import RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    *,
    rank_constant: int = 60,
    limit: int = 10,
) -> list[RetrievalHit]:
    """Fuse sparse/dense result lists with Reciprocal Rank Fusion (RRF)."""

    if rank_constant <= 0:
        raise ValueError("rank_constant must be positive")
    if limit <= 0:
        return []

    scores: dict[str, float] = defaultdict(float)
    hits_by_id: dict[str, RetrievalHit] = {}

    for hits in ranked_lists:
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit.chunk.chunk_id
            scores[chunk_id] += 1.0 / (rank_constant + rank)
            hits_by_id.setdefault(chunk_id, hit)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:limit]
    return [
        RetrievalHit(
            chunk=hits_by_id[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            channel="rrf",
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
