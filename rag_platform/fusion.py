from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

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


def diversify_by_source(
    hits: Sequence[RetrievalHit],
    *,
    limit: int = 10,
    max_per_source: int = 2,
) -> list[RetrievalHit]:
    """Prefer source diversity while preserving the upstream ranking order.

    The first pass admits at most ``max_per_source`` chunks from one source. If the
    corpus does not contain enough distinct sources to fill ``limit``, deferred hits
    are appended in their original order. This avoids turning diversity into a hard
    recall loss on small or single-source corpora.
    """

    if limit <= 0:
        return []
    if max_per_source <= 0:
        raise ValueError("max_per_source must be positive")

    selected: list[RetrievalHit] = []
    deferred: list[RetrievalHit] = []
    counts: dict[str, int] = defaultdict(int)

    for hit in hits:
        source = hit.chunk.source
        if counts[source] < max_per_source:
            selected.append(hit)
            counts[source] += 1
        else:
            deferred.append(hit)
        if len(selected) == limit:
            break

    if len(selected) < limit:
        selected.extend(deferred[: limit - len(selected)])

    return [
        RetrievalHit(
            chunk=hit.chunk,
            score=hit.score,
            rank=rank,
            channel=f"{hit.channel}+source_diverse",
        )
        for rank, hit in enumerate(selected[:limit], start=1)
    ]


def source_aware_reciprocal_rank_fusion(
    ranked_lists: list[list[RetrievalHit]],
    *,
    rank_constant: int = 60,
    limit: int = 10,
    max_per_source: int = 2,
) -> list[RetrievalHit]:
    """Run RRF over the full bounded input pool, then diversify by source.

    Each upstream retriever already bounds its candidate list. Source-aware fusion
    must therefore inspect the whole union rather than truncating RRF before the
    diversity pass; otherwise a dominant source can occupy the entire intermediate
    pool and hide relevant chunks from less frequent sources.
    """

    if limit <= 0:
        return []

    unique_candidate_ids = {
        hit.chunk.chunk_id
        for hits in ranked_lists
        for hit in hits
    }
    if not unique_candidate_ids:
        return []

    fused = reciprocal_rank_fusion(
        ranked_lists,
        rank_constant=rank_constant,
        limit=len(unique_candidate_ids),
    )
    return diversify_by_source(
        fused,
        limit=limit,
        max_per_source=max_per_source,
    )
