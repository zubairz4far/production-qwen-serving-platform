import pytest

from rag_platform.bm25 import BM25Index
from rag_platform.chunking import chunk_text
from rag_platform.evaluation import evaluate_retrieval
from rag_platform.fusion import (
    diversify_by_source,
    reciprocal_rank_fusion,
    source_aware_reciprocal_rank_fusion,
)
from rag_platform.hybrid import SourceAwareHybridRetriever
from rag_platform.types import Chunk, RetrievalHit


def test_chunking_is_overlapping_and_deterministic() -> None:
    text = " ".join(f"word{i}" for i in range(12))
    chunks = chunk_text(text, source="demo", chunk_size_words=5, overlap_words=2)
    repeated = chunk_text(text, source="demo", chunk_size_words=5, overlap_words=2)

    assert len(chunks) == 4
    assert chunks[0].text == "word0 word1 word2 word3 word4"
    assert chunks[1].text.startswith("word3 word4")
    assert [chunk.chunk_id for chunk in chunks] == [chunk.chunk_id for chunk in repeated]


def test_bm25_ranks_exact_term_match_first() -> None:
    chunks = [
        Chunk("a", "refund policy is thirty days", "policy"),
        Chunk("b", "shipping takes three business days", "shipping"),
        Chunk("c", "returns and refunds require a receipt", "returns"),
    ]
    hits = BM25Index(chunks).search("refund policy", limit=3)

    assert hits[0].chunk.chunk_id == "a"
    assert hits[0].channel == "bm25"


def test_rrf_rewards_items_present_in_multiple_channels() -> None:
    a = Chunk("a", "A", "x")
    b = Chunk("b", "B", "x")
    c = Chunk("c", "C", "x")
    sparse = [
        RetrievalHit(a, 10.0, 1, "bm25"),
        RetrievalHit(b, 9.0, 2, "bm25"),
    ]
    dense = [
        RetrievalHit(c, 0.9, 1, "dense"),
        RetrievalHit(a, 0.8, 2, "dense"),
    ]

    fused = reciprocal_rank_fusion([sparse, dense], rank_constant=10, limit=3)

    assert fused[0].chunk.chunk_id == "a"
    assert fused[0].channel == "rrf"


def test_source_diversity_caps_repeated_sources_before_backfill() -> None:
    hits = [
        RetrievalHit(Chunk("a1", "A1", "README.md"), 1.0, 1, "rerank"),
        RetrievalHit(Chunk("a2", "A2", "README.md"), 0.9, 2, "rerank"),
        RetrievalHit(Chunk("a3", "A3", "README.md"), 0.8, 3, "rerank"),
        RetrievalHit(Chunk("b1", "B1", "SECURITY.md"), 0.7, 4, "rerank"),
        RetrievalHit(Chunk("c1", "C1", "docs/GPU_BENCHMARK.md"), 0.6, 5, "rerank"),
    ]

    diversified = diversify_by_source(hits, limit=4, max_per_source=2)

    assert [hit.chunk.chunk_id for hit in diversified] == ["a1", "a2", "b1", "c1"]
    assert all("source_diverse" in hit.channel for hit in diversified)


def test_source_diversity_groups_heading_locators_by_document() -> None:
    hits = [
        RetrievalHit(
            Chunk("r1", "R1", "README.md#heading=Serving"),
            1.0,
            1,
            "rrf",
        ),
        RetrievalHit(
            Chunk("r2", "R2", "README.md#heading=Benchmarks"),
            0.9,
            2,
            "rrf",
        ),
        RetrievalHit(
            Chunk("r3", "R3", "README.md#heading=Security"),
            0.8,
            3,
            "rrf",
        ),
        RetrievalHit(
            Chunk("g1", "G1", "docs/GPU_BENCHMARK.md#heading=Protocol"),
            0.7,
            4,
            "rrf",
        ),
        RetrievalHit(
            Chunk("s1", "S1", "SECURITY.md#heading=Network"),
            0.6,
            5,
            "rrf",
        ),
    ]

    diversified = diversify_by_source(hits, limit=4, max_per_source=2)

    assert [hit.chunk.chunk_id for hit in diversified] == ["r1", "r2", "g1", "s1"]


def test_source_diversity_backfills_when_corpus_has_too_few_sources() -> None:
    hits = [
        RetrievalHit(Chunk("a1", "A1", "README.md"), 1.0, 1, "rrf"),
        RetrievalHit(Chunk("a2", "A2", "README.md"), 0.9, 2, "rrf"),
        RetrievalHit(Chunk("a3", "A3", "README.md"), 0.8, 3, "rrf"),
    ]

    diversified = diversify_by_source(hits, limit=3, max_per_source=1)

    assert [hit.chunk.chunk_id for hit in diversified] == ["a1", "a2", "a3"]


def test_source_aware_rrf_preserves_top_signal_and_adds_source_coverage() -> None:
    readme_1 = Chunk("r1", "R1", "README.md")
    readme_2 = Chunk("r2", "R2", "README.md")
    readme_3 = Chunk("r3", "R3", "README.md")
    security = Chunk("s1", "S1", "SECURITY.md")
    gpu = Chunk("g1", "G1", "docs/GPU_BENCHMARK.md")
    sparse = [
        RetrievalHit(readme_1, 10.0, 1, "bm25"),
        RetrievalHit(readme_2, 9.0, 2, "bm25"),
        RetrievalHit(readme_3, 8.0, 3, "bm25"),
        RetrievalHit(security, 7.0, 4, "bm25"),
        RetrievalHit(gpu, 6.0, 5, "bm25"),
    ]
    dense = [
        RetrievalHit(readme_1, 0.9, 1, "dense"),
        RetrievalHit(readme_2, 0.8, 2, "dense"),
        RetrievalHit(readme_3, 0.7, 3, "dense"),
        RetrievalHit(gpu, 0.6, 4, "dense"),
        RetrievalHit(security, 0.5, 5, "dense"),
    ]

    fused = source_aware_reciprocal_rank_fusion(
        [sparse, dense],
        rank_constant=10,
        limit=4,
        max_per_source=2,
    )

    assert fused[0].chunk.chunk_id == "r1"
    assert [hit.chunk.source for hit in fused].count("README.md") == 2
    assert {hit.chunk.source for hit in fused} == {
        "README.md",
        "SECURITY.md",
        "docs/GPU_BENCHMARK.md",
    }


class _RecordingRetriever:
    def __init__(self, hits: list[RetrievalHit]) -> None:
        self.hits = hits
        self.last_limit = 0

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        self.last_limit = limit
        return self.hits[:limit]


def test_source_aware_hybrid_uses_configured_deep_candidate_pool() -> None:
    readme_chunks = [Chunk(f"r{i}", f"R{i}", "README.md") for i in range(45)]
    gpu = Chunk("gpu", "GPU", "docs/GPU_BENCHMARK.md")
    sparse_hits = [
        RetrievalHit(chunk, 100.0 - rank, rank, "bm25")
        for rank, chunk in enumerate(readme_chunks, start=1)
    ] + [RetrievalHit(gpu, 1.0, 46, "bm25")]
    dense_hits = [
        RetrievalHit(chunk, 1.0 - rank / 100.0, rank, "dense")
        for rank, chunk in enumerate(readme_chunks, start=1)
    ] + [RetrievalHit(gpu, 0.1, 46, "dense")]
    sparse = _RecordingRetriever(sparse_hits)
    dense = _RecordingRetriever(dense_hits)
    retriever = SourceAwareHybridRetriever(
        sparse=sparse,
        dense=dense,
        max_per_source=2,
        candidate_limit=60,
    )

    hits = retriever.search("gpu benchmark", limit=5)

    assert sparse.last_limit == 60
    assert dense.last_limit == 60
    assert "docs/GPU_BENCHMARK.md" in [hit.chunk.source for hit in hits]


def test_retrieval_metrics() -> None:
    metrics = evaluate_retrieval(
        expected=[{"a"}, {"c", "d"}],
        retrieved=[["b", "a"], ["c", "x", "d"]],
        k=2,
    )

    assert metrics.query_count == 2
    assert metrics.hit_rate_at_k == 1.0
    assert metrics.recall_at_k == pytest.approx(0.75)
    assert metrics.mrr == pytest.approx(0.75)
