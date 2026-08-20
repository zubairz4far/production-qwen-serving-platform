import pytest

from rag_platform.bm25 import BM25Index
from rag_platform.chunking import chunk_text
from rag_platform.evaluation import evaluate_retrieval
from rag_platform.fusion import reciprocal_rank_fusion
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
