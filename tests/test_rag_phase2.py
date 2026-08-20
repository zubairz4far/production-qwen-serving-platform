from rag_platform.benchmark import BenchmarkCase, run_benchmark
from rag_platform.grounding import assemble_context, evaluate_citations
from rag_platform.ingestion import canonical_source, chunk_sections, parse_markdown
from rag_platform.rerank import Reranker
from rag_platform.types import Chunk, RetrievalHit


def _hit(
    chunk_id: str,
    text: str,
    source: str,
    score: float = 0.0,
    rank: int = 1,
) -> RetrievalHit:
    return RetrievalHit(Chunk(chunk_id, text, source), score, rank, "test")


def test_markdown_ingestion_preserves_heading_path():
    sections = parse_markdown("# Root\nintro\n## Child\ndetail", source="doc.md")
    assert [section.metadata["heading_path"] for section in sections] == [
        ["Root"],
        ["Root", "Child"],
    ]
    chunks = chunk_sections(sections, chunk_size_words=20, overlap_words=2)
    assert chunks[1].source.startswith("doc.md#heading=Root > Child")
    assert canonical_source(chunks[1].source) == "doc.md"


def test_reranker_reorders_candidates():
    class Scorer:
        def predict(self, pairs):
            return [0.1, 0.9]

    hits = [
        _hit("a", "alpha", "a.md", rank=1),
        _hit("b", "beta", "b.md", rank=2),
    ]
    ranked = Reranker(Scorer()).rerank("query", hits)
    assert [hit.chunk.chunk_id for hit in ranked] == ["b", "a"]
    assert [hit.rank for hit in ranked] == [1, 2]


def test_context_and_citation_metrics():
    hits = [
        _hit("a", "alpha", "a.md#heading=A"),
        _hit("b", "beta", "b.md#page=2", rank=2),
    ]
    bundle = assemble_context(hits)
    metrics = evaluate_citations(
        "Supported by [S1] and bad [S9].",
        bundle,
        expected_sources={"a.md"},
    )
    assert metrics.citation_precision == 0.5
    assert metrics.expected_source_coverage == 1.0


def test_source_level_benchmark_survives_chunk_locator_changes():
    class Retriever:
        def search(self, query, *, limit=5):
            return [_hit("x", "x", "guide.md#heading=Install")]

    result = run_benchmark(
        "demo",
        Retriever(),
        [BenchmarkCase("install", frozenset({"guide.md"}))],
    )
    assert result.metrics.recall_at_k == 1.0
    assert result.metrics.hit_rate_at_k == 1.0
    assert result.metrics.mrr == 1.0
