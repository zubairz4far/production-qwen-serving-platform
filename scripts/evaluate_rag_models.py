from __future__ import annotations

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from rag_platform.benchmark import load_cases, run_benchmark
from rag_platform.bm25 import BM25Index
from rag_platform.dense import QdrantDenseIndex, SentenceTransformerEmbedder
from rag_platform.hybrid import HybridRetriever
from rag_platform.ingestion import chunk_sections, load_markdown
from rag_platform.rerank import CrossEncoderScorer, Reranker, RerankingRetriever

DEFAULT_SOURCES = (
    "README.md",
    "SECURITY.md",
    "docs/GPU_BENCHMARK.md",
    "docs/RAG_PHASE1.md",
)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


def build_chunks(root: Path):
    chunks = []
    for relative_path in DEFAULT_SOURCES:
        sections = load_markdown(root / relative_path, source=relative_path)
        chunks.extend(
            chunk_sections(
                sections,
                chunk_size_words=180,
                overlap_words=30,
            )
        )
    return chunks


def metrics_payload(result) -> dict:
    return {
        "recall_at_k": result.metrics.recall_at_k,
        "hit_rate_at_k": result.metrics.hit_rate_at_k,
        "top1_accuracy": result.metrics.top1_accuracy,
        "mrr": result.metrics.mrr,
        "query_count": result.metrics.query_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Measure BM25, dense, hybrid, and reranked retrieval on one frozen set."
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/rag_retrieval_v1.jsonl"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--rerank-candidates", type=int, default=20)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    chunks = build_chunks(args.root)
    cases = load_cases(args.root / args.cases)

    sparse = BM25Index(chunks)
    embedder = SentenceTransformerEmbedder(EMBEDDING_MODEL)
    dense = QdrantDenseIndex(
        embedder=embedder,
        collection_name="rag_eval_chunks",
        client=QdrantClient(":memory:"),
    )
    dense.upsert(chunks)
    hybrid = HybridRetriever(sparse=sparse, dense=dense)
    reranked = RerankingRetriever(
        retriever=hybrid,
        reranker=Reranker(CrossEncoderScorer(RERANKER_MODEL)),
        candidate_limit=args.rerank_candidates,
    )

    retrievers = {
        "bm25": sparse,
        "dense": dense,
        "hybrid": hybrid,
        "hybrid_rerank": reranked,
    }
    results = {
        name: run_benchmark(name, retriever, cases, k=args.k)
        for name, retriever in retrievers.items()
    }

    payload = {
        "benchmark": str(args.cases),
        "k": args.k,
        "rerank_candidates": args.rerank_candidates,
        "chunk_size_words": 180,
        "overlap_words": 30,
        "embedding_model": EMBEDDING_MODEL,
        "reranker_model": RERANKER_MODEL,
        "results": {
            name: metrics_payload(result) for name, result in results.items()
        },
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    if args.output:
        output_path = args.root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
