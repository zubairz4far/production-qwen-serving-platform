# RAG Phase 2 — Ingestion, Reranking, and Frozen Evaluation

This milestone turns the retrieval primitives from Phase 1 into an evaluation-ready RAG pipeline.

## Added

- structure-aware Markdown ingestion that preserves hierarchical heading paths;
- PDF ingestion with page-level source locators;
- section-aware chunking so chunks do not cross document structure boundaries;
- canonical source IDs so evaluation remains stable when chunk boundaries change;
- cross-encoder reranking behind a small pair-scoring interface;
- bounded citation-addressable context assembly;
- citation validity and expected-source coverage metrics;
- a source-level benchmark runner with Recall@K, Hit Rate@K, Top-1 accuracy, and MRR;
- a frozen 40-question benchmark over the repository's serving, benchmarking, RAG, and security documentation;
- a CPU-only BM25 benchmark step in CI with regression floors.

## Evaluation principle

Chunk IDs are useful for exact regression tests, but they are too brittle to be the only relevance target in a long-lived RAG benchmark. The Phase 2 benchmark therefore scores canonical source identity. A heading or page locator can change without silently invalidating the relevance labels.

## Measured BM25 baseline

The reviewed baseline is stored at `evals/baselines/rag_bm25_v1.json`. It was measured in GitHub Actions on the frozen 40-question source-level benchmark at `K=5`, using 180-word chunks with 30-word overlap.

| Metric | Result |
|---|---:|
| Top-1 source accuracy | **90.0%** |
| Recall@5 | **100.0%** |
| Hit Rate@5 | **100.0%** |
| MRR | **0.9383** |
| Queries | **40** |

CI enforces a minimum Top-1 accuracy of `0.85` and minimum MRR of `0.90` so sparse retrieval regressions fail the build.

This is a controlled repository-documentation benchmark. It should not be presented as general real-world RAG accuracy.

## Retrieval ladder

The same frozen questions are intended to be measured across:

1. BM25 sparse retrieval;
2. dense retrieval with Sentence Transformers + Qdrant;
3. reciprocal-rank-fused hybrid retrieval;
4. hybrid retrieval followed by a cross-encoder reranker.

Only measured results should be promoted into a reviewed baseline file.

## Next

1. Run dense and hybrid retrieval against the same 40 frozen questions.
2. Run the cross-encoder reranker and quantify its delta over hybrid retrieval.
3. Expand the frozen set toward 50-100 questions with harder paraphrases and multi-source cases.
4. Add answer generation and semantic faithfulness evaluation on top of citation validity.
