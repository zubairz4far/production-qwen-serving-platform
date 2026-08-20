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
- a source-level benchmark runner with Recall@K, Hit Rate@K, and MRR;
- a frozen 40-question benchmark over the repository's serving, benchmarking, RAG, and security documentation;
- a CPU-only BM25 benchmark step in CI.

## Evaluation principle

Chunk IDs are useful for exact regression tests, but they are too brittle to be the only relevance target in a long-lived RAG benchmark. The Phase 2 benchmark therefore scores canonical source identity. A heading or page locator can change without silently invalidating the relevance labels.

## Retrieval ladder

The benchmark is intended to measure the same frozen questions across:

1. BM25 sparse retrieval;
2. dense retrieval with Sentence Transformers + Qdrant;
3. reciprocal-rank-fused hybrid retrieval;
4. hybrid retrieval followed by a cross-encoder reranker.

Only measured results should be promoted into a reviewed baseline file.

## Next

1. Record the CI BM25 baseline.
2. Run dense and hybrid retrieval against the same 40 frozen questions.
3. Run the cross-encoder reranker and quantify its delta over hybrid retrieval.
4. Expand the frozen set toward 50-100 questions with harder paraphrases and multi-source cases.
5. Add answer generation and semantic faithfulness evaluation on top of citation validity.
