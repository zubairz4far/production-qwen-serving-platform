# RAG Phase 1 — Retrieval Baseline

This milestone adds the retrieval foundation for a production RAG system.

## Implemented

- deterministic overlapping chunking with stable chunk IDs;
- transparent BM25 sparse retrieval implemented in-project;
- Qdrant dense vector adapter;
- Sentence Transformers embedding adapter;
- reciprocal-rank fusion (RRF) for hybrid retrieval;
- frozen-query retrieval metrics: Recall@K, Hit Rate@K, and MRR;
- unit tests for chunking, sparse ranking, fusion, and evaluation.

## Why this design

The project keeps policy and evaluation logic explicit rather than hiding the entire
retrieval path behind a framework. This makes ranking failures observable and lets us
compare sparse, dense, and hybrid retrieval independently.

## Next milestone

1. Add Markdown/PDF ingestion with structure-aware chunking.
2. Create a versioned corpus and 50-100 frozen retrieval questions.
3. Benchmark BM25 vs dense vs hybrid retrieval.
4. Add a cross-encoder reranker.
5. Add citation assembly and grounded-answer evaluation.
