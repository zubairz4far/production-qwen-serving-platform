# RAG Phase 3 — Measured Neural Retrieval

Phase 3 measures whether dense retrieval, hybrid fusion, and cross-encoder reranking actually improve the frozen retrieval benchmark established in Phase 2.

## Fixed evaluation target

All systems use the same:

- 40 frozen source-level questions in `evals/rag_retrieval_v1.jsonl`;
- four repository documentation sources;
- 180-word chunk size with 30-word overlap;
- `K=5` retrieval evaluation;
- Recall@5, Hit Rate@5, Top-1 source accuracy, and MRR metrics.

The reviewed sparse baseline is `evals/baselines/rag_bm25_v1.json`.

## Systems under measurement

1. **BM25** — transparent lexical baseline implemented in-project.
2. **Dense** — `sentence-transformers/all-MiniLM-L6-v2` embeddings indexed in Qdrant with cosine similarity.
3. **Hybrid** — BM25 and dense rankings combined with reciprocal-rank fusion.
4. **Hybrid + reranker** — hybrid candidates rescored by `cross-encoder/ms-marco-MiniLM-L6-v2`.

The cross-encoder is used only after first-stage retrieval. This keeps the expensive pairwise scorer on a bounded candidate set instead of scoring the full corpus.

## Reproducibility

The dedicated `RAG Model Evaluation` GitHub Actions workflow installs the optional `rag` dependency group and uses Qdrant local in-memory mode. Production serving can still point the same dense adapter at an HTTP Qdrant service.

Run locally with:

```bash
pip install -e ".[rag]"
python -m scripts.evaluate_rag_models --k 5 --rerank-candidates 20
```

## Release rule

No dense, hybrid, or reranker improvement is claimed until the workflow produces measured results on the frozen benchmark. Reviewed metrics are then committed under `evals/baselines/` with the exact model IDs and benchmark configuration.
