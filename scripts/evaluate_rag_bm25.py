from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_platform.benchmark import load_cases, run_benchmark
from rag_platform.bm25 import BM25Index
from rag_platform.ingestion import chunk_sections, load_markdown

DEFAULT_SOURCES = (
    "README.md",
    "SECURITY.md",
    "docs/GPU_BENCHMARK.md",
    "docs/RAG_PHASE1.md",
)


def build_sparse_index(root: Path) -> BM25Index:
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
    return BM25Index(chunks)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the frozen RAG BM25 baseline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("evals/rag_retrieval_v1.jsonl"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cases = load_cases(args.root / args.cases)
    index = build_sparse_index(args.root)
    result = run_benchmark("bm25", index, cases, k=args.k)
    payload = result.to_dict()
    print(json.dumps(payload["metrics"], indent=2, sort_keys=True))

    if args.output:
        output_path = args.root / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
