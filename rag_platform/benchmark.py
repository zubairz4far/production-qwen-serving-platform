from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable, Mapping

from .ingestion import canonical_source


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    query: str
    relevant_sources: frozenset[str]
    category: str = "general"


@dataclass(frozen=True, slots=True)
class BenchmarkMetrics:
    recall_at_k: float
    hit_rate_at_k: float
    mrr: float
    query_count: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    name: str
    metrics: BenchmarkMetrics
    rankings: list[list[str]]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "metrics": asdict(self.metrics),
            "rankings": self.rankings,
        }


def load_cases(path: str | Path) -> list[BenchmarkCase]:
    cases: list[BenchmarkCase] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sources = frozenset(row["relevant_sources"])
            if not sources:
                raise ValueError(f"line {line_number}: relevant_sources cannot be empty")
            cases.append(
                BenchmarkCase(
                    query=row["query"],
                    relevant_sources=sources,
                    category=row.get("category", "general"),
                )
            )
    return cases


def run_benchmark(
    name: str,
    retriever,
    cases: Iterable[BenchmarkCase],
    *,
    k: int = 5,
) -> BenchmarkResult:
    if k <= 0:
        raise ValueError("k must be positive")
    case_list = list(cases)
    recalls: list[float] = []
    hits: list[float] = []
    reciprocal_ranks: list[float] = []
    rankings: list[list[str]] = []

    for case in case_list:
        retrieved_hits = retriever.search(case.query, limit=k)
        source_ranking = [canonical_source(hit.chunk.source) for hit in retrieved_hits]
        rankings.append(source_ranking)

        unique_top: list[str] = []
        for source in source_ranking:
            if source not in unique_top:
                unique_top.append(source)
        matched = case.relevant_sources.intersection(unique_top)
        recalls.append(len(matched) / len(case.relevant_sources))
        hits.append(1.0 if matched else 0.0)
        first_rank = next(
            (
                rank
                for rank, source in enumerate(source_ranking, start=1)
                if source in case.relevant_sources
            ),
            None,
        )
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)

    metrics = BenchmarkMetrics(
        recall_at_k=mean(recalls) if recalls else 0.0,
        hit_rate_at_k=mean(hits) if hits else 0.0,
        mrr=mean(reciprocal_ranks) if reciprocal_ranks else 0.0,
        query_count=len(case_list),
    )
    return BenchmarkResult(name=name, metrics=metrics, rankings=rankings)


def compare_retrievers(
    retrievers: Mapping[str, object],
    cases: Iterable[BenchmarkCase],
    *,
    k: int = 5,
) -> dict[str, BenchmarkResult]:
    case_list = list(cases)
    return {
        name: run_benchmark(name, retriever, case_list, k=k)
        for name, retriever in retrievers.items()
    }
