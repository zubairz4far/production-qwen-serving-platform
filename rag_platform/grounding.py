from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from .types import RetrievalHit

_CITATION_RE = re.compile(r"\[(S\d+)\]")


@dataclass(frozen=True, slots=True)
class ContextBundle:
    text: str
    sources: dict[str, RetrievalHit]


@dataclass(frozen=True, slots=True)
class CitationMetrics:
    citation_count: int
    valid_citation_count: int
    citation_precision: float
    expected_source_coverage: float


def assemble_context(
    hits: Sequence[RetrievalHit],
    *,
    max_chars: int = 12000,
) -> ContextBundle:
    """Build bounded, citation-addressable context for answer generation."""

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    parts: list[str] = []
    sources: dict[str, RetrievalHit] = {}
    used = 0
    for index, hit in enumerate(hits, start=1):
        label = f"S{index}"
        block = f"[{label}] source={hit.chunk.source}\n{hit.chunk.text.strip()}\n"
        if parts and used + len(block) > max_chars:
            break
        if not parts and len(block) > max_chars:
            block = block[:max_chars]
        parts.append(block)
        sources[label] = hit
        used += len(block)
        if used >= max_chars:
            break
    return ContextBundle(text="\n".join(parts), sources=sources)


def evaluate_citations(
    answer: str,
    bundle: ContextBundle,
    *,
    expected_sources: set[str] | None = None,
) -> CitationMetrics:
    from .ingestion import canonical_source

    labels = _CITATION_RE.findall(answer)
    valid = [label for label in labels if label in bundle.sources]
    precision = len(valid) / len(labels) if labels else 0.0

    coverage = 1.0
    if expected_sources is not None:
        cited_sources = {
            canonical_source(bundle.sources[label].chunk.source) for label in valid
        }
        coverage = (
            len(cited_sources.intersection(expected_sources)) / len(expected_sources)
            if expected_sources
            else 1.0
        )

    return CitationMetrics(
        citation_count=len(labels),
        valid_citation_count=len(valid),
        citation_precision=precision,
        expected_source_coverage=coverage,
    )
