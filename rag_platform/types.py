from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable unit with stable identity and source metadata."""

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalHit:
    """A ranked retrieval result from one or more retrieval channels."""

    chunk: Chunk
    score: float
    rank: int
    channel: str
