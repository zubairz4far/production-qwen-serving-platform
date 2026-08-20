from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .chunking import chunk_text
from .types import Chunk

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True, slots=True)
class DocumentSection:
    """A structure-aware source segment before token/word chunking."""

    text: str
    source: str
    locator: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_ref(self) -> str:
        return f"{self.source}#{self.locator}" if self.locator else self.source


def parse_markdown(text: str, *, source: str) -> list[DocumentSection]:
    """Split Markdown on headings while preserving a hierarchical heading path."""

    heading_stack: list[str] = []
    sections: list[DocumentSection] = []
    buffer: list[str] = []
    current_path: list[str] = []

    def flush() -> None:
        nonlocal buffer
        body = "\n".join(buffer).strip()
        if body:
            heading = " > ".join(current_path) if current_path else None
            locator = f"heading={heading}" if heading else None
            sections.append(
                DocumentSection(
                    text=body,
                    source=source,
                    locator=locator,
                    metadata={"kind": "markdown", "heading_path": list(current_path)},
                )
            )
        buffer = []

    for line in text.splitlines():
        match = _HEADING_RE.match(line)
        if not match:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        heading_stack[:] = heading_stack[: level - 1]
        while len(heading_stack) < level - 1:
            heading_stack.append("")
        heading_stack.append(title)
        current_path = [part for part in heading_stack if part]
        buffer = [line]

    flush()
    return sections


def load_markdown(path: str | Path, *, source: str | None = None) -> list[DocumentSection]:
    file_path = Path(path)
    return parse_markdown(
        file_path.read_text(encoding="utf-8"),
        source=source or file_path.as_posix(),
    )


def load_pdf(path: str | Path, *, source: str | None = None) -> list[DocumentSection]:
    """Extract one structure-preserving section per PDF page."""

    from pypdf import PdfReader

    file_path = Path(path)
    reader = PdfReader(str(file_path))
    resolved_source = source or file_path.as_posix()
    sections: list[DocumentSection] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        sections.append(
            DocumentSection(
                text=text,
                source=resolved_source,
                locator=f"page={page_number}",
                metadata={"kind": "pdf", "page": page_number},
            )
        )
    return sections


def chunk_sections(
    sections: Iterable[DocumentSection],
    *,
    chunk_size_words: int = 220,
    overlap_words: int = 40,
) -> list[Chunk]:
    """Chunk each structural section independently and merge section metadata."""

    chunks: list[Chunk] = []
    for section in sections:
        for chunk in chunk_text(
            section.text,
            source=section.source_ref,
            chunk_size_words=chunk_size_words,
            overlap_words=overlap_words,
        ):
            chunks.append(
                Chunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    metadata={**section.metadata, **chunk.metadata},
                )
            )
    return chunks


def canonical_source(source: str) -> str:
    """Drop page/heading locators for source-level benchmark stability."""

    return source.split("#", 1)[0]
