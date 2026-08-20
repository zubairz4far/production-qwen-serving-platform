from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from .types import Chunk, RetrievalHit


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...


class SentenceTransformerEmbedder:
    """Sentence Transformers adapter kept behind a small testable interface."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        dimension = self._model.get_sentence_embedding_dimension()
        if dimension is None:
            raise RuntimeError("embedding model did not report a vector dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self._model.encode(
            list(texts),
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vectors.tolist()


class QdrantDenseIndex:
    """Dense vector index backed by Qdrant."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        collection_name: str = "rag_chunks",
        url: str = "http://localhost:6333",
    ) -> None:
        from qdrant_client import QdrantClient

        self.embedder = embedder
        self.collection_name = collection_name
        self.client = QdrantClient(url=url)

    def ensure_collection(self) -> None:
        from qdrant_client import models

        if self.client.collection_exists(self.collection_name):
            return
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedder.dimension,
                distance=models.Distance.COSINE,
            ),
        )

    def upsert(self, chunks: Sequence[Chunk]) -> None:
        from qdrant_client import models

        if not chunks:
            return
        self.ensure_collection()
        vectors = self.embedder.encode([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "metadata": chunk.metadata,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        self.client.upsert(collection_name=self.collection_name, points=points, wait=True)

    def search(self, query: str, *, limit: int = 10) -> list[RetrievalHit]:
        if limit <= 0:
            return []
        self.ensure_collection()
        vector = self.embedder.encode([query])[0]
        points = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            with_payload=True,
            limit=limit,
        ).points

        hits: list[RetrievalHit] = []
        for rank, point in enumerate(points, start=1):
            payload = point.payload or {}
            hits.append(
                RetrievalHit(
                    chunk=Chunk(
                        chunk_id=str(point.id),
                        text=str(payload.get("text", "")),
                        source=str(payload.get("source", "unknown")),
                        metadata=dict(payload.get("metadata") or {}),
                    ),
                    score=float(point.score),
                    rank=rank,
                    channel="dense",
                )
            )
        return hits
